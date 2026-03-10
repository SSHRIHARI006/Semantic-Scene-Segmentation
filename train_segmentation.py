import torch
import numpy as np
import os
import cv2
from PIL import Image
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from torch.cuda.amp import autocast, GradScaler


############################################
# GPU CHECK
############################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: Running on CPU (very slow)")


############################################
# DATASET
############################################

value_map = {
    0:0,100:1,200:2,300:3,500:4,
    550:5,700:6,800:7,7100:8,10000:9
}

NUM_CLASSES = len(value_map)


def convert_mask(mask):

    arr = np.array(mask)

    new_arr = np.zeros_like(arr, dtype=np.uint8)

    for k,v in value_map.items():
        new_arr[arr==k] = v

    return Image.fromarray(new_arr)


class MaskDataset(Dataset):

    def __init__(self,data_dir,transform=None,mask_transform=None):

        self.img_dir = os.path.join(data_dir,"Color_Images")
        self.mask_dir = os.path.join(data_dir,"Segmentation")

        self.files = os.listdir(self.img_dir)

        self.transform = transform
        self.mask_transform = mask_transform


    def __len__(self):
        return len(self.files)


    def __getitem__(self,idx):

        name = self.files[idx]

        img = Image.open(os.path.join(self.img_dir,name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir,name))

        mask = convert_mask(mask)

        if self.transform:

            img = self.transform(img)
            mask = self.mask_transform(mask) * 255

        return img,mask


############################################
# MODEL
############################################

class SegmentationHead(nn.Module):

    def __init__(self,in_channels,out_channels,H,W):

        super().__init__()

        self.H = H
        self.W = W

        self.decoder = nn.Sequential(

            nn.Conv2d(in_channels,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256,out_channels,1)

        )


    def forward(self,x):

        B,N,C = x.shape

        x = x.reshape(B,self.H,self.W,C).permute(0,3,1,2)

        return self.decoder(x)


############################################
# DICE LOSS
############################################

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,pred,target):

        pred = F.softmax(pred,dim=1)

        target_onehot = F.one_hot(target,NUM_CLASSES).permute(0,3,1,2).float()

        intersection = (pred * target_onehot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))

        dice = (2*intersection + 1e-6)/(union + 1e-6)

        return 1 - dice.mean()


############################################
# TRAINING FUNCTION
############################################

def train():

    batch_size = 4
    lr = 1e-4
    epochs = 20

    w = int(((960/2)//14)*14)
    h = int(((540/2)//14)*14)


    ############################################
    # TRANSFORMS
    ############################################

    transform = transforms.Compose([
        transforms.Resize((h,w)),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h,w)),
        transforms.ToTensor()
    ])


    ############################################
    # DATASET PATHS
    ############################################

    trainset = MaskDataset(
        "Offroad_Segmentation_Training_Dataset/train",
        transform,mask_transform)

    valset = MaskDataset(
        "Offroad_Segmentation_Training_Dataset/val",
        transform,mask_transform)


    ############################################
    # FAST DATA LOADERS
    ############################################

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    ############################################
    # LOAD DINOv2 BACKBONE
    ############################################

    print("Loading DINOv2 backbone...")

    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14"
    )

    backbone.eval().to(device)


    ############################################
    # GET EMBEDDING DIM
    ############################################

    img,_ = next(iter(train_loader))

    img = img.to(device)

    with torch.no_grad():

        out = backbone.forward_features(img)["x_norm_patchtokens"]

    embed_dim = out.shape[2]


    ############################################
    # SEGMENTATION HEAD
    ############################################

    model = SegmentationHead(
        embed_dim,
        NUM_CLASSES,
        h//14,
        w//14
    ).to(device)


    ############################################
    # LOSS + OPTIMIZER
    ############################################

    ce = nn.CrossEntropyLoss()
    dice = DiceLoss()

    optimizer = optim.AdamW(model.parameters(),lr=lr)

    scaler = GradScaler()


    ############################################
    # TRAINING LOOP
    ############################################

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        loop = tqdm(train_loader)

        for img,mask in loop:

            img = img.to(device,non_blocking=True)
            mask = mask.squeeze(1).long().to(device,non_blocking=True)

            with torch.no_grad():

                tokens = backbone.forward_features(img)["x_norm_patchtokens"]

            with autocast():

                logits = model(tokens)

                logits = F.interpolate(
                    logits,
                    size=img.shape[2:],
                    mode="bilinear",
                    align_corners=False
                )

                loss = ce(logits,mask) + dice(logits,mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            total_loss += loss.item()

            loop.set_description(f"Epoch {epoch+1}/{epochs}")
            loop.set_postfix(loss=loss.item())

        print("Epoch loss:", total_loss/len(train_loader))


    ############################################
    # SAVE MODEL
    ############################################

    torch.save(model.state_dict(),"segmentation_head.pth")

    print("Model saved: segmentation_head.pth")


if __name__ == "__main__":
    train()