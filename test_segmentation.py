import torch
import numpy as np
import cv2
import os
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms


NUM_CLASSES = 10


class SegmentationHead(torch.nn.Module):

    def __init__(self,in_channels,out_channels,H,W):
        super().__init__()

        self.H = H
        self.W = W

        self.decoder = torch.nn.Sequential(

            torch.nn.Conv2d(in_channels,256,3,padding=1),
            torch.nn.GELU(),

            torch.nn.Conv2d(256,256,3,padding=1),
            torch.nn.GELU(),

            torch.nn.Conv2d(256,out_channels,1)

        )


    def forward(self,x):

        B,N,C = x.shape

        x = x.reshape(B,self.H,self.W,C).permute(0,3,1,2)

        return self.decoder(x)



def test():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    w = int(((960/2)//14)*14)
    h = int(((540/2)//14)*14)


    transform = transforms.Compose([
        transforms.Resize((h,w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])


    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14"
    ).to(device)

    backbone.eval()


    model = SegmentationHead(384,NUM_CLASSES,h//14,w//14)

    model.load_state_dict(torch.load("segmentation_head.pth"))

    model.to(device).eval()


    img = Image.open("test.png").convert("RGB")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():

        tokens = backbone.forward_features(x)["x_norm_patchtokens"]

        logits = model(tokens)

        logits = F.interpolate(logits,size=x.shape[2:],mode="bilinear")

        pred = torch.argmax(logits,1).cpu().numpy()[0]


    cv2.imwrite("prediction.png",pred)


if __name__ == "__main__":
    test()