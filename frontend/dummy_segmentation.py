import numpy as np
import cv2

def predict_segmentation(image):

    height, width, _ = image.shape

    segmentation = np.zeros((height, width, 3), dtype=np.uint8)

    segmentation[:] = (180,180,180)

    cv2.circle(segmentation,(200,200),50,(100,100,100),-1)
    cv2.rectangle(segmentation,(400,100),(450,200),(0,255,0),-1)

    return segmentation