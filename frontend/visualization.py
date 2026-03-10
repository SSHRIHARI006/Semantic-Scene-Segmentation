import cv2

def overlay_segmentation(image,segmentation):

    overlay = cv2.addWeighted(image,0.6,segmentation,0.4,0)

    return overlay