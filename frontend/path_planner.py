import cv2

def compute_path(image):

    path_image = image.copy()

    height,width,_ = image.shape

    start = (width//2,height-20)
    goal = (width//2,20)

    cv2.line(path_image,start,goal,(0,255,0),5)

    return path_image