import cv2
import numpy as np
import math


class classifier():
    def __init__(self,img) -> None:
        self.img = img
        self.cnt_simple = cv2.findContours(self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
def labels_on_edge(self, Distancethreshold=3):
        # draw the contour on the original grayscale image
        moments = cv2.moments(self.cnt_simple)
        cx, cy = int(moments['m10']/moments['m00']),int(moments['m01']/moments['m00'])
        print([cx,cy])
        rect = cv2.minAreaRect(self.cnt_simple)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # calculate distance
        distance = math.sqrt((cx - int(rect[0][0]))**2 + (cy - int(rect[0][1]))**2)
        if distance >= Distancethreshold:
            label_bent = True
        else:
            label_bent = False
        return  label_bent

    