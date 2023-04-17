import cv2
import TINFW
import numpy as np

img = cv2.imread("Training/Data/Test_data/RGB/dog013.png")
H = int(700)
W = int(700)
certainty = 0.99 #float from 0 to 1
model = "TrainingModel.pth"

pred = TINFW.apply_model(img, H, W, certainty, model)
contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
cv2.drawContours(img, contours, -1, (0, 255, 255), 1)
result = cv2.bitwise_and(img, img, mask=pred)

display = np.hstack((result, img))
cv2.namedWindow('result', cv2.WINDOW_AUTOSIZE)
cv2.imshow("result", display)
cv2.waitKey(0)
cv2.imwrite("img_CutOuts4/cutOut_2.png", result)
