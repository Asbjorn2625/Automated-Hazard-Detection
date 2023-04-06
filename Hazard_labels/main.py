from pytesseract import *
import cv2
import numpy as np
import math
from imutils.object_detection import non_max_suppression

def is_only_spaces(text):
    for char in text:
        if not char.isspace():
            return False
    return True


class Hazard_labels:
    def __init__(self, image_stream):
        self.image_folder = image_stream
    def written_material(self,min_conf_EAST=0.5,min_conf_tesseract=0):
        import os
        # construct the argument parser and parse the arguments
        testImages = os.listdir("%s/%s" % (os.getcwd().replace("\\","/"), self.image_folder))
        # minimum confidence of text detection
        results = {"image": []}
        for image_paths in testImages:
            image = cv2.imread("%s/%s/%s" % (os.getcwd(),self.image_folder, image_paths))
            procesed = self.preProces(image)
            ROIs = self.EAST(procesed, min_conf_EAST)
            for ROI in ROIs:
                textarea = procesed[ROI[1]:ROI[3], ROI[0]:ROI[2]]
                OCR = self.tesseract(textarea, min_conf_tesseract)
                cv2.imshow("ROI", textarea)
                key = cv2.waitKey(1)
            results["image"].append([image_paths, OCR])
        return results


    def preProces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        ret3, th3 = cv2.threshold(gray, 0, 220, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

        # Apply Canny edge detection with threshold values of 100 and 200
        edges = cv2.Canny(th3, 100, 200)

        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, 50, 10)

        # Loop through each pair of lines detected by Hough transform
        # Identify pairs of intersecting lines with approximately the same distance and angle
        diamond_candidates = []
        if lines is not None:
            for i in range(0, len(lines)):
                line1 = lines[i][0]
                for j in range(i + 1, len(lines)):
                    line2 = lines[j][0]
                    # calculate slopes and y-intercepts of each line
                    if (line1[2] - line1[0]) != 0:
                        slope1 = (line1[3] - line1[1]) / (line1[2] - line1[0])
                    else:
                        slope1 = float('inf')
                    intercept1 = line1[1] - slope1 * line1[0]
                    if (line2[2] - line2[0]) != 0:
                        slope2 = (line2[3] - line2[1]) / (line2[2] - line2[0])
                    else:
                        slope2 = float('inf')
                    intercept2 = line2[1] - slope2 * line2[0]
                    # calculate intersection point
                    if not (np.arctan(slope1) != -np.pi / 4 and np.arctan(slope1) != np.pi / 4) and slope1 != slope2:
                        if slope1 - slope2 != 0 and not (
                                math.isnan(slope1) or math.isinf(slope1) or math.isnan(slope2) or math.isinf(slope2)):
                            x = (intercept2 - intercept1) / (slope1 - slope2)
                            y = slope1 * x + intercept1
                            diamond_candidates.append([line1[0], line1[1], int(x), int(y)])
        for lines in diamond_candidates:
            cv2.line(edges, (lines[0], lines[1]), (lines[2], lines[3]), (0, 255, 0), 10)

        # Define the structuring elements for dilation and erosion
        # Apply morphological operations to remove noise and fill gaps
        dilation_kernel = np.ones((3, 3), np.uint8)
        morph_img = cv2.dilate(edges, dilation_kernel, iterations=1)
        erosion_kernel = np.ones((2, 2), np.uint8)
        morph_img = cv2.erode(morph_img, erosion_kernel, iterations=2)

        # Find contours of the edges
        contours, hierarchy = cv2.findContours(morph_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Create a black image to use as a mask for drawing filled contours
        mask = np.zeros_like(morph_img)

        for cnt in contours:
            cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
        # Apply the mask to the original image to extract the filled letters
        letters = cv2.bitwise_xor(morph_img, mask)

        # Iterate through all the contours and filter out the non-convex ones
        image = cv2.cvtColor(letters, cv2.COLOR_GRAY2BGR)
        return image


    def EAST(self, image, min_conf):
        # image height and width should be multiple of 32
        imgWidth = 32*10
        imgHeight = 32*10

        (H, W) = image.shape[:2]
        (newW, newH) = (imgWidth, imgHeight)

        rW = W / float(newW)
        rH = H / float(newH)
        image = cv2.resize(image, (newW, newH))

        (H, W) = image.shape[:2]
        net = cv2.dnn.readNet("frozen_east_text_detection.pb")
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)

        outputLayers = []
        outputLayers.append("feature_fusion/Conv_7/Sigmoid")
        outputLayers.append("feature_fusion/concat_3")

        net.setInput(blob)
        output = net.forward(outputLayers)
        scores = output[0]
        geometry = output[1]

        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < min_conf:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        rois = []
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            if not ((startX - endX) == 0 or (startY - endY) == 0):
                rois.append([startX, startY, endX, endY])
        return rois


    def tesseract(self, image, min_conf):

        preds = {"predictions" : [], "Confidences" : []}
        pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"
        results = pytesseract.image_to_data(image, lang="eng", output_type=Output.DICT)

        # Then loop over each of the individual text
        # localizations
        for i in range(0, len(results["text"])):
            # We will also extract the OCR text itself along
            # with the confidence of the text localization
            text = results["text"][i]
            conf = int(float(results["conf"][i]))
            # filter out weak confidence text localizations
            if conf > min_conf:
                if not (is_only_spaces(text)):
                    # We will display the confidence and text to
                    # our terminal
                    preds["predictions"].append(text)
                    preds["Confidences"].append(conf)
        return preds

test = Hazard_labels("images/classes")
results = test.written_material()
print(results)
