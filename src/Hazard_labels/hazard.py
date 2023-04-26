import cv2
from textblob import TextBlob
import numpy as np
import math
from src.Segmentation.segmentation import Segmentation
from src.Text_reader.Read import ReadText

def is_only_spaces(text):
    for char in text:
        if not char.isspace():
            return False
    return True


class Hazard_labels:
    def __init__(self):
        # Move results initialization to constructor
        self.results = {"OCR_dic": {}}
        
    def written_material(self, image_stream, image_name, min_conf_tesseract=0):
        processed , th3 = self.preProces(image_stream)
        OCR = self.tesseract(processed, min_conf_tesseract)

        # Use image_name parameter instead of self.image_name
        if image_name in self.results["OCR_dic"]:
            self.results["OCR_dic"][image_name]["detectedWord"].update(OCR)
        else:
            self.results["OCR_dic"][image_name] = {"detectedWord": OCR}
        return processed, self.results ,th3

    def preProces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        ret3, th3 = cv2.threshold(gray, 0, 220, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

        # Apply Canny edge detection with threshold values of 100 and 200
        edges = cv2.Canny(th3, 100, 200)

        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, 50, 10)
        
        lines_mask = cv2.cvtColor(np.zeros_like(edges), cv2.COLOR_GRAY2BGR)

        if lines is not None:
            for i in range(0, len(lines)):
                l = lines[i][0]
                cv2.line(lines_mask, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)


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
            cv2.line(edges, (lines[0], lines[1]), (lines[2], lines[3]), (0, 255, 0), 8)

        # Define the structuring elements for dilation and erosion
        # Apply morphological operations to remove noise and fill gaps
        dilation_kernel = np.ones((3, 3), np.uint8)
        morph_img = cv2.dilate(edges, dilation_kernel, iterations=1)
        erosion_kernel = np.ones((2, 2), np.uint8)
        morph_img = cv2.erode(morph_img, erosion_kernel, iterations=2)


        # Find contours of the edges
        contours, hierarchy = cv2.findContours(morph_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Create a black image to use as a mask for drawing filled contours
        mask = np.zeros_like(morph_img)
        mask1 =np.zeros_like(morph_img)

        for cnt in contours:
            cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
            #cv2.drawContours(original, [cnt], -1, (255, 0, 255), 3)
        # Apply the mask to the original image to extract the filled letters
        mask = cv2.bitwise_xor(morph_img, mask)
        mask = cv2.erode(mask, erosion_kernel)
        # To remove the inner contours of letters O, P, R , A etc.
        contours1, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Loop through the hierarchy
        # Loop over all the contours
        # Loop over all the contours
        for i, cnt in enumerate(contours1):
            #cv2.drawContours(original, [cnt], 0, (0, 255, 0), cv2.FILLED)
            # Get the hierarchy of the contour
            #if i < len(hierarchy[0]):
            h = hierarchy[0][i]
            # Check if the contour is a letter
            if h[3] == -1:
                curr_hier = i
                # Find the holes inside the letter
                for j in range(len(contours1)):
                    if hierarchy[0][j][3] == curr_hier:
                        hole = contours1[j]
                        # Fill the hole
                        cv2.drawContours(mask1, [hole], 0, (255, 255, 255), cv2.FILLED)
        mask = mask
        mask1 = cv2.bitwise_not(mask1)

        letters = cv2.bitwise_and(mask, mask1)
        # Iterate through all the contours and filter out the non-convex ones
        image = cv2.cvtColor(letters, cv2.COLOR_GRAY2BGR)
        return image ,lines_mask


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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
                    blob = TextBlob(text)
                    preds["predictions"].append(blob.correct())
                    preds["Confidences"].append(conf)
        return preds