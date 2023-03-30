import cv2
import os
import numpy as np
from pytesseract import*
from imutils.object_detection import non_max_suppression

# construct the argument parser and parse the arguments
testImages = os.listdir("%s/images/classes" % os.getcwd())
#minimum confidence of text detection
min_conf = 0.5
for image_paths in testImages:
    image = cv2.imread("%s/images/classes/%s" % (os.getcwd(), image_paths))
    # image height and width should be multiple of 32
    imgWidth = 320
    imgHeight = 320

    orig = image.copy()
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

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        roi = orig[startY-10:endY+10, startX-10:endX+10]
        if roi.shape[0] == 0 or roi.shape[1] == 0 or roi.shape[2] == 0:
            print("hej")
            break


        canny = cv2.Canny(image=roi, threshold1=100, threshold2=200)

        pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe"
        results = pytesseract.image_to_data(roi, output_type=Output.DICT)



        # Then loop over each of the individual text
        # localizations
        for i in range(0, len(results["text"])):
            # We can then extract the bounding box coordinates
            # of the text region from  the current result
            x = results["left"][i]
            y = results["top"][i]
            w = results["width"][i]
            h = results["height"][i]

            # We will also extract the OCR text itself along
            # with the confidence of the text localization
            text = results["text"][i]
            conf = int(float(results["conf"][i]))

            # filter out weak confidence text localizations
            if conf > min_conf:
                # We will display the confidence and text to
                # our terminal
                print("Confidence: {}".format(conf))
                print("Text: {}".format(text))
                print("")

        cv2.imshow("hej", canny)
        cv2.waitKey(1)