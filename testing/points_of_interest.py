import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks




file_path = 'C:\\Users\\dadih\\OneDrive\\Desktop\\P6-points-of-intrests\\P6\\Automated-Hazard-Detection\\testing\image_name\\filenames.txt'
os_path = 'C:\\Users\\dadih\\OneDrive\\Desktop\\P6-points-of-intrests\\P6\\Automated-Hazard-Detection\\testing\\outline_of_hazard_labels'

with open(file_path, 'r') as file:
    lines = file.readlines()




for line in lines:
    # Process each line of text here
    print(line.strip())

    img_path = os_path + "\\" + str(line.strip()) + ".png"

    print(img_path)

    original_img = cv.imread(img_path)
    color_img = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)# convert from BGR to RGB
    hls = cv.cvtColor(color_img, cv.COLOR_RGB2HLS)

    img_gray = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)

    blur_gray = cv.GaussianBlur(img_gray, (15,15), 0)

    thresh_BGR = cv.threshold(blur_gray, 110, 255, cv.THRESH_BINARY)[1]

    thresh_gray = cv.threshold(img_gray, 160, 255, cv.THRESH_BINARY)[1]

    hls_blur = cv.GaussianBlur(hls, (19, 19), 0)

    lower_blue = np.array([100, 50, 210])
    upper_blue = np.array([150, 120, 250])
    mask = cv.inRange(hls_blur, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    cleaned_image = cv.dilate(mask, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(cleaned_image)

    num_labels2, labels2, stats2, centroids2 = cv.connectedComponentsWithStats(thresh_BGR)



    blur = cv.GaussianBlur(hls, (5, 5), 0)

    #making a sobel edge detection to detect the edges of the blurred hls image
    sobelx = cv.Sobel(blur, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(blur, cv.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    #thresholding the
    sobel = cv.threshold(sobel, 20, 255, cv.THRESH_BINARY)[1]

    #normslizing the image
    sobel = cv.convertScaleAbs(sobel)

    #converting the image to gray for histogram operations
    sobel_gray = cv.cvtColor(sobel, cv.COLOR_HLS2BGR)
    sobel_gray = cv.cvtColor(sobel, cv.COLOR_BGR2GRAY)

    # Compute the histogram of the grayscale image
    hist = cv.calcHist([sobel_gray], [0], None, [256], [0, 256])

    # Find the peaks in the histogram
    peaks, _ = find_peaks(hist.ravel(), height=0)



    # Set the threshold level to the minimum value between the two peaks
    thresh = np.min(peaks)

    #applying the histogram threshold to the image
    hist_thresh = cv.threshold(sobel_gray, thresh, 255, cv.THRESH_BINARY)[1]

    # Compute the distance transform
    dist_transform = cv.distanceTransform(hist_thresh, cv.DIST_L2, 5)

    # Threshold the distance transform
    ret, skeleton = cv.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)

    # Define the minimum and maximum component sizes (in pixels)
    min_size = 600
    max_size = 100000

    color1 = [0, 255, 0]  # green
    color2 = [255, 0, 0]  # red

    #finding components that are not noise or background
    number_of_components = 0
    # Create a mask for this component
    component_mask = np.zeros_like(cleaned_image)
    # Loop through the components
    for i in range(1, num_labels):
        # Check if the component meets the size criteria
        if max_size >= stats[i, cv.CC_STAT_AREA] >= min_size:
            if stats[i, cv.CC_STAT_WIDTH] > 5 * stats[i, cv.CC_STAT_HEIGHT]:
                continue


            print(stats[i, cv.CC_STAT_AREA])
            component_mask[labels == i] = 255

            # Set the color for this component
            color = color1 if stats[i, cv.CC_STAT_LEFT] < hist.shape[0] / 2 else color2

            # Apply the color to the component mask
            component_colored = cv.cvtColor(component_mask, cv.COLOR_GRAY2RGB)
            component_colored[component_mask > 0] = color

            # Draw the colored component on top of the original image
            color_img = cv.addWeighted(color_img, 1, component_colored, 0.5, 0)
            number_of_components +=1


        #getting rid of long lines
        for i in range(1, num_labels2):
            # Create a new mask for this component
            component_mask2 = np.zeros_like(thresh_BGR)

            # Check if the component meets the size criteria
            if max_size >= stats2[i, cv.CC_STAT_AREA] >= min_size:
                if stats2[i, cv.CC_STAT_WIDTH] > 2 * stats2[i, cv.CC_STAT_HEIGHT]:
                    continue

                component_mask2[labels2 == i] = 255

                # Set the color for this component
                color = color1 if stats2[i, cv.CC_STAT_LEFT] < hist.shape[0] / 2 else color2

                # Apply the color to the component mask
                component_colored = cv.cvtColor(component_mask2, cv.COLOR_GRAY2RGB)
                component_colored[component_mask2 > 0] = color

                # Draw the colored component on top of the original image
                color_img = cv.addWeighted(color_img, 1, component_colored, 0.5, 0)

                number_of_components += 1


   # Set the figsize parameter to (10,10) to make the plot bigger
    plt.figure(figsize=(20, 10))
    plt.subplot(221), plt.imshow(color_img, vmin=0, vmax=255)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(cleaned_image, vmin=0, vmax=255)
    plt.title('Hls Thresholding'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(skeleton, cmap='gray', vmin=0, vmax=255)
    plt.title('skeleton'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(hist_thresh, cmap="gray", vmin=0, vmax=255)
    plt.title('hist thresh'), plt.xticks([]), plt.yticks([])
    print(number_of_components)


    # Display the plot outside of the while loop
    plt.show()



"""
    cv.imshow("hls", hls)

   cv.imshow("hls_blur", clahe_img)

    cv.imshow("clahe_thres", clahe_img_thres)

    cv.imshow("coloer", adaptive_thresh)
    def display_hls(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDBLCLK:
            h, l, s = hls[y, x]
            h2, l2, s2 = hls_blur[y, x]
            print("HLS values at ({},{}): Hue={}, Lightness={}, Saturation={}".format(x, y, h, l, s))
            print("HLS values at ({},{}): Hue={}, Lightness={}, Saturation={}".format(x, y, h2, l2, s2))


    # Set a mouse callback to call the function when the mouse is double-clicked
    cv.setMouseCallback("hls", display_hls)

    cv.waitKey()
"""












