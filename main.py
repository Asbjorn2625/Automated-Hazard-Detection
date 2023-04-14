from Libs.Hazard_labels.hazard import Hazard_labels
from Libs.Preprocess.prep import PreProcess
from Libs.images.image_loader import ImageFetcher
import cv2

image_fetcher = ImageFetcher("Mark_files")

# Create a PreProcess object
pre_processor = PreProcess(image_fetcher)

# Creating a Hazard label object
Hazards = Hazard_labels()
counter = 0
# Process the images
for sr_image, image_name in pre_processor.process_images():
    img ,results,thresh = Hazards.written_material(sr_image, image_name)
    cv2.imwrite("Original_image_%s.png" % counter,sr_image)
    cv2.imwrite("OCR_IMAGE%s.png" % counter,thresh)
    counter = counter + 1
    cv2.waitKey(1)
print(results)