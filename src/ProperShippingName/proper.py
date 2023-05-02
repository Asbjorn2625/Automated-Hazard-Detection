from UN_labels.UN import UN
import sys
import os
sys.path.append(os.getcwd().replace("\\", "/") + "/src")
from Text_reader.ReaderClass import ReadText
class ProperShippingName():
    def __init__(self):
        self.reader = ReadText()
    def locateProperShipping(self,image):
        text= []
        bounding = self.reader.findText(image)
        config='-c tessedit_char_whitelist=UN0123456789./- --psm 6 --oem 3'
        for boxes in bounding:
            text.append(self.reader.readText(image, boxes,display=True,config=config))
        return text
    