import sys
import os
sys.path.append(os.getcwd().replace("\\", "/") + "/src")
from Text_reader.ReaderClass import ReadText


class UNLabels:
    def __init__(self, image, depth, mask):
        self.image = image
        self.depth = depth
        self.mask = mask
    def unLogo(self):
        pass
    def packagingCodes(self):
        bounding = ReadText().findText(self.image)
        for boxes in bounding:
            ReadText.readText(self.image, boxes)
    def boxDimensions(self):
        pass