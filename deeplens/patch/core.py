from deeplens.io import Patch
import cv2

class PatchGenerator(object):

    def generatePatches(self, imgref, img):
        raise NotImplemented("All patch generator subclasses must implement this")


class FixedPatchGenerator(PatchGenerator):

    def __init__(self, w, h):
        self.patchw = w
        self.patchh = h

    def generatePatches(self, imgref, img):
        height, width, channels = img.shape
        for y in range(self.patchw, width, self.patchh):
            for x in range(self.patchh, height, self.patchh):
                patchData = img[x-self.patchw:x, y-self.patchh:y, :]
                yield Patch(imgref, x,y, self.patchw, self.patchh, patchData)

class NullPatchGenerator(PatchGenerator):

    def generatePatches(self, imgref, img):
        height, width, channels = img.shape
        yield Patch(imgref,0,0,width, height, img)