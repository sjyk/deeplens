from deeplens.io import Patch
import cv2
from skimage.feature import hog

"""
Extends the patch class to handle
transformations.
"""
class TransformedPatch(Patch):

    def __init__(self, imgref, x, y, w, h, patch, metadata={}, supportsMatching=False, featurized=False):
        super(TransformedPatch, self).__init__(imgref, x, y, w, h, patch, metadata)


class Transformer(object):

    def transform(self, patch):
        raise NotImplemented("Every transformer must implement this")



class NullTransformer(Transformer):

    def transform(self, patch):
        return TransformedPatch(patch.imgref, patch.x,patch.y, patch.w, patch.h, patch.patch, patch.metadata)


class HoGTransformer(Transformer):

    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def transform(self, patch):
        hogfeatures = hog(patch.patch, self.orientations, self.pixels_per_cell, self.cells_per_block)
        return TransformedPatch(patch.imgref, patch.x,patch.y, patch.w, patch.h, hogfeatures, patch.metadata, True, True)

