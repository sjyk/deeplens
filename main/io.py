"""
This module defines the main io methods in DeepLens. Every
io method is an iterator.
"""
from os import listdir

import cv2

"""
Custom error handling class
"""


class DeepLensIOError(Exception):
    def __init__(self, message, errors={}):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # TODO: nothing added here
        self.errors = errors


"""
File scan loads a set of 
"""


class FileScan(object):
    def __init__(self, directory):

        self.directory = directory
        try:
            self.filelist = [f for f in listdir(directory)]
        except Exception:
            raise DeepLensIOError("Directory " + directory + " not found.")

    # number of files
    def size(self):
        return len(self.filelist)

    def __str__(self):
        return "FileScan(" + str(self.filelist) + ")"

    # returns an iterator over numpy arrays
    def scan(self, flags=cv2.IMREAD_COLOR):
        for filename in self.filelist:
            image = ImageRef(self.directory, filename, {'time': 1})
            yield (image, image.fetch())


class VideoScan(object):
    def __init__(self, file, fps=24, resize=0.1, sampling=20):
        self.file = file
        self.fps = fps
        self.resize = resize
        self.sampling = sampling

    def __str__(self):
        return "VideoScan(" + str(self.file) + ")"

    # returns an iterator over numpy arrays
    def scan(self, flags=cv2.IMREAD_COLOR):
        cap = cv2.VideoCapture(self.file)
        
        count = 1

        while(cap.isOpened()):
            ret, frame = cap.read()

            w,h = (0,0)
            try:
                w, h,_ = frame.shape
            except:
                break

            frame = cv2.resize(frame, (int(h*self.resize), int(w*self.resize)), interpolation=cv2.INTER_CUBIC)
            image = ImageRef(self.file, count, {'time': count})
            count+=1

            if count % self.sampling == 0:
                yield (image, frame)

        cap.release()


"""
Represents a reference to an image, can be fetched from disk 
"""


class ImageRef(object):

    def __init__(self, directory, filename, metadata={}):
        self.directory = directory
        self.filename = filename
        self.metadata = metadata

    def fetch(self):
        return cv2.imread(self.directory + "/" + self.filename)

    def __str__(self):
        return str((self.directory, self.filename))


"""
A description for the patch object
"""


class Patch(object):

    def __init__(self, imgref, x, y, w, h, patch, metadata={}):
        self.imgref = imgref
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.patch = patch
        self.metadata = metadata
        self.metadata.update(imgref.metadata)
        self.metadata['filename'] = imgref.filename

    # resizes a patch to a given size
    def resizeTo(self, tw, th):
        return cv2.resize(self.patch, (tw, th), interpolation=cv2.INTER_CUBIC)

    def __str__(self):
        return "Patch(" + str(self.patch.shape) + ") => " + str(self.imgref)
