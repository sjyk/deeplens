import pickle

class Select(object):

    def __init__(self, src, predicate):
        self.src = src
        self.predicate = predicate

    def read(self, args={}):
        for patch in self.src.read(args={'predicate': self.predicate}):
            if self.predicate(patch):
                yield patch


