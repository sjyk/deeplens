"""
The expression class provides wrappers for predicates
"""
import numpy as np

class UDFExpression(object):

    def __init__(self, predicate):
        self.predicate = predicate


class RangeExpression(UDFExpression):

    def __init__(self, attr, start, end):
        self.attr = attr
        self.start = start
        self.end = end
        predicate = lambda patch: patch.metadata[attr] >= start and patch.metadata[attr] < end
        super(RangeExpression, self).__init__(predicate)


class EqualityExpression(UDFExpression):

    def __init__(self, attr, value):
        self.attr = attr
        self.value = value
        predicate = lambda patch: patch.metadata[attr] == value
        super(EqualityExpression, self).__init__(predicate)


class JoinEqualityExpression(UDFExpression):

    def __init__(self, attr1, attr2):
        self.attr1 = attr1
        self.attr2 = attr2
        predicate = lambda patch1, patch2: patch1.metadata[attr1] == patch2.metadata[attr2]
        super(JoinEqualityExpression, self).__init__(predicate)


class ImageMatchExpression(UDFExpression):

    def __init__(self, thresh):
        self.thresh = thresh
        predicate = lambda patch1, patch2: self.norm(patch1.patch, patch2.patch) < thresh
        super(ImageMatchExpression, self).__init__(predicate)

    def norm(self, i1, i2):
        f1 = np.ndarray.flatten(i1)
        f2 = np.ndarray.flatten(i2)
        return np.linalg.norm(f1-f2)

