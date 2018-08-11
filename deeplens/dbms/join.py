from utils import get_logger

from deeplens.dbms.expressions import EqualityExpression

from sklearn.neighbors import BallTree

logger = get_logger(__name__)


class NestedLoopJoin(object):

    def __init__(self, left, right, predicate):
        logger.debug("set up the join object")
        self.left = left
        self.right = right
        self.predicate = predicate

    def read(self, args={}):
        for patch1 in self.left.read():
            for patch2 in self.right.read():
                if self.predicate.predicate(patch1,patch2):
                    yield (patch1,patch2)


class IndexLoopJoin(object):

    def __init__(self, left, right, predicate):
        logger.debug("set up the join object")
        self.left = left
        self.right = right
        self.predicate = predicate

    def read(self, args={}):
        for patch1 in self.left.read():

            value = patch1.metadata[self.predicate.attr1]
            eq = EqualityExpression(self.predicate.attr2,value)

            for patch2 in self.right.read({'predicate': eq}):
                if self.predicate.predicate(patch1,patch2):
                    yield (patch1,patch2)


class MemorySpatialJoin(object):

    def __init__(self, left, right, predicate, leaf_size=40):
        logger.debug("set up the join object")
        self.left = left
        self.right = right
        self.predicate = predicate
        self.leaf_size = leaf_size

    def read(self, args={}):

        X = []

        for patch in self.right.read():
            X.append(patch.patch)

        Y = np.array(X)
        tree = BallTree(Y, leaf_size)

        for patch in self.left.read():
            for ind in tree.query_radius(patch.patch, predicate.thresh):
                yield Y[i]
            
