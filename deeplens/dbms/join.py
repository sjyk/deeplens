from utils import get_logger

from deeplens.dbms.expressions import EqualityExpression

logger = get_logger(__name__)


class NestedLoopJoin(object):

    def __init__(self, left, right, predicate):
        logger.debug("set up the join object")
        self.left = left
        self.right = right
        self.predicate = predicate

    def read(self, args={}):
        for patch1 in self.left.read():
            for patch2 in self.left.read():
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

            for patch2 in self.left.read({'predicate': eq}):
                if self.predicate.predicate(patch1,patch2):
                    yield (patch1,patch2)
