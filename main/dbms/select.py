from main.utils import get_logger

logger = get_logger(__name__)


class Select(object):

    def __init__(self, src, predicate):
        logger.debug("set up the select object")
        self.src = src
        self.predicate = predicate

    def read(self, args={}):
        for patch in self.src.read(args={'predicate': self.predicate}):
            if self.predicate.predicate(patch):
                yield patch
