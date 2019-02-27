import pickle

from bsddb3 import db

import logging
from main.dbms.expressions import *

logger = logging.getLogger(name=__name__)


class UnclusteredPatchFile(object):

    def __init__(self, name, scanner, patcher, transformer, storage_path):
        self.scanner = scanner
        self.patcher = patcher
        self.transformer = transformer
        self.name = name
        self.storage_path = storage_path

    def build(self):
        bdb = db.DB()
        bdb.open(self.name, None, db.DB_RECNO, db.DB_CREATE)

        count = 1
        for im in self.scanner.scan():
            for patch in self.patcher.generatePatches(im[0], im[1]):
                tpatch = self.transformer.transform(patch)
                bdb.put(count, pickle.dumps(tpatch))
                count += 1
        logger.debug("count: {}".format(count))
        bdb.close()

    def read(self, args={}):
        bdb = db.DB()
        bdb.open(self.name, None, db.DB_RECNO, db.DB_DIRTY_READ)

        # get database cursor and print out database content
        cursor = bdb.cursor()
        rec = cursor.first()
        while rec:
            yield pickle.loads(rec[1])
            rec = cursor.next()

        bdb.close()


class HashIndex(object):

    def __init__(self, src, srcName, attr):
        self.src = src
        self.name = srcName
        self.attr = attr

    def build(self):

        bdb = db.DB()
        bdb.open(self.name, None, db.DB_HASH, db.DB_CREATE)

        keyct = {}

        for tpatch in self.src.read():

            # print(tpatch)

            key = tpatch.metadata[self.attr]

            if key not in keyct:
                keyct[key] = 0

            bdb.put(str.encode(key + str(keyct[key])), pickle.dumps(tpatch))
            keyct[key] += 1

    def read(self, args={}):

        if not isinstance(args['predicate'], EqualityExpression) or args[
            'predicate'].attr != self.attr:
            for i in self.readExhaustive():
                yield i

        bdb = db.DB()
        bdb.open(self.name, None, db.DB_HASH, db.DB_DIRTY_READ)

        count = 0
        while True:
            key = str.encode(args['predicate'].value + str(count))
            rec = bdb.get(key)

            if rec is None:
                break

            yield pickle.loads(rec)
            count += 1

        bdb.close()

    def readExhaustive(self, args={}):
        bdb = db.DB()
        bdb.open(self.name, None, db.DB_RECNO, db.DB_DIRTY_READ)

        # get database cursor and print out database content
        cursor = bdb.cursor()
        rec = cursor.first()
        while rec:
            yield pickle.loads(rec[1])
            rec = cursor.next()

        bdb.close()


class BTreeIndex(object):

    def __init__(self, src, srcName, attr):
        self.src = src
        self.name = srcName
        self.attr = attr

    def build(self):

        bdb = db.DB()
        bdb.open(self.name, None, db.DB_BTREE, db.DB_CREATE)

        keyct = {}

        for tpatch in self.src.read():

            # print(tpatch)

            key = tpatch.metadata[self.attr]

            if key not in keyct:
                keyct[key] = 0

            bdb.put(str.encode(key + str(keyct[key])), pickle.dumps(tpatch))
            keyct[key] += 1

    def read(self, args={}):

        if not isinstance(args['predicate'], EqualityExpression) or args[
            'predicate'].attr != self.attr:
            for i in self.readExhaustive():
                yield i

        bdb = db.DB()
        bdb.open(self.name, None, db.DB_BTREE, db.DB_DIRTY_READ)

        count = 0
        while True:
            key = str.encode(args['predicate'].value + str(count))
            rec = bdb.get(key)

            if rec == None:
                break

            yield pickle.loads(rec)
            count += 1

        bdb.close()

    def readExhaustive(self, args={}):
        bdb = db.DB()
        bdb.open(self.name, None, db.DB_RECNO, db.DB_DIRTY_READ)

        # get database cursor and print out database content
        cursor = bdb.cursor()
        rec = cursor.first()
        while rec:
            yield pickle.loads(rec[1])
            rec = cursor.next()

        bdb.close()


class FrameIndex(object):

    def __init__(self, src, srcName, attr):
        self.src = src
        self.name = srcName
        self.attr = attr

    def build(self):

        bdb = db.DB()
        bdb.open(self.name, None, db.DB_RECNO, db.DB_CREATE)

        for tpatch in self.src.read():
            key = tpatch.metadata[self.attr]

            bdb.put(key, pickle.dumps(tpatch))

    def read(self, args={}):

        if not isinstance(args['predicate'], RangeExpression) or args[
            'predicate'].attr != self.attr:
            for i in self.readExhaustive():
                yield i

        bdb = db.DB()
        bdb.open(self.name, None, db.DB_RECNO, db.DB_DIRTY_READ)

        count = max(args['predicate'].start, 1)
        while count < args['predicate'].end:
            rec = bdb.get(count)

            if rec == None:
                break

            yield pickle.loads(rec)
            count += 1

        bdb.close()

    def readExhaustive(self, args={}):
        bdb = db.DB()
        bdb.open(self.name, None, db.DB_RECNO, db.DB_DIRTY_READ)

        # get database cursor and print out database content
        cursor = bdb.cursor()
        rec = cursor.first()
        while rec:
            yield pickle.loads(rec[1])
            rec = cursor.next()

        bdb.close()
