import pickle

class UnclusteredPatchFile(object):

    def __init__(self, name, scanner, patcher, transformer, storage_path):
        self.scanner = scanner
        self.patcher = patcher
        self.transformer = transformer
        self.name = name
        self.storage_path = storage_path

    def build(self):

        #clear all previous data
        patchfile = open(self.storage_path, 'w')
        patchfile.close()

        #append
        patchfile = open(self.storage_path, 'ab')

        for im in self.scanner.scan():
            for patch in self.patcher.generatePatches(im[0],im[1]):
                tpatch = self.transformer.transform(patch)
                pickle.dump(tpatch, patchfile)

        patchfile.close()

    def read(self, args={}):
        patchfile = open(self.storage_path, 'rb')

        while(True):
            try:
                yield pickle.load(patchfile)
            except:
                break;


