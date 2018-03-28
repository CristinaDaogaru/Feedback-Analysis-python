import pickle

class PickleHandler:

    def Save(self, classfier, fileName, format):

        saveClassifierFile = open(fileName, format)
        pickle.dump(classfier, saveClassifierFile)
        saveClassifierFile.close()

    def Load(self, fileName, format):

        loadClassifierFile = open(fileName, format)
        classifier = pickle.load(loadClassifierFile)
        loadClassifierFile.close()
        return classifier







