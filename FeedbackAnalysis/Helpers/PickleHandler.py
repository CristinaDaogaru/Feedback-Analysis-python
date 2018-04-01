import pickle

class PickleHandler:

    def Save(self, savedData, fileName, format):

        saveFile = open(fileName, format)
        pickle.dump(savedData, saveFile)
        saveFile.close()


    def Load(self, fileName, format):

        loadFile = open(fileName, format)
        loadedData = pickle.load(loadFile)
        loadFile.close()
        return loadedData







