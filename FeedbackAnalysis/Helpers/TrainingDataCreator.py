import random
from Helpers.FileReader import FileReader


class TrainingDataCreator:

    @staticmethod
    def Create(filePath, reviewClass, numberOfElements):
        trainingData = []
        fileReader = FileReader()
        positiveWords = fileReader.ReadLines(filePath)

        random.shuffle(positiveWords)

        for index in range(0, numberOfElements):
            trainingData.append({"class": reviewClass, "sentence": positiveWords[index]})

        return trainingData
