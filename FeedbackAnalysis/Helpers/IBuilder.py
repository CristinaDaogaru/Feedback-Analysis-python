import abc


class IBuilder:

    def __init__(self, positiveFilePath, negativeFilePath):
        self.positiveTrainingData = []
        self.negativeTrainingData = []
        self.positiveFilePath = positiveFilePath
        self.negativeFilePath = negativeFilePath

    @abc.abstractmethod
    def Build(self):
        return;
