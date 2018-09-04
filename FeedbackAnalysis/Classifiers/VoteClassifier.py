import nltk
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier:  # inherit form ClassifierI

    def __init__(self, *classifiers):
        self.classifiers = classifiers
        self.neuralNetwork = None

    def SetNeuralNetwork(self, aNeuralNetwork, aText):
        self.neuralNetwork = aNeuralNetwork
        self.text = aText


    def classify(self, features):
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)

        if self.neuralNetwork != None:
            result = self.neuralNetwork.classify(self.text)

            if result[0] != None or result[0][0] != None:
                if result[0][0] == "positive":
                    votes.append("pos")
                else:
                    votes.append("neg")

        return mode(votes)


    def Accuracy(self, testingSet):

        return (nltk.classify.accuracy(self, testingSet)) * 100


    def confidence(self, features):
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)

        choise_votes = votes.count(mode(votes))
        conf = choise_votes / len(votes)

        return conf
