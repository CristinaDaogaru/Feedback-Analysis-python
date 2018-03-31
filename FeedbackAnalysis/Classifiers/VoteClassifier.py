import nltk
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):  # inherit form ClassifierI

    def __init__(self, *classifiers):
        self.classifiers = classifiers


    def classify(self, features):
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)

        return mode(votes)


    def Accuracy(self, testingSet):

        return (nltk.classify.accuracy(self, testingSet)) * 100


    def confidence(self, features):
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)

        choise_votes = votes.count(mode(votes))
        conf  = choise_votes / len(votes)

        return conf