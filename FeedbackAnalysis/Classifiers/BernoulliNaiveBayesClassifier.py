import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB


class BernoulliNaiveBayesClassifier:

    def __init__(self, classifier = SklearnClassifier(BernoulliNB())):

        self.classifier = classifier


    def GetClassifier(self):

        return self.classifier


    def Train(self, trainingSet):

        self.classifier.train(trainingSet)


    def Accuracy(self, testingSet):

        return (nltk.classify.accuracy(self.classifier, testingSet)) * 100
