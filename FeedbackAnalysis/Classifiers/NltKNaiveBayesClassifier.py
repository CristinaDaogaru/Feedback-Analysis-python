import nltk
from nltk.classify import ClassifierI


class NltKNaiveBayesClassifier(ClassifierI):

    def __init__(self, classifier = None):

        self.classifier = classifier

    def GetClassifier(self):

        return self.classifier


    def Train(self, trainingSet):

        self.classifier = nltk.NaiveBayesClassifier.train(trainingSet)


    def Accuracy(self, testingSet):

        return (nltk.classify.accuracy(self.classifier, testingSet)) * 100

    def ShowMostInformativeFeatures(self, number):

        return self.classifier.show_most_informative_features(number)
