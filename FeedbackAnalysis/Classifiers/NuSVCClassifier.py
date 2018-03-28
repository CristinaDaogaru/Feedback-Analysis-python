import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import NuSVC


class NuSVCClassifier:

    def __init__(self):

        self.classifier = SklearnClassifier(NuSVC())


    def Train(self, trainingSet):

        self.classifier.train(trainingSet)


    def Accuracy(self, testingSet):

        return (nltk.classify.accuracy(self.classifier, testingSet)) * 100
