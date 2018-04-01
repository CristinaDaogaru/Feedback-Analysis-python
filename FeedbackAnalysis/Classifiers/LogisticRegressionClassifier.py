import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression


class LogisticRegressionClassifier:

    def __init__(self):

        self.classifier = SklearnClassifier(LogisticRegression())


    def GetClassifier(self):

        return self.classifier


    def Train(self, trainingSet):

        self.classifier.train(trainingSet)


    def Accuracy(self, testingSet):

        return (nltk.classify.accuracy(self.classifier, testingSet)) * 100
