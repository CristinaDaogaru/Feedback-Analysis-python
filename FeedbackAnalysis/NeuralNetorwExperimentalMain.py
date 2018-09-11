import random
import nltk

from Classifiers.MultinomialNaiveBayesClassifier import MultinomialNaiveBayesClassifier
from Classifiers.NeuralNetwork import NeuralNetwork
from FeatureSet import FeatureSet
from Helpers.FileReader import FileReader
from Sentiment import Sentiment
from WordTokenizer import WordTokenizer
from Helpers.CollectionIntervalSplitter import CollectionIntervalSplitter
from Helpers.DocumentHandler import DocumentHandler
from Helpers.WordsHandler import WordsHandler
from Helpers.PickleHandler import PickleHandler

from Classifiers.NltKNaiveBayesClassifier import NltKNaiveBayesClassifier
from Classifiers.MultinomialNaiveBayesClassifier import MultinomialNaiveBayesClassifier
from Classifiers.VoteClassifier import VoteClassifier
from Classifiers.SGDClassifier import SGDClassifier
from Classifiers.BernoulliNaiveBayesClassifier import BernoulliNaiveBayesClassifier
from Classifiers.LinearSVCClassifier import LinearSVCClassifier
from Classifiers.NuSVCClassifier import NuSVCClassifier
from Classifiers.LogisticRegressionClassifier import LogisticRegressionClassifier


def RunExperiments(dataLength, numberOfIterations):


    shortPositiveReviewsPath = "short_reviews/positive.txt"
    shortNegativeReviewsPath = "short_reviews/negative.txt"


    fileReader = FileReader()

    positiveReviews = fileReader.ReadToEnd(shortPositiveReviewsPath, "r")
    negativeReviews = fileReader.ReadToEnd(shortNegativeReviewsPath, "r")


    documentHandler = DocumentHandler()
    positiveDocumentsReview = documentHandler.GetPositiveDocumets(positiveReviews)
    negativeDocumentsReview = documentHandler.GetNegativeDocumets(negativeReviews)


    documents = []
    documents = documents + positiveDocumentsReview
    documents = documents + negativeDocumentsReview


    random.shuffle(documents)

    # We'll train against the first dataLength featureset
    # and test against the second dataLength featureset

    neuralNetwork = NeuralNetwork()
    neuralNetwork.TrainNetwork(dataLength, addPositiveNegativeWords=False, hidden_neurons=10, alpha=0.1, epochs=1000, dropout=False, dropout_percent=0.2)

    accurasyList = []

    while numberOfIterations > 0:

        print("\n\nIteration %s\n\n" % numberOfIterations)
        numberOfIterations = numberOfIterations - 1

        random.shuffle(documents)

        collectionIntervalSplitter = CollectionIntervalSplitter()
        testingSet = collectionIntervalSplitter.FirstElements(documents, dataLength)


        correctAnswersCount = 0
        wrongAnswersCount = 0


        for (review, type) in testingSet:

            result = neuralNetwork.classify(review)
            if result != None and result[0] != None and result[0][0] != None:

                if (result[0][0] == "positive" and type == "pos") or (result[0][0] == "negative" and type == "neg"):
                    correctAnswersCount = correctAnswersCount + 1
                else:
                    wrongAnswersCount = wrongAnswersCount + 1


        print("\nCorrect answers: ", correctAnswersCount)
        print("wrong answers: ", wrongAnswersCount)


        accurasy = (100 * correctAnswersCount)  / dataLength
        print("Accuracy: %s\n" % accurasy)
        accurasyList.append(accurasy)


    print("\nNeural Network accuracy: %s\n" % (sum(accurasyList) / float(len(accurasyList))))



def main():

    RunExperiments(dataLength=1000, numberOfIterations=100)




if __name__ == '__main__':
    main()





