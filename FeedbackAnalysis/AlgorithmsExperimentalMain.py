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


def RunExperiments(dataLength):


    avrageResult = {"NaiveBayes": 0, "MultinomialNaiveBayes": 0, "BernouliNaiveBayes": 0, "LogisticRegression": 0,
                    "SGD": 0, "LinearSVM": 0, "NuSVC": 0}

    shortPositiveReviewsPath = "short_reviews/positive.txt"
    shortNegativeReviewsPath = "short_reviews/negative.txt"

    #positiveWordsPath = "PositiveNegativeWords/positive-words.txt"
    #negativeWordsPath = "PositiveNegativeWords/negative-words.txt"

    fileReader = FileReader()

    positiveReviews = fileReader.ReadToEnd(shortPositiveReviewsPath, "r")
    negativeReviews = fileReader.ReadToEnd(shortNegativeReviewsPath, "r")

    #positiveWords = fileReader.ReadToEnd(positiveWordsPath, "r")
    #negativeWords = fileReader.ReadToEnd(negativeWordsPath, "r")

    documentHandler = DocumentHandler()

    positiveDocumentsReview = documentHandler.GetPositiveDocumets(positiveReviews)
    #positiveDocumentsReview += documentHandler.GetPositiveDocumets(positiveWords)

    negativeDocumentsReview = documentHandler.GetNegativeDocumets(negativeReviews)
    #negativeDocumentsReview += documentHandler.GetNegativeDocumets(negativeWords)

    documents = []
    documents = documents + positiveDocumentsReview
    documents = documents + negativeDocumentsReview


    # REGION Get Documents

    # pickleHandler = PickleHandler()
    # pickleHandler.Save(documents, "PickleFiles/document.pickle", "wb")

    # END REGION

    # REGION Get all words

    wordTokenizer = WordTokenizer()
    allWords = []

    allWords = allWords + wordTokenizer.GetWords(positiveReviews, True)
    allWords = allWords + wordTokenizer.GetWords(negativeReviews, True)

    # END REGION

    # REGION Compute Words Frequancy

    wordHandler = WordsHandler()
    allWords = wordHandler.FrequancyDistributions(allWords)

    # END REGION

    # REGION Create feature sets

    wordFeatures = list(allWords.keys())[:5000]

    # pickleHandler.Save(wordFeatures, "PickleFiles/wordFeatures.pickle", "wb")

    featureSet = FeatureSet(wordFeatures)
    featureSets = featureSet.Create(documents)

    # pickleHandler.Save(featureSets, "PickleFiles/featureSets.pickle", "wb")

    random.shuffle(featureSets)

    # We'll train against the first dataLength featureset
    # and test against the second dataLength featureset

    collectionIntervalSplitter = CollectionIntervalSplitter()

    trainingSet = collectionIntervalSplitter.FirstElements(featureSets, dataLength)
    testingSet = collectionIntervalSplitter.LastElements(featureSets, dataLength)

    # END REGION

    # REGION Print the NLTK Naive Bayes classifier accuracy

    nltkNaiveBayesClassifier = NltKNaiveBayesClassifier()
    nltkNaiveBayesClassifier.Train(trainingSet)

    # show the most informative 15 features

    # nltkNaiveBayesClassifier.ShowMostInformativeFeatures(15)

    naiveBayesAccuracy = nltkNaiveBayesClassifier.Accuracy(testingSet)
    avrageResult["NaiveBayes"] += naiveBayesAccuracy

    print("Original Naive Bayes Algorithm accuracy percent: ", naiveBayesAccuracy)

    # pickleHandler.Save(nltkNaiveBayesClassifier.GetClassifier(), "PickleFiles/nltkNaiveBayesClassifier.pickle",
    #                   "wb")

    # END REGION

    # REGION Print the Multinomial Naive Bayes classifier accuracy

    multinomialNaiveBayesClassifie = MultinomialNaiveBayesClassifier()
    multinomialNaiveBayesClassifie.Train(trainingSet)

    multinomialNaiveBayesAccuracy = multinomialNaiveBayesClassifie.Accuracy(testingSet)
    avrageResult["MultinomialNaiveBayes"] += multinomialNaiveBayesAccuracy

    print("Multinomial Naive Bayes classifier accuracy percent: ", multinomialNaiveBayesAccuracy)

    # pickleHandler.Save(multinomialNaiveBayesClassifie.GetClassifier(),
    #                  "PickleFiles/multinomialNaiveBayesClassifie.pickle", "wb")

    # END REGION

    # REGION Print the Bernoulli Naive Bayes classifier accuracy

    bernoulliNaiveBayesClassifie = BernoulliNaiveBayesClassifier()
    bernoulliNaiveBayesClassifie.Train(trainingSet)

    beurnoulliNaiveBayesAccuracy = bernoulliNaiveBayesClassifie.Accuracy(testingSet)
    avrageResult["BernouliNaiveBayes"] += beurnoulliNaiveBayesAccuracy

    print("Bernouli Naive Bayes classifier accuracy percent: ", beurnoulliNaiveBayesAccuracy)

    # pickleHandler.Save(bernoulliNaiveBayesClassifie.GetClassifier(),
    #                   "PickleFiles/bernoulliNaiveBayesClassifie.pickle",
    #                   "wb")

    # END REGION

    # REGION Print the Logistic Regression classifier accuracy

    logisticRegressionClassifie = LogisticRegressionClassifier()
    logisticRegressionClassifie.Train(trainingSet)

    logisticRegresionAccuracy = logisticRegressionClassifie.Accuracy(testingSet)
    avrageResult["LogisticRegression"] += logisticRegresionAccuracy

    print("Logistic Regression classifier accuracy percent: ", logisticRegresionAccuracy)

    # pickleHandler.Save(logisticRegressionClassifie.GetClassifier(),
    #                   "PickleFiles/logisticRegressionClassifie.pickle",
    #                   "wb")

    # END REGION

    # REGION Print the SGD classifier accuracy

    SGDClassifie = SGDClassifier()
    SGDClassifie.Train(trainingSet)

    SGDAccuracy = SGDClassifie.Accuracy(testingSet)
    avrageResult["SGD"] += SGDAccuracy

    print("SGD classifier accuracy percent: ", SGDAccuracy)

    # pickleHandler.Save(SGDClassifie.GetClassifier(), "PickleFiles/SGDClassifie.pickle", "wb")

    # END REGION

    # REGION Print the Linear SVC classifier accuracy

    linearSVCClassifier = LinearSVCClassifier()
    linearSVCClassifier.Train(trainingSet)

    linearSVMAccuracy = linearSVCClassifier.Accuracy(testingSet)
    avrageResult["LinearSVM"] += linearSVMAccuracy

    print("Linear SVC classifier accuracy percent: ", linearSVMAccuracy)

    # pickleHandler.Save(linearSVCClassifier.GetClassifier(), "PickleFiles/linearSVCClassifier.pickle", "wb")

    # END REGION

    # REGION Print the NuSVC classifier accuracy

    nuSVCClassifier = NuSVCClassifier()
    nuSVCClassifier.Train(trainingSet)

    NuSVCAccuracy = nuSVCClassifier.Accuracy(testingSet)
    avrageResult["NuSVC"] += NuSVCAccuracy

    print("NuSVC classifier accuracy percent: ", NuSVCAccuracy)
    # pickleHandler.Save(nuSVCClassifier.GetClassifier(), "PickleFiles/nuSVCClassifier.pickle", "wb")




def main():

    RunExperiments(1000)




if __name__ == '__main__':
    main()





