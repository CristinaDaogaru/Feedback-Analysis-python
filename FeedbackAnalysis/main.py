import random
import nltk

from Classifiers.MultinomialNaiveBayesClassifier import MultinomialNaiveBayesClassifier
from FeatureSet import FeatureSet
from Helpers.FileReader import FileReader
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


def main():

    # TODO : Refactor DocumentHandler class based on the fallowing code

    # REGION Get Documents


    shortPositiveReviewsPath = "short_reviews/positive.txt"
    shortNegativeReviewsPath = "short_reviews/negative.txt"


    fileReader = FileReader()

    positiveReviews = fileReader.ReadToEnd(shortPositiveReviewsPath, "r")
    negativeReviews = fileReader.ReadToEnd(shortNegativeReviewsPath, "r")


    # Commented this because the documents were pickeled


    # documentHandler = DocumentHandler()
    #
    # positiveDocumentsReview = documentHandler.GetPositiveDocumets(positiveReviews)
    # negativeDocumentsReview = documentHandler.GetNegativeDocumets(negativeReviews)
    #
    #
    # documents = []
    # documents = documents + positiveDocumentsReview
    # documents = documents + negativeDocumentsReview


    # End commented this because the documents were pickeled


    pickleHandler = PickleHandler()

    #pickleHandler.Save(documents, "PickleFiles/document.pickle", "wb")
    documents = pickleHandler.Load("PickleFiles/document.pickle", "rb")

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

    word_features = list(allWords.keys())[:5000]
    featureSet = FeatureSet(word_features)

    featureSets = featureSet.Create(documents)
    random.shuffle(featureSets)

    # We'll train against the first 10000 featureset
    # and test against the second 10000 featureset

    collectionIntervalSplitter = CollectionIntervalSplitter()

    trainingSet = collectionIntervalSplitter.FirstElements(featureSets, 10000)
    testingSet = collectionIntervalSplitter.LastElements(featureSets, 10000)


    # END REGION


    # REGION Print the NLTK Naive Bayes classifier accuracy

    nltkNaiveBayesClassifier = NltKNaiveBayesClassifier()
    nltkNaiveBayesClassifier.Train(trainingSet)

    # show the most informative 15 features
    nltkNaiveBayesClassifier.ShowMostInformativeFeatures(15)

    print("Original Naive Bayes Algorithm accuracy percent: ", (nltkNaiveBayesClassifier.Accuracy(testingSet)))


    # END REGION


    # REGION Print the Multinomial Naive Bayes classifier accuracy

    multinomialNaiveBayesClassifie = MultinomialNaiveBayesClassifier()
    multinomialNaiveBayesClassifie.Train(trainingSet)

    print("Multinomial Naive Bayes classifier accuracy percent: ", (multinomialNaiveBayesClassifie.Accuracy(testingSet)))

    # END REGION

    # REGION Print the Bernoulli Naive Bayes classifier accuracy

    bernoulliNaiveBayesClassifie = MultinomialNaiveBayesClassifier()
    bernoulliNaiveBayesClassifie.Train(trainingSet)

    print("Bernouli Naive Bayes classifier accuracy percent: ", (bernoulliNaiveBayesClassifie.Accuracy(testingSet)))

    # END REGION


    # REGION Print the Logistic Regression classifier accuracy

    logisticRegressionClassifie = MultinomialNaiveBayesClassifier()
    logisticRegressionClassifie.Train(trainingSet)

    print("Logistic Regression classifier accuracy percent: ", (logisticRegressionClassifie.Accuracy(testingSet)))

    # END REGION


    # REGION Print the SGD classifier accuracy

    SGDClassifie = MultinomialNaiveBayesClassifier()
    SGDClassifie.Train(trainingSet)

    print("SGD classifier accuracy percent: ", (SGDClassifie.Accuracy(testingSet)))

    # END REGION


    # REGION Print the Linear SVC classifier accuracy

    linearSVCClassifier = LinearSVCClassifier()
    linearSVCClassifier.Train(trainingSet)

    print("Linear SVC classifier accuracy percent: ", (linearSVCClassifier.Accuracy(testingSet)))

    # END REGION


    # REGION Print the NuSVC classifier accuracy

    nuSVCClassifier = NuSVCClassifier()
    nuSVCClassifier.Train(trainingSet)

    print("NuSVC classifier accuracy percent: ", (nuSVCClassifier.Accuracy(testingSet)))

    # END REGION


    # REGION Print the Voted classifier accuracy

    voteClassifier = VoteClassifier(nltkNaiveBayesClassifier.GetClassifier(),
                                    multinomialNaiveBayesClassifie.GetClassifier(),
                                    bernoulliNaiveBayesClassifie.GetClassifier(),
                                    logisticRegressionClassifie.GetClassifier(),
                                    SGDClassifie.GetClassifier(),
                                    linearSVCClassifier.GetClassifier(),
                                    nuSVCClassifier.GetClassifier())

    print("Vote classifier accuracy percent: ", (voteClassifier.Accuracy(testingSet)))

    # END REGION


if __name__ == '__main__':
    main()



