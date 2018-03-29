import random
import FeatureSet
from Helpers.FileReader import FileReader
from WordTokenizer import WordTokenizer
from Classifiers.NltKNaiveBayesClassifier import NltKNaiveBayesClassifier
from Helpers.CollectionIntervalSplitter import CollectionIntervalSplitter
from Helpers.DocumentHandler import DocumentHandler
from Helpers.WordsHandler import WordsHandler


def main():

    # TODO : Refactor DocumentHandler class based on the fallowing code

    # REGION Get Documents

    documentHandler = DocumentHandler()

    shortPositiveReviewsPath = "short_reviews/positive.txt"
    shortNegativeReviewsPath = "short_reviews/negative.txt"

    positiveReviews = FileReader(shortPositiveReviewsPath, "r")
    negativeReviews = FileReader(shortNegativeReviewsPath, "r")

    positiveDocumentsReview = documentHandler.GetPositiveDocumets(positiveReviews)
    negativeDocumentsReview = documentHandler.GetNegativeDocumets(negativeReviews)


    documents = []
    documents.append(positiveDocumentsReview)
    documents.append(negativeDocumentsReview)

    # END REGION


    # REGION Get all words

    wordTokenizer = WordTokenizer()
    allWords = []

    allWords.append(wordTokenizer.GetWords(positiveDocumentsReview, True))
    allWords.append(wordTokenizer.GetWords(negativeDocumentsReview, True))

    # END REGION


    # REGION Compute Words Frequancy

    wordHandler = WordsHandler()
    allWords = wordHandler.FrequancyDistributions(allWords)

    # END REGION


    # REGION Create feature sets

    featureSet = FeatureSet()

    featureSets = featureSet.Create(allWords, 5000, documents)
    random.shuffle(featureSets)

    collectionIntervalSplitter = CollectionIntervalSplitter()

    # We'll train against the first 10000 featureset
    # and test against the second 10000 featureset

    trainingSet = collectionIntervalSplitter.FirstElements(featureSets, 10000)
    testingSet = collectionIntervalSplitter.LastElements(featureSets, 10000)


    # END REGION


    # REGION Print the NLTK Naive Bayes classifier accuracy

    nltkNaiveBayesClassifier = NltKNaiveBayesClassifier()
    nltkNaiveBayesClassifier.Train(trainingSet)

    print("Original Naive Bayes Algorithm accuracy percent: ", (nltkNaiveBayesClassifier.Accuracy(testingSet)))

    # show the most informative 15 features
    nltkNaiveBayesClassifier.ShowMostInformativeFeatures(15)

    # END REGION


if __name__ == '__main__':
    main()



