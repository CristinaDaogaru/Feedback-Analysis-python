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


def main():


    # TODO : Refactor DocumentHandler class based on the fallowing code

    # REGION Get Documents


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


    pickleHandler = PickleHandler()
    pickleHandler.Save(documents, "PickleFiles/document.pickle", "wb")

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


    #END REGION


    #REGION Create feature sets


    wordFeatures = list(allWords.keys())[:5000]

    pickleHandler.Save(wordFeatures, "PickleFiles/wordFeatures.pickle", "wb")


    featureSet = FeatureSet(wordFeatures)
    featureSets = featureSet.Create(documents)


    pickleHandler.Save(featureSets, "PickleFiles/featureSets.pickle", "wb")

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

    pickleHandler.Save(nltkNaiveBayesClassifier.GetClassifier(), "PickleFiles/nltkNaiveBayesClassifier.pickle", "wb")


    # END REGION


    # REGION Print the Multinomial Naive Bayes classifier accuracy

    multinomialNaiveBayesClassifie = MultinomialNaiveBayesClassifier()
    multinomialNaiveBayesClassifie.Train(trainingSet)

    print("Multinomial Naive Bayes classifier accuracy percent: ", (multinomialNaiveBayesClassifie.Accuracy(testingSet)))

    pickleHandler.Save(multinomialNaiveBayesClassifie.GetClassifier(), "PickleFiles/multinomialNaiveBayesClassifie.pickle", "wb")


    # END REGION


    # REGION Print the Bernoulli Naive Bayes classifier accuracy

    bernoulliNaiveBayesClassifie = BernoulliNaiveBayesClassifier()
    bernoulliNaiveBayesClassifie.Train(trainingSet)

    print("Bernouli Naive Bayes classifier accuracy percent: ", (bernoulliNaiveBayesClassifie.Accuracy(testingSet)))

    pickleHandler.Save(bernoulliNaiveBayesClassifie.GetClassifier(), "PickleFiles/bernoulliNaiveBayesClassifie.pickle", "wb")


    # END REGION


    # REGION Print the Logistic Regression classifier accuracy

    logisticRegressionClassifie = LogisticRegressionClassifier()
    logisticRegressionClassifie.Train(trainingSet)

    print("Logistic Regression classifier accuracy percent: ", (logisticRegressionClassifie.Accuracy(testingSet)))

    pickleHandler.Save(logisticRegressionClassifie.GetClassifier(), "PickleFiles/logisticRegressionClassifie.pickle", "wb")


    # END REGION


    # REGION Print the SGD classifier accuracy

    SGDClassifie = SGDClassifier()
    SGDClassifie.Train(trainingSet)

    print("SGD classifier accuracy percent: ", (SGDClassifie.Accuracy(testingSet)))

    pickleHandler.Save(SGDClassifie.GetClassifier(), "PickleFiles/SGDClassifie.pickle", "wb")


    # END REGION


    # REGION Print the Linear SVC classifier accuracy

    linearSVCClassifier = LinearSVCClassifier()
    linearSVCClassifier.Train(trainingSet)

    print("Linear SVC classifier accuracy percent: ", (linearSVCClassifier.Accuracy(testingSet)))

    pickleHandler.Save(linearSVCClassifier.GetClassifier(), "PickleFiles/linearSVCClassifier.pickle", "wb")


    # END REGION


    # REGION Print the NuSVC classifier accuracy

    nuSVCClassifier = NuSVCClassifier()
    nuSVCClassifier.Train(trainingSet)

    print("NuSVC classifier accuracy percent: ", (nuSVCClassifier.Accuracy(testingSet)))

    pickleHandler.Save(nuSVCClassifier.GetClassifier(), "PickleFiles/nuSVCClassifier.pickle", "wb")


    # END REGION



    # REGION Print the Neural Network result

    neuralNetwork = NeuralNetwork()
    neuralNetwork.TrainNetwork()


    #a = neuralNetwork.classify("Good movie")

    # print("Print entire result : %s", a)
    # print('\n')
    # print("print first component : %s ", a[0][0])
    # print('\n')
    # print("print second component : %s ", a[0][1])
    # print("\n")


    # END REGION




    # REGION Print the Voted classifier accuracy

    voteClassifier = VoteClassifier(nltkNaiveBayesClassifier.GetClassifier(),
                                    multinomialNaiveBayesClassifie.GetClassifier(),
                                    bernoulliNaiveBayesClassifie.GetClassifier(),
                                    logisticRegressionClassifie.GetClassifier(),
                                    #SGDClassifie.GetClassifier(),
                                    linearSVCClassifier.GetClassifier(),
                                    nuSVCClassifier.GetClassifier())

    moviePositiveReview = "The new school is nice. The teachers seem prepared. The colleagues are very polite. I will integrate myself perfectly here. I think all will be just fine."
    movieNegativeReview = "The new school is looks nice but the teachers are rude . The colleagues are rude too. I don't think I will integrate very well. Probably I will move back to my old schoold."



    voteClassifier.SetNeuralNetwork(neuralNetwork, moviePositiveReview)


    #print("Vote classifier accuracy percent: ", (voteClassifier.Accuracy(testingSet)))


    wordFeatures = pickleHandler.Load("PickleFiles/wordFeatures.pickle", "rb")


    sentiment = Sentiment(voteClassifier, wordFeatures)


    result = sentiment.FindSentiment(moviePositiveReview)


    print("\n\nSentence : %s\nResult : %s\n\n" % (moviePositiveReview, result))


    voteClassifier.SetNeuralNetwork(neuralNetwork, movieNegativeReview)
    result = sentiment.FindSentiment(movieNegativeReview)

    print("\n\n!!!! Sentence : %s\nResult : %s\n\n" % (movieNegativeReview, result))



    # print(sentiment.FindSentiment(
    #     "This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))
    # print(sentiment.FindSentiment(
    #     "This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))

    # END REGION


if __name__ == '__main__':
    main()
