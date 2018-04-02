import random

from Sentiment import Sentiment
from Helpers.CollectionIntervalSplitter import CollectionIntervalSplitter
from Helpers.PickleHandler import PickleHandler

from Classifiers.NltKNaiveBayesClassifier import NltKNaiveBayesClassifier
from Classifiers.MultinomialNaiveBayesClassifier import MultinomialNaiveBayesClassifier
from Classifiers.VoteClassifier import VoteClassifier
from Classifiers.LinearSVCClassifier import LinearSVCClassifier
from Classifiers.NuSVCClassifier import NuSVCClassifier

def main():

    # TODO : Refactor DocumentHandler class based on the fallowing code

    # REGION Create feature sets

    pickleHandler = PickleHandler()
    featureSets = pickleHandler.Load("PickleFiles/featureSets.pickle", "rb")


    random.shuffle(featureSets)

    # We'll train against the first 10000 featureset
    # and test against the second 10000 featureset

    collectionIntervalSplitter = CollectionIntervalSplitter()

    trainingSet = collectionIntervalSplitter.FirstElements(featureSets, 10000)
    testingSet = collectionIntervalSplitter.LastElements(featureSets, 10000)


    # END REGION


    # REGION Get the NLTK Naive Bayes classifier


    classifier = pickleHandler.Load("PickleFiles/nltkNaiveBayesClassifier.pickle", "rb")
    nltkNaiveBayesClassifier = NltKNaiveBayesClassifier(classifier)


    # END REGION


    # REGION Get the Multinomial Naive Bayes classifier


    classifier = pickleHandler.Load("PickleFiles/multinomialNaiveBayesClassifie.pickle", "rb")
    multinomialNaiveBayesClassifie = MultinomialNaiveBayesClassifier(classifier)


    # END REGION


    # REGION Get the Bernoulli Naive Bayes classifier


    classifier = pickleHandler.Load("PickleFiles/bernoulliNaiveBayesClassifie.pickle", "rb")
    bernoulliNaiveBayesClassifie = MultinomialNaiveBayesClassifier(classifier)


    # END REGION


    # REGION Get the Logistic Regression classifier


    classifier = pickleHandler.Load("PickleFiles/logisticRegressionClassifie.pickle", "rb")
    logisticRegressionClassifie = MultinomialNaiveBayesClassifier(classifier)


    # END REGION


    # REGION Get the SGD classifier


    classifier = pickleHandler.Load("PickleFiles/SGDClassifie.pickle", "rb")
    SGDClassifie = MultinomialNaiveBayesClassifier(classifier)


    # END REGION


    # REGION Get the Linear SVC classifier


    classifier = pickleHandler.Load("PickleFiles/linearSVCClassifier.pickle", "rb")
    linearSVCClassifier = LinearSVCClassifier(classifier)


    # END REGION


    # REGION Get the NuSVC classifier


    classifier = pickleHandler.Load("PickleFiles/nuSVCClassifier.pickle", "rb")
    nuSVCClassifier = NuSVCClassifier(classifier)


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


    # REGION Classifie random input text
    

    wordFeatures = pickleHandler.Load("PickleFiles/wordFeatures.pickle", "rb")

    sentiment = Sentiment(voteClassifier, wordFeatures)

    print(sentiment.FindSentiment(
        "This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))
    print(sentiment.FindSentiment(
        "This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))


    # END REGION


if __name__ == '__main__':
    main()



