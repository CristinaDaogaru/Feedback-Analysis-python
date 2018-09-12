import random
import nltk
import pickle
from Classifiers.MultinomialNaiveBayesClassifier import MultinomialNaiveBayesClassifier
from Classifiers.NeuralNetwork import NeuralNetwork
from Classifiers.NeuralNetworkClean import NeuralNetworkClean
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

from nltk.tokenize import word_tokenize

from nltk.classify import ClassifierI

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression


from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC


from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import NuSVC


import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import numpy as np
import time
from Helpers.FileReader import FileReader
from Helpers.TrainingDataCreator import TrainingDataCreator


import nltk
from nltk.classify import ClassifierI
from statistics import mode


from FeatureSet import FeatureSet



class FileReader:

    def ReadToEnd(self, filePath, format):

        return open(filePath, format).read()


    def ReadLines(self, filePath):

        file = open(filePath, "r")
        return list(file)




class DocumentHandler:

    def GetDocuments(self, nltkCorpusReviews):

        documents = [(list(nltkCorpusReviews.words(fileid)), category)
                      for category in nltkCorpusReviews.categories()
                      for fileid in nltkCorpusReviews.fileids(category)]

        return documents


    def GetPositiveDocumets(self, reviews):

        documents = []
        for review in reviews.split('\n'):
            documents.append((review, "pos"))

        return documents


    def GetNegativeDocumets(self, reviews):

        documents = []
        for review in reviews.split('\n'):
            documents.append((review, "neg"))

        return documents


    def Shuffle(self, documents):

        random.shuffle(documents)
        return documents





class PickleHandler:

    def Save(self, savedData, fileName, format):

        saveFile = open(fileName, format)
        pickle.dump(savedData, saveFile)
        saveFile.close()


    def Load(self, fileName, format):

        loadFile = open(fileName, format)
        loadedData = pickle.load(loadFile)
        loadFile.close()
        return loadedData




class WordTokenizer:


    # Return a list with all the words contained in the text parameter
    # LowerCase parameter is optional and is by default true
    def GetWords(self, text, lowerCase = True):

        all_words = []
        words = self.WordTokenize(text)

        for w in words:

            if True == lowerCase:
                all_words.append(w.lower())
            else:
                all_words.append(w.upper())

        return all_words


    def WordTokenize(self, text):

        return word_tokenize(text)



class WordsHandler:

    def FrequancyDistributions(self, words):

        return nltk.FreqDist(words)


    def MostCommonWords(self, words, number):

        return words.most_common(number)


    def WordFrequency(self, words, word):

        return words[word]





class FeatureSet:


    def __init__(self, wordFeatures):

        self.wordFeatures = wordFeatures

    def Create(self, documents):

        featureSets = [(self.FindFeatures(rev), category) for (rev, category) in documents]
        return featureSets


    def FindFeatures(self, document):

        ########### Commented because of new files training data set #################

        # words = set(document)

        ########### End Commented because of new files training data set #################

        wordTokenizer = WordTokenizer()
        words = wordTokenizer.WordTokenize(document)

        features = {}

        # if the words w from the 3000 words appears in document then store true
        # store false otherwise
        for w in self.wordFeatures:
            features[w] = (w in words)

        return features





class CollectionIntervalSplitter:


    def FirstElements(self, elements, number):

        return elements[:number]


    def LastElements(self, elements, number):

        return elements[number:]


    def BetweenElements(self, elements, startIndex, stopIndex):

        return elements[startIndex:stopIndex]





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





class LogisticRegressionClassifier:

    def __init__(self, classifier = SklearnClassifier(LogisticRegression())):

        self.classifier = classifier


    def GetClassifier(self):

        return self.classifier


    def Train(self, trainingSet):

        self.classifier.train(trainingSet)


    def Accuracy(self, testingSet):

        return (nltk.classify.accuracy(self.classifier, testingSet)) * 100






class LinearSVCClassifier:

    def __init__(self, classifier = SklearnClassifier(LinearSVC())):

        self.classifier = classifier


    def GetClassifier(self):

        return self.classifier


    def Train(self, trainingSet):

        self.classifier.train(trainingSet)


    def Accuracy(self, testingSet):

        return (nltk.classify.accuracy(self.classifier, testingSet)) * 100






class NuSVCClassifier:

    def __init__(self, classifier = SklearnClassifier(NuSVC())):

        self.classifier = classifier


    def GetClassifier(self):

        return self.classifier


    def Train(self, trainingSet):

        self.classifier.train(trainingSet)


    def Accuracy(self, testingSet):

        return (nltk.classify.accuracy(self.classifier, testingSet)) * 100








class NeuralNetworkClean:


    def __init__(self):
        self.stemmer = LancasterStemmer()


    def TrainNetwork(self, dataLength, addPositiveNegativeWords=False, hidden_neurons=10, alpha=0.1, epochs=1000, dropout=False, dropout_percent=0.2):

        self.stemmer = LancasterStemmer()

        # Create data training with positive and negative words

        positiveWordsFilePath = "PositiveNegativeWords/positive-words.txt"
        negativeWordsFilePath = "PositiveNegativeWords/negative-words.txt"

        positiveWordsTrainingData = TrainingDataCreator.Create(positiveWordsFilePath, "positive", 2006)
        negativeWordsTrainingData = TrainingDataCreator.Create(negativeWordsFilePath, "negative", 2783)

        # Create data training with positive and negative sentences

        positiveReviewsFilePath = "short_reviews/positiveShort"
        negativeReviewsFilePath = "short_reviews/negativeShort"

        positiveReviewsTrainingData = TrainingDataCreator.Create(positiveReviewsFilePath, "positive", int(dataLength/2))
        negativeReviewsTrainingData = TrainingDataCreator.Create(negativeReviewsFilePath, "negative", int(dataLength/2))

        # 2 classes of training data
        training_data = []
        training_data.append({"class": "positive", "sentence": "This is good"})
        training_data.append({"class": "positive", "sentence": "Very nice"})
        training_data.append({"class": "positive", "sentence": "I like it"})
        training_data.append({"class": "positive", "sentence": "Better now"})


        if addPositiveNegativeWords == True:
            training_data += positiveWordsTrainingData

        training_data += positiveReviewsTrainingData

        training_data.append({"class": "negative", "sentence": "I don't like it"})
        training_data.append({"class": "negative", "sentence": "Bad"})
        training_data.append({"class": "negative", "sentence": "Very bad"})
        training_data.append({"class": "negative", "sentence": "This is not good"})

        if addPositiveNegativeWords == True:
            training_data += negativeWordsTrainingData

        training_data += negativeReviewsTrainingData

        # with open('synapses.json') as json_file:
        #     training_data = json.load(json_file)

        #print("%s sentences in training data" % len(training_data))

        # in 3
        self.words = []
        self.classes = []
        documents = []
        ignore_words = ['?', '!', '.', ',', '@', '#', '^', '&', '*', '(', ')', '_', '=', '<', '>', '|']

        # loop through each sentence in our training data
        for pattern in training_data:

            # tokenize each word in the sentence
            w = nltk.word_tokenize(pattern['sentence'])

            # add to our words list
            self.words.extend(w)

            # add to documents in our corpus
            documents.append((w, pattern['class']))

            # add to our classes list
            if pattern['class'] not in self.classes:
                self.classes.append(pattern['class'])

        # stem and lower each word and remove duplicates
        self.words = [self.stemmer.stem(w.lower()) for w in self.words if w not in ignore_words]
        self.words = list(set(self.words))

        # remove duplicates
        self.classes = list(set(self.classes))

        #print(len(documents), "documents")
        #print(len(self.classes), "classes", self.classes)
        #print(len(self.words), "unique stemmed words", self.words)

        # in 4
        # create our training data
        training = []
        output = []
        # create an empty array for our output
        output_empty = [0] * len(self.classes)

        # training set, bag of words for each sentence
        for doc in documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [self.stemmer.stem(word.lower()) for word in pattern_words]
            # create our bag of words array
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            training.append(bag)
            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            output.append(output_row)

        #print("# words", len(self.words))
        #print("# classes", len(self.classes))

        # sample training/output
        i = 0
        w = documents[i][0]
        #print([self.stemmer.stem(word.lower()) for word in w])
        #print(training[i])
        #print(output[i])

        # in 5

        # in 9
        X = np.array(training)
        y = np.array(output)

        start_time = time.time()

        self.train(X, y, hidden_neurons, alpha, epochs, dropout, dropout_percent)

        elapsed_time = time.time() - start_time
        #print("processing time:", elapsed_time, "seconds")

        # in 10
        # probability threshold
        self.ERROR_THRESHOLD = 0.2
        # load our calculated synapse values
        synapse_file = 'synapses.json'

        self.synapse_0 = []
        synapse_1 = []

        with open(synapse_file) as data_file:
            synapse = json.load(data_file)
            self.synapse_0 = np.asarray(synapse['synapse0'])
            self.synapse_1 = np.asarray(synapse['synapse1'])

        # self.classify("This movie was awesome!")
        # self.classify("Good job")
        # self.classify("this was a bad experience")
        # self.classify("very nice movie")
        # self.classify("bad decision")
        # self.classify("good enough")
        # print("\n\n")
        # self.classify("you make my day", show_details=True)
        #
        # print("\n\n")
        #
        # self.classify("This movie was awesome! The acting was great, plot was wonderful")
        #
        # self.classify(
        #     "This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10")
        #
        # self.classify("This is good")
        #
        # self.classify("This is not good")
        #
        # print("\n\nFinish")


    # compute sigmoid nonlinearity
    def sigmoid(self, x):
        output = 1 / (1 + np.exp(-x))
        return output



    # convert output of sigmoid function to its derivative
    def sigmoid_output_to_derivative(self, output):
        return output * (1 - output)

    def clean_up_sentence(self, sentence):
        # tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word
        sentence_words = [self.stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words


    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self, sentence, words, show_details=False):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)

        return (np.array(bag))



    # in 6


    # ANN and Gradient Descent code from https://iamtrask.github.io//2015/07/27/python-network-part2/
    def train(self, X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
        #print("Training with %s neurons, alpha:%s, dropout:%s %s" % (
        #    hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
        #print("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X), len(X[0]), 1, len(self.classes)))
        np.random.seed(1)

        last_mean_error = 1
        # randomly initialize our weights with mean 0
        synapse_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
        synapse_1 = 2 * np.random.random((hidden_neurons, len(self.classes))) - 1

        prev_synapse_0_weight_update = np.zeros_like(synapse_0)
        prev_synapse_1_weight_update = np.zeros_like(synapse_1)

        synapse_0_direction_count = np.zeros_like(synapse_0)
        synapse_1_direction_count = np.zeros_like(synapse_1)

        for j in iter(range(epochs + 1)):

            # Feed forward through layers 0, 1, and 2
            layer_0 = X
            layer_1 = self.sigmoid(np.dot(layer_0, synapse_0))

            if (dropout):
                layer_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))], 1 - dropout_percent)[0] * (
                        1.0 / (1 - dropout_percent))

            layer_2 = self.sigmoid(np.dot(layer_1, synapse_1))

            # how much did we miss the target value?
            layer_2_error = y - layer_2

            if (j % 10000) == 0 and j > 5000:
                # if this 10k iteration's error is greater than the last iteration, break out
                if np.mean(np.abs(layer_2_error)) < last_mean_error:
                    #print("delta after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error))))
                    last_mean_error = np.mean(np.abs(layer_2_error))
                else:
                    #print("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error)
                    break

            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            layer_2_delta = layer_2_error * self.sigmoid_output_to_derivative(layer_2)

            # how much did each l1 value contribute to the l2 error (according to the weights)?
            layer_1_error = layer_2_delta.dot(synapse_1.T)

            # in what direction is the target l1?
            # were we really sure? if so, don't change too much.
            layer_1_delta = layer_1_error * self.sigmoid_output_to_derivative(layer_1)

            synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
            synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

            if (j > 0):
                synapse_0_direction_count += np.abs(
                    ((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
                synapse_1_direction_count += np.abs(
                    ((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))

            synapse_1 += alpha * synapse_1_weight_update
            synapse_0 += alpha * synapse_0_weight_update

            prev_synapse_0_weight_update = synapse_0_weight_update
            prev_synapse_1_weight_update = synapse_1_weight_update

        now = datetime.datetime.now()

        # persist synapses
        synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
                    'datetime': now.strftime("%Y-%m-%d %H:%M"),
                    'words': self.words,
                    'classes': self.classes
                    }
        synapse_file = "synapses.json"

        with open(synapse_file, 'w') as outfile:
            json.dump(synapse, outfile, indent=4, sort_keys=True)
        #print("saved synapses to:", synapse_file)




    def classify(self, sentence, show_details=False):

        results = self.think(sentence, show_details)
        results = [[i, r] for i, r in enumerate(results) if r > self.ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_results = [[self.classes[r[0]], r[1]] for r in results]
        #print("%s \n classification: %s" % (sentence, return_results))
        return return_results

    def think(self, sentence, show_details=False):
        x = self.bow(sentence.lower(), self.words, show_details)
        if show_details:
            print("sentence:", sentence, "\n bow:", x)
        # input layer is our bag of words
        l0 = x
        # matrix multiplication of input and hidden layer
        l1 = self.sigmoid(np.dot(l0, self.synapse_0))
        # output layer
        l2 = self.sigmoid(np.dot(l1, self.synapse_1))
        return l2






class VoteClassifier:  # inherit form ClassifierI

    def __init__(self, *classifiers):
        self.classifiers = classifiers
        self.neuralNetwork = None

    def SetNeuralNetwork(self, aNeuralNetwork, aText):
        self.neuralNetwork = aNeuralNetwork
        self.text = aText


    def classify(self, features):
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)

        if self.neuralNetwork != None:
            result = self.neuralNetwork.classify(self.text)

            if result[0] != None or result[0][0] != None:
                if result[0][0] == "positive":
                    votes.append("pos")
                else:
                    votes.append("neg")

        return mode(votes)


    def Accuracy(self, testingSet):

        return (nltk.classify.accuracy(self, testingSet)) * 100


    def confidence(self, features):
        votes = []
        for c in self.classifiers:
            v = c.classify(features)
            votes.append(v)

        choise_votes = votes.count(mode(votes))
        conf = choise_votes / len(votes)

        return conf






class Sentiment:

    def __init__(self, classifier, wordFeatures):

        self.votedClassifier = classifier
        self.featureSet = FeatureSet(wordFeatures)

    def FindSentiment(self, text):

        features = self.featureSet.FindFeatures(text)
        return self.votedClassifier.classify(features) #, self.votedClassifier.confidence(features)





def main():


    # TODO : Refactor DocumentHandler class based on the fallowing code

    # REGION Get Documents

    shortPositiveReviewsPath = "short_reviews/positive.txt"
    shortNegativeReviewsPath = "short_reviews/negative.txt"

    positiveWordsPath = "PositiveNegativeWords/positive-words.txt"
    negativeWordsPath = "PositiveNegativeWords/negative-words.txt"

    fileReader = FileReader()

    positiveReviews = fileReader.ReadToEnd(shortPositiveReviewsPath, "r")
    negativeReviews = fileReader.ReadToEnd(shortNegativeReviewsPath, "r")

    positiveWords = fileReader.ReadToEnd(positiveWordsPath, "r")
    negativeWords = fileReader.ReadToEnd(negativeWordsPath, "r")

    documentHandler = DocumentHandler()

    positiveDocumentsReview = documentHandler.GetPositiveDocumets(positiveReviews)
    positiveDocumentsReview += documentHandler.GetPositiveDocumets(positiveWords)

    negativeDocumentsReview = documentHandler.GetNegativeDocumets(negativeReviews)
    negativeDocumentsReview += documentHandler.GetNegativeDocumets(negativeWords)

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

    trainingSet = collectionIntervalSplitter.FirstElements(featureSets, 1000)
    testingSet = collectionIntervalSplitter.LastElements(featureSets, 1000)


    # END REGION


    # REGION Print the NLTK Naive Bayes classifier accuracy

    nltkNaiveBayesClassifier = NltKNaiveBayesClassifier()
    nltkNaiveBayesClassifier.Train(trainingSet)

    # show the most informative 15 features
    #nltkNaiveBayesClassifier.ShowMostInformativeFeatures(15)
    #print("Original Naive Bayes Algorithm accuracy percent: ", (nltkNaiveBayesClassifier.Accuracy(testingSet)))

    pickleHandler.Save(nltkNaiveBayesClassifier.GetClassifier(), "PickleFiles/nltkNaiveBayesClassifier.pickle", "wb")


    # END REGION


    # REGION Print the Multinomial Naive Bayes classifier accuracy

    # multinomialNaiveBayesClassifie = MultinomialNaiveBayesClassifier()
    # multinomialNaiveBayesClassifie.Train(trainingSet)

    #print("Multinomial Naive Bayes classifier accuracy percent: ", (multinomialNaiveBayesClassifie.Accuracy(testingSet)))

    #pickleHandler.Save(multinomialNaiveBayesClassifie.GetClassifier(), "PickleFiles/multinomialNaiveBayesClassifie.pickle", "wb")


    # END REGION


    # REGION Print the Bernoulli Naive Bayes classifier accuracy

    # bernoulliNaiveBayesClassifie = BernoulliNaiveBayesClassifier()
    # bernoulliNaiveBayesClassifie.Train(trainingSet)

    #print("Bernouli Naive Bayes classifier accuracy percent: ", (bernoulliNaiveBayesClassifie.Accuracy(testingSet)))

    #pickleHandler.Save(bernoulliNaiveBayesClassifie.GetClassifier(), "PickleFiles/bernoulliNaiveBayesClassifie.pickle", "wb")


    # END REGION


    # REGION Print the Logistic Regression classifier accuracy

    logisticRegressionClassifie = LogisticRegressionClassifier()
    logisticRegressionClassifie.Train(trainingSet)

    #print("Logistic Regression classifier accuracy percent: ", (logisticRegressionClassifie.Accuracy(testingSet)))

    #pickleHandler.Save(logisticRegressionClassifie.GetClassifier(), "PickleFiles/logisticRegressionClassifie.pickle", "wb")


    # END REGION


    # REGION Print the SGD classifier accuracy

    #SGDClassifie = SGDClassifier()
    #SGDClassifie.Train(trainingSet)

    #print("SGD classifier accuracy percent: ", (SGDClassifie.Accuracy(testingSet)))

    #pickleHandler.Save(SGDClassifie.GetClassifier(), "PickleFiles/SGDClassifie.pickle", "wb")


    # END REGION


    # REGION Print the Linear SVC classifier accuracy

    linearSVCClassifier = LinearSVCClassifier()
    linearSVCClassifier.Train(trainingSet)

    #print("Linear SVC classifier accuracy percent: ", (linearSVCClassifier.Accuracy(testingSet)))

    #pickleHandler.Save(linearSVCClassifier.GetClassifier(), "PickleFiles/linearSVCClassifier.pickle", "wb")


    # END REGION


    # REGION Print the NuSVC classifier accuracy

    nuSVCClassifier = NuSVCClassifier()
    nuSVCClassifier.Train(trainingSet)

    #print("NuSVC classifier accuracy percent: ", (nuSVCClassifier.Accuracy(testingSet)))

    #pickleHandler.Save(nuSVCClassifier.GetClassifier(), "PickleFiles/nuSVCClassifier.pickle", "wb")


    # END REGION



    # REGION Print the Neural Network result

    neuralNetwork = NeuralNetworkClean()
    neuralNetwork.TrainNetwork(dataLength=1000, addPositiveNegativeWords=False, hidden_neurons=20, alpha=0.1, epochs=1000, dropout=False, dropout_percent=0.2)


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
                                    #multinomialNaiveBayesClassifie.GetClassifier(),
                                    #bernoulliNaiveBayesClassifie.GetClassifier(),
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


    #print("\n\nSentence : %s\nResult : %s\n\n" % (moviePositiveReview, result))


    voteClassifier.SetNeuralNetwork(neuralNetwork, movieNegativeReview)
    result = sentiment.FindSentiment(movieNegativeReview)

    #print("\n\n!!!! Sentence : %s\nResult : %s\n\n" % (movieNegativeReview, result))


    print("Positive")


    # print(sentiment.FindSentiment(
    #     "This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))
    # print(sentiment.FindSentiment(
    #     "This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))

    # END REGION


if __name__ == '__main__':
    main()
