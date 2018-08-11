import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import numpy as np
import time
from Helpers.FileReader import FileReader
from Helpers.TrainingDataCreator import TrainingDataCreator


class NeuralNetwork:


    def __init__(self):
        self.stemmer = LancasterStemmer()


    def TrainNetwork(self):

        self.stemmer = LancasterStemmer()

        # Create data training with positive and negative words

        positiveWordsFilePath = "PositiveNegativeWords/positive-words.txt"
        negativeWordsFilePath = "PositiveNegativeWords/negative-words.txt"

        positiveWordsTrainingData = TrainingDataCreator.Create(positiveWordsFilePath, "positive", 1000)
        negativeWordsTrainingData = TrainingDataCreator.Create(negativeWordsFilePath, "negative", 1000)

        # Create data training with positive and negative sentences

        positiveReviewsFilePath = "short_reviews/positiveShort"
        negativeReviewsFilePath = "short_reviews/negativeShort"

        positiveReviewsTrainingData = TrainingDataCreator.Create(positiveReviewsFilePath, "positive", 1000)
        negativeReviewsTrainingData = TrainingDataCreator.Create(negativeReviewsFilePath, "negative", 1000)

        # 2 classes of training data
        training_data = []
        training_data.append({"class": "positive", "sentence": "This is good"})
        training_data.append({"class": "positive", "sentence": "Very nice"})
        training_data.append({"class": "positive", "sentence": "I like it"})
        training_data.append({"class": "positive", "sentence": "Better now"})

        training_data += positiveWordsTrainingData
        training_data += positiveReviewsTrainingData

        training_data.append({"class": "negative", "sentence": "I don't like it"})
        training_data.append({"class": "negative", "sentence": "Bad"})
        training_data.append({"class": "negative", "sentence": "Very bad"})
        training_data.append({"class": "negative", "sentence": "This is not good"})

        training_data += negativeWordsTrainingData
        training_data += negativeReviewsTrainingData

        # with open('synapses.json') as json_file:
        #     training_data = json.load(json_file)

        print("%s sentences in training data" % len(training_data))

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

        print(len(documents), "documents")
        print(len(self.classes), "classes", self.classes)
        print(len(self.words), "unique stemmed words", self.words)

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

        print("# words", len(self.words))
        print("# classes", len(self.classes))

        # sample training/output
        i = 0
        w = documents[i][0]
        print([self.stemmer.stem(word.lower()) for word in w])
        print(training[i])
        print(output[i])

        # in 5

        # in 9
        X = np.array(training)
        y = np.array(output)

        start_time = time.time()

        self.train(X, y, hidden_neurons=10, alpha=0.1, epochs=1000, dropout=True, dropout_percent=0.2)

        elapsed_time = time.time() - start_time
        print("processing time:", elapsed_time, "seconds")

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

        self.classify("This movie was awesome!")
        self.classify("Good job")
        self.classify("this was a bad experience")
        self.classify("very nice movie")
        self.classify("bad decision")
        self.classify("good enough")
        print("\n\n")
        self.classify("you make my day", show_details=True)

        print("\n\n")

        self.classify("This movie was awesome! The acting was great, plot was wonderful")

        self.classify(
            "This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10")

        self.classify("This is good")

        self.classify("This is not good")

        print("\n\nFinish")


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

    # in 6


    # ANN and Gradient Descent code from https://iamtrask.github.io//2015/07/27/python-network-part2/
    def train(self, X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
        print("Training with %s neurons, alpha:%s, dropout:%s %s" % (
            hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
        print("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X), len(X[0]), 1, len(self.classes)))
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
                    print("delta after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error))))
                    last_mean_error = np.mean(np.abs(layer_2_error))
                else:
                    print("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error)
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
        print("saved synapses to:", synapse_file)




    def classify(self, sentence, show_details=False):

        results = self.think(sentence, show_details)

        results = [[i, r] for i, r in enumerate(results) if r > self.ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_results = [[self.classes[r[0]], r[1]] for r in results]
        print("%s \n classification: %s" % (sentence, return_results))
        return return_results
