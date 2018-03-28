import nltk

class WordsHandler:

    def FrequancyDistributions(self, words):

        return nltk.FreqDist(words)


    def MostCommonWords(self, words, number):

        return words.most_common(number)


    def WordFrequency(self, words, word):

        return words[word]

