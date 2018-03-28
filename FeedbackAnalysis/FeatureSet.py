import WordTokenizer


class FetureSet:


    def Create(self, allWords, number, documents):

        wordFeatures = list(allWords.keys())[:number]
        featuresets = [(self.FindFeatures(rev, wordFeatures), category) for (rev, category) in documents]

        return featuresets


    def FindFeatures(self, document, wordFeatures):


        ########### Commented because of new files training data set #################

        # words = set(document)

        ########### End Commented because of new files training data set #################
        #
        # if type(document) is string:
        #     words = set(document)
        # else:

        words = WordTokenizer.GetWords(document, True)

        features = {}

        # if the words w from the 3000 words appears in document then store true
        # store false otherwise
        for w in wordFeatures:
            features[w] = (w in words)

        return features


