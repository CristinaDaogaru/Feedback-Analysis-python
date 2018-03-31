from WordTokenizer import WordTokenizer


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
        #
        # if type(document) is string:
        #     words = set(document)
        # else:


        wordTokenizer = WordTokenizer()
        words = wordTokenizer.WordTokenize(document)

        features = {}

        # if the words w from the 3000 words appears in document then store true
        # store false otherwise
        for w in self.wordFeatures:
            features[w] = (w in words)

        return features
