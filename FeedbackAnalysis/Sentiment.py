from FeatureSet import FeatureSet


class Sentiment:

    def __init__(self, classifier, wordFeatures):

        self.votedClassifier = classifier
        self.featureSet = FeatureSet(wordFeatures)

    def FindSentiment(self, text):

        features = self.featureSet.FindFeatures(text)
        return self.votedClassifier.classify(features) #, self.votedClassifier.confidence(features)