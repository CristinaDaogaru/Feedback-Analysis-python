from nltk import word_tokenize

import WordTokenizer


class FetureSet:


    def Find(self, allWords, number, document, ):

        word_features = list(allWords.keys())[:number]

        ### ////////

        # wordFeatureSet


    def FindFeatures(self, document, wordFeatureSet ):


        ########### Commented because of new files training data set #################

        # words = set(document)

        ########### End Commented because of new files training data set #################

        words = WordTokenizer.GetWords(document)

        #word_tokenize(document)

        features = {}

        # if the words w from the 3000 words appears in document then store true
        # store false otherwise
        for w in wordFeatureSet:
            features[w] = (w in words)

        return features












def find_features(document):

    ########### Commented because of new files training data set #################

    #words = set(document)

    ########### End Commented because of new files training data set #################

    words = word_tokenize(document)

    features = {}

    # if the words w from the 3000 words appears in document then store true
    # store false otherwise
    for w in word_features:
        features[w] = (w in words)

    return features

# search in the negative data set
# cv000_29416.txt is just a file from the corpora movie_reviews
#print( (find_features(movie_reviews.words('neg/cv000_29416.txt'))))


# find the top 3000 features and theier category
featuresets = [(find_features(rev), category) for (rev, category) in documents]

