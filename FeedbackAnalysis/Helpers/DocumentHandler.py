import random


class DocumentHandler:

    def GetDocuments(self, nltkCorpusReviews):

        documents = [(list(nltkCorpusReviews.words(fileid)), category)
                      for category in nltkCorpusReviews.categories()
                      for fileid in nltkCorpusReviews.fileids(category)]

        return documents


    def GetDocuments(self, fileNamePositiveReviews, fileNameNegativeReviews, format):

        positiveReviews = open(fileNamePositiveReviews, format).read()
        negativeReviews = open(fileNameNegativeReviews, format).read()

        documents = []

        for review in positiveReviews.split('\n'):
            documents.append((review, "pos"))

        for review in negativeReviews.split('\n'):
            documents.append((review, "neg"))

        return documents


    def GetPositiveDocumets(self, fileName, format):

        positiveReviews = open(fileName, format).read()
        documents = []

        for review in positiveReviews.split('\n'):
            documents.append((review, "pos"))

        return documents


    def GetNegativeDocumets(self, fileName, format):

        negativeReviews = open(fileName, format).read()
        documents = []

        for review in negativeReviews.split('\n'):
            documents.append((review, "neg"))

        return documents


    def Shuffle(self, documents):

        random.shuffle(documents)
        return documents
