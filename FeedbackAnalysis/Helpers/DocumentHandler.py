import random


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
