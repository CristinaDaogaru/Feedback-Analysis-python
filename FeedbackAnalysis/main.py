from Helpers.DocumentHandler import DocumentHandler


def main():

    # TODO : Refactor DocumentHandler class based on the fallowing code

    documentHandler = DocumentHandler()

    shortPositiveReviewsPath = "short_reviews/positive.txt"
    shortNegativeReviewsPath = "short_reviews/negative.txt"

    positiveDocumentsReview = documentHandler.GetPositiveDocumets(shortPositiveReviewsPath, "r")
    negativeDocumentsReview = documentHandler.GetNegativeDocumets(shortNegativeReviewsPath, "r")

    documents = []

    for r in positiveDocumentsReview.split('\n'):
        documents.append((r, "pos"))

    for r in negativeDocumentsReview.split('\n'):
        documents.append((r, "neg"))



if __name__ == '__main__':
    main()



