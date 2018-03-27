from nltk.tokenize import word_tokenize

class WordTokenizer:


    def GetWords(self, text, lowerCase):

        all_words = []
        words = word_tokenize(text)

        for w in words:

            if True == lowerCase:
                all_words.append(w.lower())
            else:
                all_words.append(w.upper())

        return all_words



