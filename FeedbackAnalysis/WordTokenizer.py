from nltk.tokenize import word_tokenize

class WordTokenizer:


    # Return a list with all the words contained in the text parameter
    # LowerCase parameter is optional and is by default true
    def GetWords(self, text, lowerCase = True):

        all_words = []
        words = self.WordTokenize(text)

        for w in words:

            if True == lowerCase:
                all_words.append(w.lower())
            else:
                all_words.append(w.upper())

        return all_words


    def WordTokenize(self, text):

        return word_tokenize(text)

