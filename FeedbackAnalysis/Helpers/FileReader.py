
class FileReader:

    def ReadToEnd(self, filePath, format):

        return open(filePath, format).read()



    def ReadLines(self, filePath):

        file = open(filePath, "r")
        return list(file)


