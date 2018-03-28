

class CollectionIntervalSplitter:


    def FirstElements(self, elements, number):

        return elements[:number]


    def LastElements(self, elements, number):

        return elements[number:]


    def BetweenElements(self, elements, startIndex, stopIndex):

        return elements[startIndex:stopIndex]



