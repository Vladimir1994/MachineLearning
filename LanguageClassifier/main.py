import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import scipy.sparse as sp

def parseTrainData():
    f = codecs.open("data/train.txt", "r", "utf-8")
    data = f.readlines()
    markers = list()
    corpus = list()
    for line in data:
        markers.append(line.split('\t')[0])
        corpus.append(prepareLine(line.split('\t')[1]))
    return markers, corpus

def parseTestData():
    test = codecs.open("data/test.txt", "r", "utf-8")
    testData = test.readlines()
    return testData

def prepareLine(line):
    deletingSymbols = '1234567890":;@#$%^*()-_+=,.?!<>\\|/][{}&'
    for sym in deletingSymbols:
        line = line.replace(sym, ' ')
    line.lower()
    return line

def train(markers, corpus):
    wordVectorizer = CountVectorizer(binary = True)
    charVectorizer = CountVectorizer(analyzer='char')
    wordFeatures = wordVectorizer.fit_transform(corpus)
    charFeatures = charVectorizer.fit_transform(corpus)
    features = sp.hstack((wordFeatures, charFeatures), format='csr')
    classifier = LinearSVC()
    classifier.fit(features, markers)
    return classifier, wordVectorizer, charVectorizer

def classify(classifier, wordVectorizer, charVectorizer, line):
    wordFeatures = wordVectorizer.transform([line])
    charFeatures = charVectorizer.transform([line])
    features = sp.hstack((wordFeatures, charFeatures), format='csr')
    classificationResult = classifier.predict(features)
    return classificationResult[0]

def makeOutput(classifier, wordVectorizer, charVectorizer, testData):
    o = open("data/output.txt", "w")
    for line in testData:
        o.write("%s\n" % classify(classifier, wordVectorizer, charVectorizer, prepareLine(line)))
    o.close()

def main():
    markers, corpus = parseTrainData()
    classifier, wordVectorizer, charVectorizer = train(markers, corpus)
    testData = parseTestData()
    makeOutput(classifier, wordVectorizer, charVectorizer, testData)

if __name__ == "__main__":
    main()