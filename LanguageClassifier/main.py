import codecs
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import scipy.sparse as sp


def read_train_data():
    f = codecs.open("data/train.txt", "r", "utf-8")
    data = f.readlines()
    markers = list()
    corpus = list()
    for line in data:
        markers.append(line.split('\t')[0])
        corpus.append(prepare_line(line.split('\t')[1]))
    return markers, corpus


def read_test_data():
    test = codecs.open("data/test.txt", "r", "utf-8")
    test_data = test.readlines()
    return test_data


def prepare_line(line):
    line = re.sub('[^a-zA-Z0-9]', ' ', line)
    line.lower()
    return line


def train(markers, corpus):
    word_vectorizer = CountVectorizer(binary=True)
    char_vectorizer = CountVectorizer(analyzer='char')
    word_features = word_vectorizer.fit_transform(corpus)
    char_features = char_vectorizer.fit_transform(corpus)
    features = sp.hstack((word_features, char_features), format='csr')
    classifier = LinearSVC()
    classifier.fit(features, markers)
    return classifier, word_vectorizer, char_vectorizer


def classify(classifier, word_vectorizer, char_vectorizer, line):
    word_features = word_vectorizer.transform([line])
    char_features = char_vectorizer.transform([line])
    features = sp.hstack((word_features, char_features), format='csr')
    classificationResult = classifier.predict(features)
    return classificationResult[0]


def make_output(classifier, word_vectorizer, char_vectorizer, test_data):
    o = open("data/output.txt", "w")
    for line in test_data:
        o.write("%s\n" % classify(classifier, word_vectorizer, char_vectorizer,
                                  prepare_line(line)))
    o.close()


def main():
    markers, corpus = read_train_data()
    classifier, word_vectorizer, char_vectorizer = train(markers, corpus)
    test_data = read_test_data()
    make_output(classifier, word_vectorizer, char_vectorizer, test_data)


if __name__ == "__main__":
    main()