import codecs
import json
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model


def classify(mail, vectorizer, classifier):
    features = vectorizer.transform([mail])
    res = classifier.predict(features)
    return res[0]


def make_output(test_dir, vectorizer, classifier):
    with codecs.open('test.txt', 'w', 'utf-8') as out:
        for f in os.listdir(test_dir):
            mail = json.load(open(os.path.join(test_dir, f)), 'utf-8')
            result = classify(mail['from'].encode('ascii', 'ignore') + ' ' +
                              mail['subject'].encode('ascii', 'ignore') + ' ' +
                              mail['body'].encode('ascii', 'ignore'),
                              vectorizer, classifier)
            out.write(u'%s\t%s\n' % (f, result))


def read_train(train_dir):
    for f in os.listdir(train_dir):
        with open(os.path.join(train_dir, f)) as fo:
            mail = json.load(fo, 'utf-8')
            yield mail


def train(train_mails):
    corpus = list()
    is_spam = list()

    for mail in train_mails:
        corpus.append(mail['from'].encode('ascii', 'ignore') + ' ' +
                      mail['subject'].encode('ascii', 'ignore') + ' ' +
                      mail['body'].encode('ascii', 'ignore'))
        is_spam.append(mail['is_spam'])
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(corpus)
    classifier = linear_model.LogisticRegression()
    classifier.fit(features, is_spam)
    return classifier, vectorizer


def main():
    train_mails = list(read_train('spam_data/train'))
    classifier, vectorizer = train(train_mails)
    make_output('spam_data/test', vectorizer, classifier)


if __name__ == '__main__':
    main()