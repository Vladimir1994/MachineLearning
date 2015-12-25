import codecs
import json
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model

def classify(mail, vectorizer, logreg):
    vect_mail = vectorizer.transform([mail])
    res = logreg.predict(vect_mail)
    return res[0]
 
def make_output(test_dir, vectorizer, logreg):
    with codecs.open('test.txt', 'w', 'utf-8') as out:
        for f in os.listdir(test_dir):
            mail = json.load(open(os.path.join(test_dir, f)), 'utf-8')
            result = classify(mail['from'].encode('ascii','ignore') + ' ' +
                              mail['subject'].encode('ascii','ignore') + ' ' +
                              mail['body'].encode('ascii','ignore'), vectorizer, logreg)
            out.write(u'%s\t%s\n' % (f, result))

def read_train(train_dir):
    for f in os.listdir(train_dir):
        with open(os.path.join(train_dir, f), 'r') as fo:
            mail = json.load(fo, 'utf-8')
            yield mail

if __name__ == '__main__':
    train_mails = list(read_train('spam_data/train'))
    corpus = list()
    is_spam = list()

    for mail in train_mails:
        corpus.append(mail['from'].encode('ascii','ignore') + ' ' +
                      mail['subject'].encode('ascii','ignore') + ' ' +
                      mail['body'].encode('ascii','ignore'))
        is_spam.append(mail['is_spam'])
    vectorizer = CountVectorizer()
    cnt_vect = vectorizer.fit_transform(corpus)
    logreg = linear_model.LogisticRegression()
    logreg.fit(cnt_vect, is_spam)
    make_output('spam_data/test', vectorizer, logreg)