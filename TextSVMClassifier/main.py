import numpy
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV


def readData():
    newsgroups = datasets.fetch_20newsgroups(subset='all',
                                             categories=['alt.atheism',
                                                         'sci.space'])
    corpus = newsgroups.data
    markers = newsgroups.target
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(corpus)
    return markers, features, vectorizer


def main():
    markers, features, vectorizer = readData()
    grid = {'C': numpy.power(10.0, numpy.arange(-5, 6))}
    cv = KFold(markers.size, n_folds=5, shuffle=True, random_state=241)
    clf = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
    gs.fit(features, markers)
    print(gs.best_score_)
    print(gs.best_params_)
    scoresAbs = abs(gs.best_estimator_.coef_.data)
    sortedScoresIndices = gs.best_estimator_.coef_.indices[scoresAbs.argsort()]
    top10Indices = sortedScoresIndices[-10:]
    featureNames = numpy.asarray(vectorizer.get_feature_names())[top10Indices]
    print " ".join(numpy.sort(featureNames))


if __name__ == "__main__":
    main()