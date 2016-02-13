import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import scale
from sklearn.grid_search import GridSearchCV

def preprocessData(data):
    markers = data[:, 0]
    features = data[:, 1:]
    features = scale(features)
    return markers, features


def crossVal(markers, features):
    grid = {'n_neighbors': range(1, 50)}
    crossval = KFold(n=len(markers), n_folds=5, shuffle=True, random_state=42)
    clf = KNeighborsClassifier()
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=crossval)
    gs.fit(features, markers)
    return gs.best_score_, gs.best_params_


def main():
    data = np.loadtxt('wine.csv', delimiter=',')
    markers, features = preprocessData(data)
    minLoss, optimalK = crossVal(markers, features)
    print(minLoss, optimalK)


if __name__ == "__main__":
    main()