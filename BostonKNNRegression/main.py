import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston


def loadData():
    data = load_boston()
    markers = data.target
    features = data.data
    features = scale(features)
    return markers, features


def crossVal(markers, features):
    crossval = KFold(n=len(markers), n_folds=5, shuffle=True, random_state=42)
    kfoldResult = np.array([])
    for p in np.linspace(1, 10, 200):
        clf = KNeighborsRegressor(n_neighbors=5, weights='distance',
                                  metric='minkowski', p=p)
        meanScore = np.mean(cross_val_score(clf, features, markers,
                            cv=crossval, scoring='mean_squared_error'))
        kfoldResult = np.hstack((kfoldResult, meanScore))

    minLoss = kfoldResult.max()
    optimalP = kfoldResult.argmax() + 1
    return minLoss, optimalP


def main():
    markers, features = loadData()
    minLoss, optimalP = crossVal(markers, features)
    print(minLoss, optimalP)


if __name__ == "__main__":
    main()
