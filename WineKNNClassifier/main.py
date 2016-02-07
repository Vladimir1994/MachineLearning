import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import scale


def preprocessData(data):
    markers = data[:, 0]
    features = data[:, 1:]
    features = scale(features)
    return markers, features


def crossVal(markers, features):
    crossval = KFold(n=len(markers), n_folds=5, shuffle=True, random_state=42)
    kfoldResult = np.array([])
    for k in range(1, 50):
        clf = KNeighborsClassifier(n_neighbors=k)
        meanScore = np.mean(cross_val_score(clf, features, markers,
                            cv=crossval))
        kfoldResult = np.hstack((kfoldResult, meanScore))

    minLoss = kfoldResult.max()
    optimalK = kfoldResult.argmax() + 1
    return minLoss, optimalK


def main():
    data = np.loadtxt('wine.csv', delimiter=',')
    markers, features = preprocessData(data)
    minLoss, optimalK = crossVal(markers, features)
    print(minLoss, optimalK)


if __name__ == "__main__":
    main()