import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.preprocessing import scale
from sklearn.grid_search import GridSearchCV


def preprocess_data(data):
    markers = data[:, 0]
    features = data[:, 1:]
    features = scale(features)
    return markers, features


def cross_val(markers, features):
    grid = {'n_neighbors': range(1, 50)}
    cv = KFold(n=len(markers), n_folds=5, shuffle=True, random_state=42)
    clf = KNeighborsClassifier()
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
    gs.fit(features, markers)
    return gs.best_score_, gs.best_params_


def main():
    data = np.loadtxt('wine.csv', delimiter=',')
    markers, features = preprocess_data(data)
    min_loss, optimal_k = cross_val(markers, features)
    print(min_loss, optimal_k)


if __name__ == "__main__":
    main()