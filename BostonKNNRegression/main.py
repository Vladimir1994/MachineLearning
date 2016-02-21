import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston
from sklearn.grid_search import GridSearchCV


def loadData():
    data = load_boston()
    markers = data.target
    features = data.data
    features = scale(features)
    return markers, features


def crossVal(markers, features):
    grid = {'p': np.linspace(1, 10, 200)}
    crossval = KFold(n=len(markers), n_folds=5, shuffle=True, random_state=42)

    reg = KNeighborsRegressor(n_neighbors=5, weights='distance',
                              metric='minkowski')
    gs = GridSearchCV(reg, grid, scoring='mean_squared_error', cv=crossval)
    gs.fit(features, markers)
    return gs.best_score_, gs.best_params_


def main():
    markers, features = loadData()
    minLoss, optimalP = crossVal(markers, features)
    print(minLoss, optimalP)


if __name__ == "__main__":
    main()
