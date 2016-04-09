import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston
from sklearn.grid_search import GridSearchCV


def load_data():
    data = load_boston()
    markers = data.target
    features = data.data
    features = scale(features)
    return markers, features


def cross_val(markers, features):
    grid = {'p': np.linspace(1, 10, 200)}
    crossval = KFold(n=len(markers), n_folds=5, shuffle=True, random_state=42)

    reg = KNeighborsRegressor(n_neighbors=5, weights='distance',
                              metric='minkowski')
    gs = GridSearchCV(reg, grid, scoring='mean_squared_error', cv=crossval)
    gs.fit(features, markers)
    return gs.best_score_, gs.best_params_


def main():
    markers, features = load_data()
    min_loss, p_optimal = cross_val(markers, features)
    print(min_loss, p_optimal)


if __name__ == "__main__":
    main()
