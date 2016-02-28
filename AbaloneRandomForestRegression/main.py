import pandas
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV


def main():
    data = pandas.read_csv('abalone.csv')
    data['Sex'] = data['Sex'].map(lambda x:
                                  1 if x == 'M' else (-1 if x == 'F' else 0))
    features = np.asarray(data.ix[:, :'ShellWeight'])
    markers = np.asarray(data['Rings'])
    grid = {'n_estimators': range(1, 51)}
    cv = KFold(n=len(markers), n_folds=5, shuffle=True, random_state=1)
    reg = RandomForestRegressor(random_state=1)
    gs = GridSearchCV(reg, grid, scoring='r2', cv=cv)
    gs.fit(features, markers)

    means = [score_tuple[1] for score_tuple in gs.grid_scores_]
    n_estimators = np.array(np.where(np.asarray(means) > 0.52)) + 1
    print(n_estimators[0][0])


if __name__ == "__main__":
    main()

