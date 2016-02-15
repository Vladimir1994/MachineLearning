import numpy as np
from sklearn.metrics import roc_auc_score

import LogisticRegression


def main():
    data = np.loadtxt('data-logistic.csv', delimiter=',')
    markers = data[:, 0]
    features = data[:, 1:]
    lg = LogisticRegression.LogisticRegression()
    lg.fit(features, markers, regCoef=0)
    yScore = lg.countScore(features)
    print(roc_auc_score(markers, yScore))

    lgL2 = LogisticRegression.LogisticRegression()
    lgL2.fit(features, markers, regCoef=10)
    yScoreL2 = lgL2.countScore(features)
    print(roc_auc_score(markers, yScoreL2))


if __name__ == "__main__":
    main()
