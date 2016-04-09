import numpy as np
from sklearn.metrics import roc_auc_score

import LogisticRegression


def main():
    data = np.loadtxt('data-logistic.csv', delimiter=',')
    markers = data[:, 0]
    features = data[:, 1:]
    logreg = LogisticRegression.LogisticRegression()
    logreg.fit(features, markers)
    y_score = logreg.count_score(features)
    print(roc_auc_score(markers, y_score))

    logreg_l2 = LogisticRegression.LogisticRegression()
    logreg_l2.fit(features, markers, reg_coef=10)
    y_score_l2 = logreg_l2.count_score(features)
    print(roc_auc_score(markers, y_score_l2))


if __name__ == "__main__":
    main()

