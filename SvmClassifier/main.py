import numpy as np
from sklearn.svm import SVC


def main():
    data = np.loadtxt('svm-data.csv', delimiter=',')
    markers = data[:, 0]
    features = data[:, 1:]
    clf = SVC(C=100000, kernel='linear', random_state=241)
    clf.fit(features, markers)
    sup_idx = clf.support_
    print(" ".join([str(el + 1) for el in sup_idx]))


if __name__ == "__main__":
    main()
