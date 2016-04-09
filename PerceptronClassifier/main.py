import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def count_accuracy(train_markers, train_features, test_markers,
                   test_features, is_standard):
    if is_standard:
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

    clf = Perceptron()
    clf.fit(train_features, train_markers)
    predictions = clf.predict(test_features)
    accuracy = accuracy_score(test_markers, predictions)
    return accuracy


def main():
    train_data = np.loadtxt('perceptron-train.csv', delimiter=',')
    test_data = np.loadtxt('perceptron-test.csv', delimiter=',')
    train_markers = train_data[:, 0]
    train_features = train_data[:, 1:]
    test_markers = test_data[:, 0]
    test_features = test_data[:, 1:]
    acc_standard = count_accuracy(train_markers, train_features, test_markers,
                                  test_features, True)
    acc_not_standard = count_accuracy(train_markers, train_features,
                                      test_markers, test_features, False)
    acc_dif = acc_standard - acc_not_standard
    print(acc_dif)


if __name__ == "__main__":
    main()
