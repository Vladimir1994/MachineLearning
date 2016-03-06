import pandas
import numpy as np
from matplotlib import pyplot
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss


def main():
    data = pandas.read_csv('gbm-data.csv').values
    features = data[:, 1:]
    markers = data[:, 0]
    features_train, features_test, markers_train, markers_test = \
        train_test_split(features, markers,
                         test_size=0.8,
                         random_state=241)

    learning_rates = [1, 0.5, 0.3, 0.2, 0.1]
    num_of_subplot = len(learning_rates) * 100 + 11
    test_loss_matrix = {}

    for learning_rate in learning_rates:
        clf = GradientBoostingClassifier(n_estimators=250, verbose=True,
                                         random_state=241,
                                         learning_rate=learning_rate)
        train_loss = []
        test_loss = []

        clf.fit(features_train, markers_train)

        for i, y_train_pred_proba in enumerate(
                clf.staged_predict_proba(features_train)):
            train_loss.append(log_loss(markers_train, y_train_pred_proba))

        for i, y_test_pred_proba in enumerate(
                clf.staged_predict_proba(features_test)):
            test_loss.append(log_loss(markers_test, y_test_pred_proba))
        test_loss_matrix[learning_rate] = test_loss

        pyplot.subplot(num_of_subplot)
        pyplot.plot(train_loss, 'g', linewidth=2)
        pyplot.plot(test_loss, 'r', linewidth=2)
        pyplot.legend(['train', 'test'])
        num_of_subplot += 1

    test_loss_02 = test_loss_matrix[0.2]
    argmin_test_loss = np.argmin(test_loss_02)
    min_test_loss = np.min(test_loss_02)
    print("test_loss min idx:", argmin_test_loss)
    print("test_loss min", min_test_loss)
    clf_rf = RandomForestClassifier(n_estimators=argmin_test_loss,
                                    random_state=241)
    clf_rf.fit(features_train, markers_train)
    y_pred_proba = clf_rf.predict_proba(features_test)
    print("log_loss for random forest", log_loss(markers_test, y_pred_proba))
    pyplot.show()


if __name__ == "__main__":
    main()