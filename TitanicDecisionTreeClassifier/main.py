import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def preprocess_data(data):
    data = data[['Survived', 'Sex', 'Age', 'Fare', 'Pclass']].dropna(axis=0)

    sex = list(data['Sex'])
    for i in range(len(sex)):
        if sex[i] == 'male':
            sex[i] = 1
        else:
            sex[i] = 0

    features = np.transpose(np.array([data['Pclass'], data['Fare'],
                            data['Age'], sex]))
    markers = np.transpose(np.array(data['Survived']))

    return markers, features


def main():
    data = pandas.read_csv('titanic.csv', index_col='PassengerId')
    markers, features = preprocess_data(data)
    clf = DecisionTreeClassifier()
    clf.fit(features, markers)
    importances = clf.feature_importances_
    print(importances)


if __name__ == "__main__":
    main()
