import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def countAccuracy(trainMarkers, trainFeatures, testMarkers,
                  testFeatures, isStandard):
    if isStandard:
        scaler = StandardScaler()
        trainFeatures = scaler.fit_transform(trainFeatures)
        testFeatures = scaler.transform(testFeatures)

    clf = Perceptron()
    clf.fit(trainFeatures, trainMarkers)
    predictions = clf.predict(testFeatures)
    accuracy = accuracy_score(testMarkers, predictions)
    return accuracy


def main():
    trainData = np.loadtxt('perceptron_train.csv', delimiter=',')
    testData = np.loadtxt('perceptron_test.csv', delimiter=',')
    trainMarkers = trainData[:, 0]
    trainFeatures = trainData[:, 1:]
    testMarkers = testData[:, 0]
    testFeatures = testData[:, 1:]
    accStandard = countAccuracy(trainMarkers, trainFeatures, testMarkers,
                  testFeatures, True)
    accNotStandard = countAccuracy(trainMarkers, trainFeatures, testMarkers,
                  testFeatures, False)
    accDif = accStandard - accNotStandard
    print(accDif)


if __name__ == "__main__":
    main()
