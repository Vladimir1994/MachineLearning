import numpy as np
from scipy.special import expit as sigmoid
from scipy.spatial import distance


class LogisticRegression:
    def __init__(self):
        self.coef_ = np.array([])

    def fit(self, features, markers, gradStep=0.1, regCoef=0):
        if features.shape[0] != len(markers):
            raise ValueError("Features rows count must be equal to"
                             " markers length.")

        accuracy = 1e-5
        self.coef_ = np.zeros(features.shape[1])
        stopCond = True

        while stopCond:
            prevCoef = np.copy(self.coef_)
            for j in range(len(self.coef_)):
                sum = 0
                for i in range(features.shape[0]):
                    sum += markers[i] * features[i, j] * (1 - \
                          sigmoid(markers[i] * np.dot(features[i], prevCoef)))
                self.coef_[j] += gradStep / features.shape[0] * sum - \
                                 gradStep * regCoef * prevCoef[j]
            stopCond = distance.euclidean(prevCoef, self.coef_) > accuracy


    def predict(self, features):
        if len(features) != len(self.coef_):
            raise ValueError("Test object must have same features as"
                             " train objects.")

        return np.sign(np.dot(features, self.coef_))


    def countScore(self, features):
        yScore = np.array([])
        for f in features:
            yScore = np.append(yScore, sigmoid(np.dot(f, self.coef_)))
        return yScore

    