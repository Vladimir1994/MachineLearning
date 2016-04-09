import numpy as np
from scipy.special import expit as sigmoid
from scipy.spatial import distance


class LogisticRegression:
    def __init__(self):
        self.coef_ = np.array([])

    def fit(self, features, markers, grad_step=0.1, reg_coef=0):
        if features.shape[0] != len(markers):
            raise ValueError("Features rows count must be equal to"
                             " markers length.")

        accuracy = 1e-5
        self.coef_ = np.zeros(features.shape[1])
        stop_cond = True

        while stop_cond:
            prev_coef = np.copy(self.coef_)
            for j in range(len(self.coef_)):
                sum = 0
                for i in range(features.shape[0]):
                    sum += markers[i] * features[i, j] * (1 -
                           sigmoid(markers[i] * np.dot(features[i],
                                                       prev_coef)))
                self.coef_[j] += grad_step / features.shape[0] * sum - \
                                 grad_step * reg_coef * prev_coef[j]
            stop_cond = distance.euclidean(prev_coef, self.coef_) > accuracy


    def predict(self, features):
        if len(features) != len(self.coef_):
            raise ValueError("Test object must have same features as"
                             " train objects.")

        return np.sign(np.dot(features, self.coef_))


    def count_score(self, features):
        y_score = np.array([])
        for f in features:
            y_score = np.append(y_score, sigmoid(np.dot(f, self.coef_)))
        return y_score

