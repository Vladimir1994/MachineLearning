from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,\
    recall_score, f1_score, precision_recall_curve
import pandas
import numpy as np
import matplotlib.pyplot as plt


def TruePositiveRate(true, predicted):
    return np.sum((true == 1) * (predicted == 1))


def TrueNegativeRate(true, predicted):
    return np.sum((true == 0) * (predicted == 0))


def FalsePositiveRate(true, predicted):
    return np.sum((true == 0) * (predicted == 1))


def FalseNegativeRate(true, predicted):
    return np.sum((true == 1) * (predicted == 0))


def MaxPrecision(precision, recall, minRecall=0):
    return precision[(recall >= minRecall)].max()


def main():
    dataClf = pandas.read_csv('classification.csv')
    true = np.asarray(dataClf['true'])
    predicted = np.asarray(dataClf['pred'])
    print('TP: ' + str(TruePositiveRate(true, predicted)))
    print('FP: ' + str(FalsePositiveRate(true, predicted)))
    print('FN: ' + str(FalseNegativeRate(true, predicted)))
    print('TN: ' + str(TrueNegativeRate(true, predicted)))

    print('accuracy: ' + str(accuracy_score(true, predicted)))
    print('precision: ' + str(precision_score(true, predicted)))
    print('recall: ' + str(recall_score(true, predicted)))
    print('f1: ' + str(f1_score(true, predicted)))

    dataScores = pandas.read_csv('scores.csv')
    scoreTrue = np.asarray(dataScores['true'])
    scoreLogReg = np.asarray(dataScores['score_logreg'])
    scoreSVM = np.asarray(dataScores['score_svm'])
    scoreKNN = np.asarray(dataScores['score_knn'])
    scoreTree = np.asarray(dataScores['score_tree'])
    print('AUC-ROC LogReg: ' + str(roc_auc_score(true, scoreLogReg)))
    print('AUC-ROC SVM: ' + str(roc_auc_score(true, scoreSVM)))
    print('AUC-ROC KNN: ' + str(roc_auc_score(true, scoreKNN)))
    print('AUC-ROC Tree: ' + str(roc_auc_score(true, scoreTree)))

    print('Max precision if recall >= 0.7:')
    precisionLogReg, recallLogReg, thresholdsLogReg = \
        precision_recall_curve(scoreTrue, scoreLogReg)
    plt.plot(recallLogReg, precisionLogReg, label='LogReg')
    print('LogReg: ' + str(MaxPrecision(precisionLogReg,
                                      recallLogReg, minRecall=0.7)))
    
    precisionSVM, recallSVM, thresholdsSVM = \
        precision_recall_curve(scoreTrue, scoreSVM)
    plt.plot(recallSVM, precisionSVM, label='SVM')
    print('SVM: ' + str(MaxPrecision(precisionSVM, recallSVM, minRecall=0.7)))
    
    precisionKNN, recallKNN, thresholdsKNN = \
        precision_recall_curve(scoreTrue, scoreKNN)
    plt.plot(recallKNN, precisionKNN, label='KNN')
    print('KNN: ' + str(MaxPrecision(precisionKNN, recallKNN, minRecall=0.7)))
    
    precisionTree, recallTree, thresholdsTree = \
        precision_recall_curve(scoreTrue, scoreTree)
    plt.plot(recallTree, precisionTree, label='Tree')
    print('Tree: ' + str(MaxPrecision(precisionTree, recallTree,
                                      minRecall=0.7)))

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

