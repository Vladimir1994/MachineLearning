from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,\
    recall_score, f1_score, precision_recall_curve
import pandas
import numpy as np
import matplotlib.pyplot as plt


def true_positive_rate(true, predicted):
    return np.sum((true == 1) * (predicted == 1))


def true_negative_rate(true, predicted):
    return np.sum((true == 0) * (predicted == 0))


def false_positive_rate(true, predicted):
    return np.sum((true == 0) * (predicted == 1))


def false_negative_rate(true, predicted):
    return np.sum((true == 1) * (predicted == 0))


def max_precision(precision, recall, min_recall=0):
    return precision[(recall >= min_recall)].max()


def main():
    data = pandas.read_csv('classification.csv')
    true_data = np.asarray(data['true'])
    predicted_data = np.asarray(data['pred'])
    print('TP: ' + str(true_positive_rate(true_data, predicted_data)))
    print('FP: ' + str(false_positive_rate(true_data, predicted_data)))
    print('FN: ' + str(false_negative_rate(true_data, predicted_data)))
    print('TN: ' + str(true_negative_rate(true_data, predicted_data)))

    print('accuracy: ' + str(accuracy_score(true_data, predicted_data)))
    print('precision: ' + str(precision_score(true_data, predicted_data)))
    print('recall: ' + str(recall_score(true_data, predicted_data)))
    print('f1: ' + str(f1_score(true_data, predicted_data)))

    data_scores = pandas.read_csv('scores.csv')
    score_true = np.asarray(data_scores['true'])
    score_logreg = np.asarray(data_scores['score_logreg'])
    score_svm = np.asarray(data_scores['score_svm'])
    score_knn = np.asarray(data_scores['score_knn'])
    score_tree = np.asarray(data_scores['score_tree'])
    print('AUC-ROC LogReg: ' + str(roc_auc_score(true_data, score_logreg)))
    print('AUC-ROC SVM: ' + str(roc_auc_score(true_data, score_svm)))
    print('AUC-ROC KNN: ' + str(roc_auc_score(true_data, score_knn)))
    print('AUC-ROC Tree: ' + str(roc_auc_score(true_data, score_tree)))

    print('Max precision if recall >= 0.7:')
    precision_logreg, recall_logreg, thresholds_logreg = \
        precision_recall_curve(score_true, score_logreg)
    plt.plot(recall_logreg, precision_logreg, label='LogReg')
    print('LogReg: ' + str(max_precision(precision_logreg,
                                         recall_logreg, min_recall=0.7)))
    
    precision_svm, recall_svm, thresholds_svm = \
        precision_recall_curve(score_true, score_svm)
    plt.plot(recall_svm, precision_svm, label='SVM')
    print('SVM: ' + str(max_precision(precision_svm, recall_svm,
                                      min_recall=0.7)))
    
    precision_knn, recall_knn, thresholds_knn = \
        precision_recall_curve(score_true, score_knn)
    plt.plot(recall_knn, precision_knn, label='KNN')
    print('KNN: ' + str(max_precision(precision_knn, recall_knn,
                                      min_recall=0.7)))
    
    precision_tree, recall_tree, thresholds_tree = \
        precision_recall_curve(score_true, score_tree)
    plt.plot(recall_tree, precision_tree, label='Tree')
    print('Tree: ' + str(max_precision(precision_tree, recall_tree,
                                       min_recall=0.7)))

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

