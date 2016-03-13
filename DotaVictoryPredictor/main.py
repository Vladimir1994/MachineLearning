import pandas
import numpy as np
import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression


def preprocess_features(features):
    features = features.fillna(0)
    drop_cols = ['duration', 'tower_status_radiant',
                 'tower_status_dire', 'barracks_status_radiant',
                 'barracks_status_dire']  # target and future variables
    features = features.drop(drop_cols, 1)
    return features


def remove_categorical_features(features):
    categorial_features = ['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero',
                           'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero',
                           'd3_hero', 'd4_hero', 'd5_hero']
    return features.drop(categorial_features, 1)


def add_bag_of_words(features, features_remcat, unique_names):
    features_pick = np.zeros((features.shape[0], max(unique_names)))
    for i, match_id in enumerate(features.index):
        for p in range(0, 5):
            features_pick[i, features.ix[match_id,
                                         'r%d_hero' % (p + 1)] - 1] = 1
            features_pick[i, features.ix[match_id,
                                         'd%d_hero' % (p + 1)] - 1] = -1
    features_remcat = np.hstack([features_remcat, features_pick])
    return features_remcat


def gbm_task(features, markers, data):
    print("Task 1.1")
    columns_contains_null = data.columns.values[data.isnull().
                                                any(axis=0).values]
    print("Columns contains_null names:")
    print(" ".join([str(el) for el in columns_contains_null]))
    print("")

    print("Task 1.2 \n"
          "Target variable: radiant_win. \n")

    print("Task 1.3")
    clf = GradientBoostingClassifier(n_estimators=30)
    cv = KFold(n=markers.shape[0], n_folds=5, shuffle=True)
    start_time = datetime.datetime.now()
    score = cross_val_score(estimator=clf, X=features, y=markers,
                            scoring='roc_auc', cv=cv)
    t = datetime.datetime.now() - start_time
    print("Cross validation time: " + str(t))
    print("Cross validation auc-roc mean score: " + str(score.mean()))
    print("")

    print("Task 1.4")
    clf_new = GradientBoostingClassifier()  # n_estimators=100
    score_new = cross_val_score(estimator=clf_new, X=features, y=markers,
                                scoring='roc_auc', cv=cv)
    print("Cross validation auc-roc mean score: " + str(score_new.mean()))
    print("")


def logitreg_task(features, markers):
    scaler = StandardScaler()
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cv = KFold(markers.shape[0], n_folds=5, shuffle=True)
    clf = LogisticRegression()
    gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=cv)

    print("Task 2.1")
    features_scaled = scaler.fit_transform(features)
    start_time = datetime.datetime.now()
    gs.fit(features_scaled, markers)
    t = datetime.datetime.now() - start_time
    print("Cross validation time: " + str(t))
    print("Cross validation auc-roc mean score: " + str(gs.best_score_))
    print("")

    print("Task 2.2")
    features_remcat = remove_categorical_features(features)
    features_remcat_scaled = scaler.fit_transform(features_remcat)
    start_time = datetime.datetime.now()
    gs.fit(features_remcat_scaled, markers)
    t = datetime.datetime.now() - start_time
    print("Cross validation time: " + str(t))
    print("Cross validation auc-roc mean score: " + str(gs.best_score_))
    print("")

    print("Task 2.3")
    hero_name_features = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero',
                          'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero',
                          'd4_hero', 'd5_hero']
    unique_names = set()
    for name in hero_name_features:
        unique_names = unique_names.union(
            set(features[name].unique()))
    print("Unique hero names in training sample count: " +
          str(len(unique_names)))
    print("Unique hero names in Dota 2 count: " +
          str(max(unique_names)))
    print("")

    print("Task 2.4")
    features_remcat = remove_categorical_features(features)
    features_remcat_bw = add_bag_of_words(features, features_remcat,
                                          unique_names)
    features_remcat_wp_scaled = scaler.fit_transform(features_remcat_bw)
    start_time = datetime.datetime.now()
    gs.fit(features_remcat_wp_scaled, markers)
    t = datetime.datetime.now() - start_time
    print("Cross validation time: " + str(t))
    print("Cross validation auc-roc mean score: " + str(gs.best_score_))
    print("")

    print("Task 2.5.")
    features_test = pandas.read_csv('features_test.csv', index_col='match_id')
    features_test = features_test.fillna(0)
    features_test_remcat = remove_categorical_features(features_test)
    features_test_remcat_bw = add_bag_of_words(features_test,
                                               features_test_remcat,
                                               unique_names)
    features_test_remcat_wp_scaled = \
        scaler.fit_transform(features_test_remcat_bw)
    markers_pred = gs.best_estimator_.\
        predict_proba(features_test_remcat_wp_scaled)
    print("Maximum prediction value: " + str(max(markers_pred[:, 0])))
    print("Minimum prediction value: " + str(min(markers_pred[:, 0])))


def main():
    data = pandas.read_csv('features.csv', index_col='match_id')
    markers = data['radiant_win']
    features = data.drop('radiant_win', axis=1)
    features = preprocess_features(features)

    gbm_task(features, markers, data)
    logitreg_task(features, markers)


if __name__ == "__main__":
    main()