import numpy as np
import pandas
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack


def preprocess_data(data):
    data['FullDescription'] = data['FullDescription'].str.lower()
    data['FullDescription'] = np.asarray(data['FullDescription'].
                                         replace('[^a-z0-9]', ' ',
                                                 regex=True))
    data['LocationNormalized'].fillna('nan', inplace=True)
    data['ContractTime'].fillna('nan', inplace=True)
    return data


def train(data):
    data = preprocess_data(data)
    vectorizer = TfidfVectorizer(min_df=5)
    full_description_vect = vectorizer.fit_transform(data['FullDescription'])

    encoder = DictVectorizer()
    features_categ = encoder.fit_transform(data[['LocationNormalized',
                                                 'ContractTime']].
                                           to_dict('records'))
    features = hstack((full_description_vect, features_categ))
    values = np.asarray(data['SalaryNormalized'])
    regression = Ridge(alpha=1)
    regression.fit(features, values)
    return regression, vectorizer, encoder


def predict(data, regression, vectorizer, encoder):
    data = preprocess_data(data)
    full_description_vect = vectorizer.transform(data['FullDescription'])
    features_categ = encoder.transform(data[['LocationNormalized',
                                             'ContractTime']].
                                       to_dict('records'))
    features = hstack((full_description_vect, features_categ))
    predicted_values = regression.predict(features)
    return predicted_values


def main():
    train_data = pandas.read_csv('salary-train.csv')
    regression, vectorizer, encoder = train(train_data)
    test_data = pandas.read_csv('salary-test-mini.csv')
    predicted_values = predict(test_data, regression, vectorizer, encoder)
    print(" ".join([str(el) for el in predicted_values]))


if __name__ == "__main__":
    main()

