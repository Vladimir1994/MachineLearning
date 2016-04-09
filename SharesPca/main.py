import numpy as np
from sklearn.decomposition import PCA


def main():
    features = np.loadtxt('close_prices.csv', delimiter=',',
                          usecols=range(1, 31), skiprows=1)
    pca = PCA(n_components=10)
    pca_features = pca.fit_transform(features)

    sum = 0
    cnt = 0
    while sum < 0.9:
        sum += pca.explained_variance_ratio_[cnt]
        cnt += 1
    print(cnt)

    djia_idx = np.loadtxt('djia_index.csv', delimiter=',',
                          usecols=[1], skiprows=1)
    first_component_vals = pca_features[:, 0]
    corr_matrix = np.corrcoef(djia_idx, first_component_vals)
    print(corr_matrix[0, 1])

    i = np.identity(features.shape[1])
    coef = pca.transform(i)
    print(np.argmax(coef[:, 0]) + 1)


if __name__ == "__main__":
    main()