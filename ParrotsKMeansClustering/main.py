import numpy as np
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage import img_as_float

from measure import compare_psnr


def main():
    image = imread('parrots.jpg')
    image = img_as_float(image)
    features = image.reshape((-1, image.shape[-1]))
    psnr_mean = []
    psnr_median = []

    for n_clusters in range(2, 21):
        kmeans = KMeans(n_clusters=n_clusters, random_state=241)
        markers = kmeans.fit_predict(features)

        seg_image_mean = np.zeros_like(image)
        seg_image_median = np.zeros_like(image)
        for cluster_marker in range(n_clusters):
            mean_col = np.mean(features[markers == cluster_marker], axis=0)
            median_col = np.median(features[markers == cluster_marker], axis=0)
            seg_mask = markers.reshape((image.shape[0], image.shape[1])) == \
                cluster_marker
            seg_image_mean[seg_mask] = mean_col
            seg_image_median[seg_mask] = median_col

        psnr_mean.append(compare_psnr(image, seg_image_mean))
        psnr_median.append(compare_psnr(image, seg_image_median))

    minimal_required_prsne = 20
    print((np.argwhere(psnr_mean > minimal_required_prsne))[0])


if __name__ == "__main__":
    main()

    