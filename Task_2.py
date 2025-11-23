import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def cumulativeGraph(cumulative, comp, percent):
    max_percent = int(cumulative[-1] * 100)
    points = np.array([[i, np.where(cumulative >= i / 100)[0][0] + 1] for i in range(1, max_percent + 1)])

    fig, ax = plt.subplots()

    target_point = [points[percent - 1][1], points[percent - 1][0]]
    ax.axline(xy1=target_point, xy2=(0, target_point[1]), linestyle='--', color='red')
    ax.axline(xy1=target_point, xy2=(target_point[0], 0), linestyle='--', color='green')
    plt.annotate(f'{target_point[0]}', xy=target_point)

    ax.plot(points[:, 1], points[:, 0])

    plt.show()

def compressImage():
    # img loading and transform in matrix
    image_raw = imread(".\\img\\photo.jpg")
    print(image_raw.shape)
    plt.imshow(image_raw)
    plt.show()

    # transformation into black-white
    image_sum = image_raw.sum(axis=2)
    print(image_sum.shape)
    image_bw = image_sum / image_sum.max()
    print(image_bw.max())

    plt.imshow(image_bw, cmap='gray')
    plt.show()

    # pca usage
    # for 95 %
    pca = PCA().fit(image_bw)
    cum_var = np.cumsum(pca.explained_variance_ratio_)

    k = np.where(cum_var >= 0.95)[0][0] + 1

    cumulativeGraph(cum_var, 0, 95)

    image_pca = PCA(n_components=k)
    reduced_image = image_pca.fit_transform(image_bw)
    reconstructed_image = image_pca.inverse_transform(reduced_image)

    plt.suptitle(f"Reconstruction {k} components", fontsize=16)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.show()

    # different components
    comp_list = [5, 25, 100, 180]

    for comp in comp_list:
        image_pca = PCA(n_components=comp)
        reduced_image = image_pca.fit_transform(image_bw)
        reconstructed_image = image_pca.inverse_transform(reduced_image)

        plt.suptitle(f"Reconstruction {comp} components", fontsize=16)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.show()
