import numpy as np
import Task_1 as t1
from matplotlib.image import imread
import matplotlib.pyplot as plt

def cumulativeGraph(cumulative, comp):
    points = np.array([[i, np.where(cumulative >= (i/100))[0][0] + 1] for i in range(1, 101)])

    fig, ax = plt.subplots()

    target_point = [points[comp - 1][1], points[comp - 1][0]]
    ax.axline(xy1=target_point, xy2=(0, target_point[1]), linestyle='--', color='red')
    ax.axline(xy1=target_point, xy2=(target_point[0], 0), linestyle='--', color='green')
    plt.annotate(f'{target_point[0]}', xy=target_point)

    ax.plot(points[:, 1], points[:, 0])

    plt.show()

def pca(image_bw, comp):
    # standardization of data
    image_standard = image_bw - np.mean(image_bw, axis=0)

    # find covariance matrix and eigen
    covar_image = np.cov(image_standard, rowvar=False)
    eigen = np.array(t1.getEigen(covar_image), dtype=object)

    # sort and take 95%
    indexes = np.argsort(eigen[:, 0])[::-1]
    eigen = eigen[indexes]

    cumulative = np.cumsum(eigen[:, 0]) / np.sum(eigen[:, 0])
    k = np.where(cumulative >= (comp/100))[0][0] + 1
    cumulativeGraph(cumulative, comp)

    transform_matrix = eigen[:, 1]
    transform_matrix = np.array([el for el in transform_matrix[:k]]).T

    final_image = image_standard @ transform_matrix
    final_image = final_image @ transform_matrix.T

    return final_image + np.mean(image_bw, axis=0)

def compressImage():
    # img loading and transform in matrix
    image_raw = imread(".\\img\\photo.jpg")
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
    new_image = pca(image_bw.T, 65)
    plt.imshow(new_image.T, cmap='gray')
    plt.show()

    new_image = pca(image_bw, 80)
    plt.imshow(new_image, cmap='gray')
    plt.show()

