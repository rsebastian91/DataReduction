# -*- coding: utf-8 -*-
"""

@author: robin
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

plt.close('all')


_COLORS = np.load('pca_toy_colors.npy')
pca_toy_4d = np.load("pca_toy_4d.npy")

y = pca_toy_4d[:, -1]  # labels
x = pca_toy_4d[:, :-1]  # 4D coordinates




def main():
    plot_slice(x, y, drop_dim=4)

    pca = PCA(n_components=2)

    xq = pca.fit_transform(x)

    xq = plot_2d(xq, y)



def plot_slice(x, y, drop_dim):
    """
    Plot a slice of the 4D input data from the toy data


    Input: - x: 4D coordinates
             y: labels
        
            drop_dim: dimension to omit from the 4D data
        
    """
    x = np.array(x)
    x_3d = np.delete(x, drop_dim - 1, axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    y = y.astype(int)
    if len(x_3d.shape) != 2 or len(y) != x.shape[0] or x_3d.shape[1] != 3:
        raise ValueError("Invalid input shape")
    ax.scatter(x_3d[:,0], x_3d[:,1], x_3d[:,2], c=_COLORS[y])
    plt.show()
    
def plot_2d(x, y):
    """
    Plot the low-dimensional representation of the toy data

    Input: - x: 2D representation
           - y: labels
           
    """
    x = np.array(x)
    y = np.array(y).reshape(-1).astype(int)
    fig = plt.figure()
    if len(x.shape) != 2 or len(y) != x.shape[0] or x.shape[1] != 2:
        raise ValueError("Invalid input shape")
    plt.scatter(x[:,0], x[:,1], c=_COLORS[y], s=30)
    plt.gca().set_aspect('equal')
    plt.show()
  

if __name__ == '__main__':
     main()
    