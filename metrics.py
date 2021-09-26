# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import anndata
import scprep
import scanpy as sc
import sklearn
from sklearn.model_selection import train_test_split
import tempfile
import os
import scipy
from scipy import sparse
import torch
import umap
import matplotlib.pyplot as plt

#KNN-AUC
def knn_auc(adata, proportion_neighbors=0.1, n_svd=100):
    """Return KNN-AUC of the computed alignment for both modalities.
    
    KNN_AUC measures which fraction of  nearest neighbors of each 
    observation are preserved after the alignment.
    
    Args:
        adata: Anndata frame that contains the aligned representation for 
            both modalities.
        proportion_neighbors: the proportion of nearest neignbors among 
            total number of observations.
        n_svd: the dimension for PCA representation, used to calculated the
            true indices of nearest neighbors for each observation.
    
    Returns:
        area_under_curve: the KNN_AUC scores that quantify how much
            the alignment has preserved the original knn strucutre.
    """
    n_svd = min([n_svd, min(adata.X.shape)-1])
    n_neighbors = int(np.ceil(proportion_neighbors*adata.X.shape[0]))
    X_pca = sklearn.decomposition.TruncatedSVD(n_svd).fit_transform(adata.X)
    _, indices_true = (
        sklearn.neighbors.NearestNeighbors(n_neighbors = n_neighbors).fit(X_pca).kneighbors(X_pca)
    )
    _, indices_pred = (
        sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors).fit(adata.obsm["aligned"]).kneighbors(adata.obsm["mode2_aligned"])
    )
    neighbors_match = np.zeros(n_neighbors, dtype=int)
    for i in range(adata.shape[0]):
        _, pred_matches, true_matches = np.intersect1d(
            indices_pred[i], indices_true[i], return_indices=True
        )
        neighbors_match_idx = np.maximum(pred_matches, true_matches)
        neighbors_match += np.sum(np.arange(n_neighbors) >= neighbors_match_idx[:, None], axis = 0,)
    neighbors_match_curve = neighbors_match/(np.arange(1, n_neighbors + 1) * adata.shape[0])
    area_under_curve = np.mean(neighbors_match_curve)
    return area_under_curve

#MSE

def _square(X):
    if sparse.issparse(X):
        X.data = X.data ** 2
        return X
    else:
        return scprep.utils.toarray(X) ** 2

def mse(adata):
    """Measures how far away the paired observations is seperated in
    the aligned latent space.
    
    Args:
        adata: Anndata frame that contains the aligned representation for 
            both modalities.
    
    Returns: 
        The fraction of mse between paired observations and mse between
        the shuffled observations. Smaller value indicates that paired observations
        are aligned closely together.
    """
    X=scprep.utils.toarray(adata.obsm["aligned"])
    Y=scprep.utils.toarray(adata.obsm["mode2_aligned"])

    X_shuffled = X[np.random.permutation(np.arange(X.shape[0])), :]
    error_random = np.mean(np.sum(_square(X_shuffled - Y)))
    error_abs = np.mean(np.sum(_square(X - Y)))
    return error_abs/error_random


def plot_multimodal_umap(adata, title, num_points=None, connect_modalities=False):
    """Generates the UMAP plot for alignment.
    
    In the UMAP, one color corresponds to one modality, and the dots
    correspond to observations. The length of green dashed line visualizes
    how far sway the paired observations are in the latent space.
    
    Args:
        adata: Anndata frame that contains the aligned representation for 
            both modalities.
        num_points: the number of observations to be visualized.
        connect_modalities: whether to add dashed green lines to the plot.
    """
    X=scprep.utils.toarray(adata.obsm["aligned"][:num_points])
    Y=scprep.utils.toarray(adata.obsm["mode2_aligned"][:num_points])
    
    sizes = np.sum(_square(X - Y), axis=1)
    
    sizes = 100 * sizes / sizes.max() + 1
    reduced_data = umap.UMAP(random_state=42).fit_transform(np.vstack([X, Y]))[:, :2]
    X_reduced, Y_reduced = reduced_data[:len(X)], reduced_data[len(X):]

    plt.figure(figsize=(8, 6), dpi=80)
        
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=sizes, alpha=0.5, label="RNA-seq")
    plt.scatter(Y_reduced[:, 0], Y_reduced[:, 1], s=sizes, alpha=0.5, label="ATAC")
    if connect_modalities:
        x_coordinates = reduced_data[:, 0].reshape((2, len(X)))
        y_coordinates = reduced_data[:, 1].reshape((2, len(X)))
        plt.plot(x_coordinates, y_coordinates, '--', c="green", alpha=0.25)
    plt.title(title)
    plt.legend()
    
    plt.show()
