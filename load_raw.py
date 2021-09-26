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
import sys
import scipy
from scipy import sparse

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

import matplotlib.pyplot as plt


def load_raw_cell_lines():
    """Load raw data of scicar cell lines from URLs.
    
    Load RNA, ATAC observations into scipy sparse matrices,
    the genes and cells information into panda dataframe.
    
    Returns:
        rna_data: the scipy CSR matrix for raw scicar RNA-seq observations, 
            arranged by "cells x genes".
        atac_data: the scipy CSR matrix for raw scicar ATAC observations, 
            arranged by "cells x peaks".
        rna_genes: panda dataframe that record the gene information in RNA-seq.
        atac_genes: panda dataframe that record the gene information in ATAC-seq.
        rna_cells: panda dataframe that record the cell information in RNA-seq.
        atac_cells: panda dataframe that record the cell information in ATAC-seq.
    """
    rna_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM3271040"
        "&format=file&file=GSM3271040%5FRNA%5FsciCAR%5FA549%5Fgene%5Fcount.txt.gz")
    rna_cells_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM3271040"
         "&format=file&file=GSM3271040%5FRNA%5FsciCAR%5FA549%5Fcell.txt.gz"
    )
    rna_genes_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM3271040"
        "&format=file&file=GSM3271040%5FRNA%5FsciCAR%5FA549%5Fgene.txt.gz"
    )
    atac_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM3271041"
        "&format=file&file=GSM3271041%5FATAC%5FsciCAR%5FA549%5Fpeak%5Fcount.txt.gz"
    )
    atac_cells_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM3271041"
        "&format=file&file=GSM3271041%5FATAC%5FsciCAR%5FA549%5Fcell.txt.gz"
    )
    atac_genes_url = (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM3271041"
        "&format=file&file=GSM3271041%5FATAC%5FsciCAR%5FA549%5Fpeak.txt.gz"
    )
    rna_genes = pd.read_csv(rna_genes_url, low_memory=False, index_col=0)
    atac_genes =  pd.read_csv(atac_genes_url, low_memory=False, index_col=1)
    rna_cells = pd.read_csv(rna_cells_url, low_memory=False, index_col=0)
    atac_cells = pd.read_csv(atac_cells_url, low_memory=False, index_col=0)

    with tempfile.TemporaryDirectory() as tempdir:
        rna_file = os.path.join(tempdir, "rna.mtx.gz")
        scprep.io.download.download_url(rna_url, rna_file)
        rna_data = scprep.io.load_mtx(rna_file, cell_axis="col").tocsr()
        atac_file = os.path.join(tempdir, "atac.mtx.gz")
        scprep.io.download.download_url(atac_url, atac_file)
        atac_data = scprep.io.load_mtx(atac_file, cell_axis="col").tocsr()
    return rna_data, atac_data, rna_cells, atac_cells, rna_genes, atac_genes


def create_joint_dataset(X, Y, X_index=None, X_columns=None, Y_index=None, Y_columns=None):
    """Keep observations for the same cell type and create the joint Anndata frame.
        
    Selects the observations of the common cell types in RNA-seq and 
    ATAC-seq raw data and merge them into an AnnData dataframe.
    
    Args:
        X: the raw scipy CSR matrix for RNA-seq raw observation data (n_cells x n_genes).
        Y: the raw scipy CSR matrix for ATAC-seq raw observation data (n_cells x n_peaks).
        X_index: cell types for RNA-seq observations.
        X_columns: gene names for RNA-seq observations.
        Y_index: cell types for ATAC-seq observations.
        Y_columns: gene names for ATAC-seq observation.
        
    Returns:
        adata: the merged AnnData dataframe for common observations of RNA-seq and 
            ATAC-seq, where RNA-seq info is in X and ATAC-seq info in obsm["mode_2"].
        joint_index: ndarray of the common cell type, sorted in ascending order.
    """
    if X_index is None:
        X_index = X.index
    if X_columns is None:
        X_columns = X.columns
    if Y_index is None:
        Y_index = Y.index
    if Y_columns is None:
        Y_columns = Y.columns
    joint_index = np.sort(np.intersect1d(X_index, Y_index))
    try:
        X = X.loc[joint_index]
        Y = Y.loc[joint_index]
    except AttributeError:
        x_keep_idx = np.isin(X_index, joint_index)
        y_keep_idx = np.isin(Y_index, joint_index)
        X = X[x_keep_idx]
        Y = Y[y_keep_idx]
        X_index_sub = scprep.utils.toarray(X_index[x_keep_idx])
        Y_index_sub = scprep.utils.toarray(Y_index[y_keep_idx])
        X = X[np.argsort(X_index_sub)]
        Y = Y[np.argsort(Y_index_sub)]
        # check order is correct
        assert (X_index_sub[np.argsort(X_index_sub)] == joint_index).all()
        assert (Y_index_sub[np.argsort(Y_index_sub)] == joint_index).all()
    adata = anndata.AnnData(
        scprep.utils.to_array_or_spmatrix(X).tocsr(),
        obs = pd.DataFrame(index = joint_index),
        var = pd.DataFrame(index = X_columns),
    )
    adata.obsm["mode2"] = scprep.utils.to_array_or_spmatrix(Y).tocsr()
    adata.uns["mode2_obs"] = joint_index
    adata.uns["mode2_var"] = scprep.utils.toarray(Y_columns)
    return adata, joint_index


def subset_mode2_genes(adata, keep_genes):
    """Select the kept genes in ATAC-seq data.
    
    Args:
        adata: the AnnData dataframe from which to filter genes.
        keep_genes: the genes to be kept in modalities two (ATAC-seq observation).
    
    Return:
        adata: the AnnData dataframe where ATAC-seq observations have been filtered.
    """
    adata.obsm["mode2"] = adata.obsm["mode2"][:, keep_genes]
    adata.uns["mode2_var"] = adata.uns["mode2_var"][keep_genes]
    if "mode2_varnames" in adata.uns:
        for varname in adata.uns["mode2_varnames"]:
            adata.uns[varname] = adata.uns[varname][keep_genes]
    return adata


def filter_joint_data_empty_cells(adata):
    """Filters out the empty observations in merged AnnData.
    
    If one cell observation has all zero entries in either modality
    (RNA-seq or ATAC-seq), it will be filtered out.
    
    Args:
        adata: the raw joint AnnData dataframe that merges both RNA-seq and ATAC-seq.
        
    Returns:
        adata: anndata frame where empty observations are fitlered out.
        keep_cells: the indices of kept cells.
    """
    assert np.all(adata.uns["mode2_obs"] == adata.obs.index)
    #filter out cells
    n_cells_mode1 = scprep.utils.toarray(adata.X.sum(axis = 1)).flatten()
    n_cells_mode2 = scprep.utils.toarray(adata.obsm["mode2"].sum(axis = 1)).flatten()
    keep_cells = np.minimum(n_cells_mode1, n_cells_mode2) > 1
    adata.uns["mode2_obs"] = adata.uns["mode2_obs"][keep_cells]
    adata = adata[keep_cells, :].copy()
    #filter out genes
    sc.pp.filter_genes(adata, min_counts=1)
    n_genes_mode2 = scprep.utils.toarray(adata.obsm["mode2"].sum(axis=0)).flatten()
    keep_genes_mode2 = n_genes_mode2 > 0
    adata = subset_mode2_genes(adata, keep_genes_mode2)
    return adata, keep_cells


def merge_data(rna_data, atac_data, rna_cells, atac_cells, rna_genes, atac_genes):
    """Merges raw data into a preprocessed anndata frame.
    
    Merge raw data of RNA-seq and ATAC-seq into a Anndata frame, 
    and filtered out the empty observations.
    
    Args:
        rna_data: the scipy CSR matrix for raw scicar RNA-seq observations, 
            arranged by "cells x genes".
        atac_data: the scipy CSR matrix for raw scicar ATAC observations, 
            arranged by "cells x peaks".
        rna_genes: panda dataframe that record the gene information in RNA-seq.
        atac_genes: panda dataframe that record the gene information in ATAC-seq.
        rna_cells: panda dataframe that record the cell information in RNA-seq.
        atac_cells: panda dataframe that record the cell information in ATAC-seq.
        
    Returns:
        scicar_data: merged joint anndata frame with empty observations removed.
        joint_index: the indices of common observations for both modalities.
        keep_cells_idx: the indices of the kept (non-empty) cell observations in the 
            joint dataset.
    """
    scicar_data, joint_index = create_joint_dataset(
        rna_data, atac_data, 
        X_index=rna_cells.index, 
        X_columns=rna_genes.index, 
        Y_index=atac_cells.index,
        Y_columns=atac_genes.index
    )
    scicar_data.obs = rna_cells.loc[scicar_data.obs.index]
    scicar_data.var = rna_genes
    for key in atac_cells.columns:
        scicar_data.obs[key] = atac_cells[key]
    scicar_data.uns["mode2_varnames"] = []
    for key in atac_genes.columns:
        varname = "mode2_var_{}".format(key)
        scicar_data.uns[varname] = atac_genes[key].values
        scicar_data.uns["mode2_varnames"].append(varname)
    scicar_data, keep_cells_idx = filter_joint_data_empty_cells(scicar_data)
    return scicar_data, joint_index, keep_cells_idx


def subset_joint_data(adata, n_cells=600, n_genes=1500):
    """Randomly selects a subset form the joint dataframe.
    
    First randomly selects n_cells common observations from both modalities,
    then randomly chooses n_genes from the selected observations. Used for
    evaluation and testing. Note that subset_joint_data will filter out empty
    observations, so the final number of cells and genes in the subset might
    be smaller than argument inputs.
    
    Args:
        adata: the merge data from which to select the subset.
        n_cells: number of observations(cell types) to select.
        n_genes: number of genes to select.
    
    Returns: 
        adata: the randomly selected subset in AnnData.
    """
    if adata.shape[0] > n_cells:
        keep_cells = np.random.choice(adata.shape[0], n_cells, replace=False)
        adata = adata[keep_cells].copy()
        adata.uns["mode2_obs"] = adata.uns["mode2_obs"][keep_cells]
        adata, _ = filter_joint_data_empty_cells(adata)
    if adata.shape[1] > n_genes:
        keep_mode1_genes = np.random.choice(adata.shape[1], n_genes, replace=False)
        adata = adata[:, keep_mode1_genes].copy()

    if adata.obsm["mode2"].shape[1] > n_genes:
        keep_genes_mode2 = np.random.choice(
            adata.obsm["mode2"].shape[1], n_genes, replace=False
        )
        adata = subset_mode2_genes(adata, keep_genes_mode2)
    adata, _ = filter_joint_data_empty_cells(adata)
    return adata


def train_test_split(adata, train_prop=0.7):
    """Splits the joint dataset into train subset and test subset according
    to certain proportion.
    
    Args:
        adata: the dataframe from which to split.
        train_prop: the proportion of training data in input dataset.
        
    Returns:
        train_adata: the anndata dataframe that contains the training data.
        test_adata: the anndata dataframe that contains the testing data.
    """
    n_train = int(adata.shape[0]*0.7)
    n_test = adata.shape[0] - n_train

    train_cells = np.random.choice(adata.shape[0], n_train, replace=False)
    test_mask = np.isin(np.arange(adata.shape[0]), train_cells, invert=True)
    train_adata = adata[train_cells].copy()
    train_adata.uns["mode2_obs"] = adata.uns["mode2_obs"][train_cells]
    test_adata = adata[test_mask].copy()
    test_adata.uns["mode2_obs"] = adata.uns["mode2_obs"][test_mask]
    return train_adata, test_adata, train_cells, test_mask


def split_with_mask(adata, indices_train, test_mask):
    train_adata = adata[indices_train].copy()
    train_adata.uns["mode2_obs"] = adata.uns["mode2_obs"][indices_train]
    test_adata = adata[test_mask].copy()
    test_adata.uns["mode2_obs"] = adata.uns["mode2_obs"][test_mask]
    return train_adata, test_adata

def split_mask_single(adata, indices_train, test_mask):
    train_adata = adata[indices_train].copy()
    test_adata = adata[test_mask].copy()
    return train_adata, test_adata