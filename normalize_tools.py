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
import inspect

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import functools

def normalizer(func, *args, **kwargs):
    @functools.wraps(func)
    def normalize(adata, *args, obsm=None, obs=None, var=None, **kwargs):
        if obsm is not None:
            cache_name = "{}_{}".format(obsm, func.__name__)
            if cache_name in adata.obsm:
                adata.obsm[obsm] = adata.obsm[cache_name]
            else:
                name_obs = obs
                name_var = var
                obs = adata.uns[obs] if obs else adata.obs
                var = adata.uns[var] if var else adata.var
                adata_temp = anndata.AnnData(adata.obsm[obsm], obs = obs, var = var)
                func(adata_temp, *args, **kwargs)
                adata.obsm[obsm] = adata.obsm[cache_name] = adata_temp.X
                adata.uns[name_var] = adata_temp.var
                adata.uns[name_obs] = adata_temp.obs
        else:
            if func.__name__ in adata.layers:
                adata.X = adata.layers[func.__name__]
            else:
                func(adata, *args, **kwargs)
                adata.layers[func.__name__] = adata.X
    return normalize

def _cpm(adata):
    adata.layers["count"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum = 1e6, key_added = "size_factors")

@normalizer
def log_cpm(adata):
    _cpm(adata)
    sc.pp.log1p(adata)
    
@normalizer
def sqrt_cpm(adata):
    cpm(adata)
    adata.X = scprep.transform.sqrt(adata.X)

@normalizer
def hvg_by_sc(adata, proportion = .1):
    n_genes = len(adata.var)
    n_genes_to_keep = int(n_genes*proportion)
    sc.pp.highly_variable_genes(adata, n_top_genes = n_genes_to_keep, subset=True)

@normalizer
def binarize(adata):
    adata.X[adata.X.nonzero()] = 1

@normalizer    
def filter_adata(
    adata,
    filter_gene_min_counts=5, 
    filter_gene_min_cells=5,
    filter_gene_max_cells=0.1,
):
    def ensure_count(value, max_val):
        retval = value
        if isinstance(value, float):
            retval = int(round(value * max_val))
        return retval
    
    sc.pp.filter_genes(adata, min_counts=filter_gene_min_counts)
    sc.pp.filter_genes(adata, min_cells=filter_gene_min_cells)
    sc.pp.filter_genes(
        adata, 
        max_cells=ensure_count(
            filter_gene_max_cells, max_val=adata.X.shape[0]
        ),
    )

@normalizer
def normalize_count_table(adata, size_factor=True, log_trans=True, normalize=True, proportion=None):
    
    #size normalize adata so that the total count per cell 
    #will be equal to median total count per cell
    if size_factor:
        n_counts = np.squeeze(np.array(adata.X.sum(axis=1)))
        sc.pp.normalize_total(adata, inplace=True)
        adata.obs["size_factors"] = n_counts / np.median(n_counts)
        adata.uns["median_counts"] = np.median(n_counts)
    else:
        adata.obs["size_factors"] = 1.0
        adata.uns["median_counts"] = 1.0
    
    if log_trans:
        sc.pp.log1p(adata, chunked=True, copy=False, chunk_size=10000)
    
    if proportion is not None:
        n_genes = len(adata.var)
        n_genes_to_keep = int(n_genes * proportion)
        sc.pp.highly_variable_genes(adata, n_top_genes = n_genes_to_keep, subset=True)
        
    if normalize:
        sc.pp.scale(adata, zero_center=True, copy=False)


def r_function(filename):
    curr_frame = inspect.currentframe()
    prev_frame = inspect.getframeinfo(curr_frame.f_back)
    filepath = os.path.join(os.path.dirname(prev_frame.filename), filename)
    print(filepath)
    with open(filepath, "r") as handle:
        r_code = handle.read()
    out_func = scprep.run.RFunction(setup="", args="sce", body=r_code)
    out_func.__r_file__ = filepath
    return out_func    

#scran normalization funciton
_scran = scprep.run.RFunction(
    setup="library('scran')",
    args="sce, min.mean=0.1",
    body="""
    sce <- computeSumFactors(sce, min.mean=min.mean, assay.type="X")
    sizeFactors(sce)
    """,
)
#log scran pooling
#dividing all counts for each cell by a cell-specific scaling factor(size factor)
#assumption:any cell-specific bias (e.g.,  amplification efficiency) affects all genes equally 
#via scaling of the expected mean count for that cell
@normalizer
def log_scran_pooling(adata):
    scprep.run.install_bioconductor("scran")
    adata.obs["size_factor"] = _scran(adata)

    #why here is multiplication instead of division
    adata.X = scprep.utils.matrix_vector_elementwise_multiply(
        adata.X, adata.obs["size_factor"], axis = 0
    )
    sc.pp.log1p(adata)