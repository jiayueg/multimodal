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