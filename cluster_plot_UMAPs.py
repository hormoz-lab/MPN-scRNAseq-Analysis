#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:41:21 2020

@author: dve
"""

from scipy import stats, sparse
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import loompy, h5py, sys, os, random, csv
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from functools import reduce
import anndata as ad

# HELPER FUNCTIONS
def merge_scanpy(to_merge, sample_ids):
    first = True
    for data in to_merge:
        if first:
            genes_to_use = data.var.index
            first = False
        else:
            genes_to_use = genes_to_use.intersection(data.var.index)
    
    aligned = []
    raws = []
    for data in to_merge:
        to_append = data[:, genes_to_use]
        indices = [np.nonzero(np.isin(data.var.index, [x]))[0][0] for x in genes_to_use]
        raws.append(data.raw.X[:, indices])
        aligned.append(to_append)
    
    first = True
    for i in range(len(aligned)):
        data = aligned[i]
        sample_ID = sample_ids[i]
        data.obs['sample_id'] = sample_ID
        if first:
            merged_raw = raws[i]
            merged_X = data.X#.toarray()
            new_var = data.var
            merged_obs = data.obs
            first = False
        else:
            merged_X = np.append(merged_X, data.X, axis=0)
            merged_raw = sparse.vstack([merged_raw, raws[i]])
            merged_obs = merged_obs.append(data.obs)
    to_return = ad.AnnData(merged_X, obs=merged_obs, var=new_var, dtype='float64')
    temp_raw = ad.AnnData(merged_raw.todense(), obs=merged_obs, var=new_var, dtype='float64')
    to_return.raw = temp_raw
    return(to_return)

# LOAD DATA (.h5ad scanpy files), change file locations
sc_PV1 = sc.read(...)
sc_PV2 = sc.read(...)
sc_PV3 = sc.read(...)
sc_ET1 = sc.read(...)
sc_ET2 = sc.read(...)
sc_ET3 = sc.read(...)
sc_ET_V617L = sc.read(...)
sc_healthy1 = sc.read(...)
sc_healthy2 = sc.read(...)
sc_healthy3 = sc.read(...)
# batch corrected output from Seurat
sc_integrated_all = sc.read("sc_data_integrated_final.h5ad")

# MERGE DATA BEFORE BATCH CORRECTION FOR PLOTTING
data_to_merge = [sc_healthy1, sc_healthy2, sc_healthy3, sc_ET1, sc_ET3, sc_ET2, sc_PV1, sc_ET_V617L, sc_PV2, sc_PV3]
labels_merge = ["healthy1", "healthy2", "healthy3", "ET1", "ET3", "ET2", "PV1", "ETV617L", "PV2", "PV3"]
sc_merged_all = merge_scanpy(data_to_merge, labels_merge)
sc.tl.pca(sc_merged_all, svd_solver='arpack', use_highly_variable=False)
sc.pp.neighbors(sc_merged_all)
sc.tl.umap(sc_merged_all)

# CLUSTER BATCH CORRECTED DATA AND TRANSFER CLUSTER ASSIGNMENTS TO RAW DATA
sc.tl.pca(sc_integrated_all, svd_solver='arpack', use_highly_variable=False)
sc.pp.neighbors(sc_integrated_all)
sc.tl.umap(sc_integrated_all)
sc.tl.louvain(sc_integrated_all, resolution=0.8)

# dictionary mapping louvain cluster ids (ints) to cell type names (basophil, HSC, etc)
# YOU MUST CREATE YOUR OWN UNIQUE MAPPING AFTER RUNNING LOUVAIN CLUSTERING BY
# CHECKING EXPRESSION OF CELL TYPE SPECIFIC MARKERS (LYZ, ITGA2B, etc) OF
# EACH CLUSTER AND MAPPING THAT CLUSTER TO A CELL TYPE
cell_type_map = {}

sc_integrated_all.obs["cell_type"] = [cell_type_map[int(x)] for x in sc_integrated_all.obs["louvain"]]
sc_merged_all.obs["cell_type"] = sc_integrated_all.obs["cell_type"].values.tolist()
sc_merged_all.obs["louvain"] = sc_integrated_all.obs["louvain"].values.tolist()

sc_ET1.obs["cell_type"] = sc_integrated_all.obs[sc_integrated_all.obs['sample_id'] == 'ET1']['cell_type'].tolist()
sc_ET_V617L.obs["cell_type"] = sc_integrated_all.obs[sc_integrated_all.obs['sample_id'] == 'ETV617L']['cell_type'].tolist()
sc_ET3.obs["cell_type"] = sc_integrated_all.obs[sc_integrated_all.obs['sample_id'] == 'ET3']['cell_type'].tolist()
sc_PV2.obs["cell_type"] = sc_integrated_all.obs[sc_integrated_all.obs['sample_id'] == 'PV2']['cell_type'].tolist()
sc_ET2.obs["cell_type"] = sc_integrated_all.obs[sc_integrated_all.obs['sample_id'] == 'ET2']['cell_type'].tolist()
sc_healthy1.obs["cell_type"] = sc_integrated_all.obs[sc_integrated_all.obs['sample_id'] == 'healthy1']['cell_type'].tolist()
sc_PV3.obs["cell_type"] = sc_integrated_all.obs[sc_integrated_all.obs['sample_id'] == 'PV3']['cell_type'].tolist()
sc_PV1.obs["cell_type"] = sc_integrated_all.obs[sc_integrated_all.obs['sample_id'] == 'PV1']['cell_type'].tolist()
sc_healthy2.obs["cell_type"] = sc_integrated_all.obs[sc_integrated_all.obs['sample_id'] == 'healthy2']['cell_type'].tolist()
sc_healthy3.obs["cell_type"] = sc_integrated_all.obs[sc_integrated_all.obs['sample_id'] == 'healthy3']['cell_type'].tolist()

# PLOT CELL TYPES ON UMAPs (Fig. 2C, top row)
list_pal = ["navy", "deeppink", "blueviolet", "steelblue", "gold", "tan", "red", "green", "darkorange", "sienna"]

for label in labels_merge:
    to_plot = sc_merged[sc_merged.obs["sample_id"] == label]
    fig = sc.pl.umap(to_plot, color="cell_type", palette=list_pal, title="", return_fig=True, legend_loc=None, size=50)
    plt.axis('off')
    plt.tight_layout()
    #plt.savefig()
    plt.show()
    
embed_name = "X_umap"

# PLOT MUTANT AND WT JAK2 TRANSCRIPTS ON UMAPs (Fig. 2C, bottom two rows)
labels_patient = ["ET1", "ET3", "ET2", "PV1", "ETV617L", "PV2", "PV3"]

for label in labels_patient:
    to_plot = sc_merged[sc_merged.obs["sample_id"] == label]
    if label == "ET1":
        prefix = "JAK2"
    else:
        prefix = "jak2"
    embed_name = "X_umap"
    cmap = {0: 'lightgrey', 1:'lightgrey', 2:'red', 3:'red'}
    cat_1 = prefix+'_cancer_present'
    cat_2 = prefix+'_WT_present'
    color_vals = np.multiply(to_plot.obs[cat_1],2)
    color_vals = color_vals + to_plot.obs[cat_2]
    filters = np.nonzero(np.isin(color_vals, [2,3]))[0]
    color_vals = np.array([cmap[x] for x in color_vals])

    plt.figure()
    plt.scatter(to_plot.obsm[embed_name][:,0], to_plot.obsm[embed_name][:,1], c=color_vals, s=3, zorder=1, alpha=0.5)
    plt.scatter(to_plot.obsm[embed_name][filters,0], to_plot.obsm[embed_name][filters,1], c=color_vals[filters], s=8, zorder=2)
    plt.axis('off')
    #plt.savefig()
    plt.show()
    
for label in labels_patient:
    to_plot = sc_merged[sc_merged.obs["sample_id"] == label]
    if label == "ET1":
        prefix = "JAK2"
    else:
        prefix = "jak2"
    embed_name = "X_umap"
    cmap = {0: 'lightgrey', 1:'darkblue', 2:'lightgrey', 3:'darkblue'}
    cat_1 = prefix+'_cancer_present'
    cat_2 = prefix+'_WT_present'
    color_vals = np.multiply(to_plot.obs[cat_1],2)
    color_vals = color_vals + to_plot.obs[cat_2]
    filters = np.nonzero(np.isin(color_vals, [1,3]))[0]
    color_vals = np.array([cmap[x] for x in color_vals])

    plt.figure()
    plt.scatter(to_plot.obsm[embed_name][:,0], to_plot.obsm[embed_name][:,1], c=color_vals, s=3, zorder=1, alpha=0.5)
    plt.scatter(to_plot.obsm[embed_name][filters,0], to_plot.obsm[embed_name][filters,1], c=color_vals[filters], s=8, zorder=2)
    plt.axis('off')
    #plt.savefig()
    plt.show()
    
# PLOT EXPRESSION OF SELECTED MARKERS FOR ET 1 (Fig. 2B)
to_plot = sc_ET1.copy()
to_plot.obsm["X_umap"] = sc_merged[sc_merged.obs["sample_id"] == "190114"].obsm["X_umap"]
to_plot.X = np.multiply(to_plot.raw.X.todense(), np.reshape(to_plot.obs["n_counts"].values, (len(to_plot.obs["n_counts"]), 1)))
to_plot.X = to_plot.X/100000

maxima = [400, 15, 100, 20, 2000, 15, 15, 15, 800, 50]
markers = ["MPO", "CRHBP", "ITGA2B", "CD79A", "HBB", "CD34", "CD14", "JAK2", "CLC", "IRF8"]

for i in range(len(markers)):
    marker = markers[i]
    max_umi = maxima[i]
    fig = sc.pl.umap(to_plot, color=marker, cmap="viridis", vmax=max_umi, use_raw=False, title="", return_fig=True, size=40)
    plt.axis('off')
    plt.tight_layout()
    #plt.savefig()
    plt.show()