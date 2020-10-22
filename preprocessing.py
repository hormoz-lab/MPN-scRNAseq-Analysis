#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:06:11 2020

@author: dve
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import loompy, h5py, sys, os, random, csv, gseapy
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.colors
from functools import reduce
import anndata as ad

# HELPER FUNCTIONS
def get_mutations_all(sc_data, mut_file, prefix="jak2", file_header="Jak2", bc_order = None, filter_arr=None):
    if not bc_order is None:
        jak2_calls = pd.read_csv(mut_file, header=0)
        jak2_calls["barcodes"] = bc_order
        jak2_calls.set_index("barcodes", inplace=True)
    else:
        jak2_calls = pd.read_csv(mut_file, header=0)
    if not filter_arr is None:
        filter_ind = np.where(filter_arr)
        jak2_calls = jak2_calls.iloc[filter_ind.tolist()]
    if not bc_order is None:
        sc_data.obs = sc_data.obs.join(jak2_calls)
        sc_data.obs.rename(columns={file_header+'WT': prefix+'_WT', file_header+'Cancer':prefix+'_cancer'}, inplace=True)
        #print(sc_data.obs.columns)
    else:
        sc_data.obs[prefix+'_WT'] = jak2_calls[file_header+'WT'].values
        sc_data.obs[prefix+'_cancer'] = jak2_calls[file_header+'Cancer'].values
    sc_data.obs[prefix+'_cancer_ratio'] = sc_data.obs[prefix+'_cancer']/(sc_data.obs[prefix+'_WT'] + sc_data.obs[prefix+'_cancer'])
    sc_data.obs[prefix+'_cancer_present'] = sc_data.obs[prefix+'_cancer'] > 0
    sc_data.obs[prefix+'_WT_present'] = sc_data.obs[prefix+'_WT'] > 0
    sc_data.obs[prefix+'_any_present'] = sc_data.obs[prefix+'_cancer'] + sc_data.obs[prefix+'_WT'] > 0
    sc_data.obs[prefix+"_total"] = sc_data.obs[prefix+"_WT"] + sc_data.obs[prefix+"_cancer"]
    return(sc_data)

def add_mito_cc(sc_data):
    cell_cycle_genes = [x.strip() for x in open('/home/dsv4/prolif_fields/regev_lab_cell_cycle_genes.txt')]
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    sc.tl.score_genes_cell_cycle(sc_data, s_genes=s_genes, g2m_genes=g2m_genes)

    mito_genes=[]
    
    # EDIT AND ADD LOCATION OF CELL CYCLE GENE LIST
    cell_cycle_file_loc = ""
    with open(cell_cycle_file_loc) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            mito_genes.append(row[2])

    mito_genes = list(filter(lambda x: x in sc_data.var_names, mito_genes))
    sc_data.obs['percent_mito'] = np.sum(
        sc_data[:, mito_genes].X, axis=1) / np.sum(sc_data.X, axis=1)
    # add the total counts per cell as observations-annotation to adata
    sc_data.obs['n_counts'] = sc_data.X.sum(axis=1)
    return(sc_data)

def norm_and_dim_red(sc_data):
    sc.pp.filter_cells(sc_data, min_counts=2000)
    sc_data = sc_data[sc_data.obs['percent_mito'] < 0.2, :]
    sc.pp.filter_genes(sc_data, min_cells=3)
    sc.pp.normalize_per_cell(sc_data, counts_per_cell_after=1e5)
    sc_data.raw = sc_data
    sc.pp.log1p(sc_data)
    sc.pp.highly_variable_genes(sc_data, min_mean=0.0125, max_mean=30, min_disp=0.1)
    sc.pp.regress_out(sc_data, ['n_counts', 'percent_mito'])
    sc.tl.pca(sc_data, svd_solver='arpack', use_highly_variable=False)
    sc.pp.neighbors(sc_data)
    sc.tl.umap(sc_data)
    #sc.tl.draw_graph(sc_data)
    return(sc_data)

# LOAD 10X DATA (insert appropriate file locations to filtered feature matrices (CellRanger output, h5 files))
sc_ET1 = sc.read_10x_h5(...)
sc_ET2 = sc.read_10x_h5(...)
sc_ET3 = sc.read_10x_h5(...)
sc_ET_V617L = sc.read_10x_h5(...)
sc_PV1 = sc.read_10x_h5(...)
sc_PV2 = sc.read_10x_h5(...)
sc_PV3 = sc.read_10x_h5(...)
sc_healthy = sc.read_10x_h5(...)
sc_healthy2 = sc.read_10x_h5(...)
sc_healthy3 = sc.read_10x_h5(...)

# ADD AMPLICON MUTATION CALLS TO ANNDATA OBS
# adjust file locations as appropriate, if all files are in same directory you can add a variable mut_calls_dir with the directory location
sc_PV3 = get_mutations_all(sc_PV3, mut_calls_dir+"Python_AnnData_NS_500_All_Cells_0701_Jak2.csv", prefix="jak2", file_header="Jak2")
sc_PV3 = get_mutations_all(sc_PV3, mut_calls_dir+"Python_AnnData_NS_10000_All_Cells_0701_Tet2C.csv", prefix="tet2", file_header="Tet2C")
sc_PV1 = get_mutations_all(sc_PV1, mut_calls_dir+"0722/Python_AnnData_NS_1000_All_Cells_0722_Jak2.csv")
sc_ET3 = get_mutations_all(sc_ET3, mut_calls_dir+"0222/Python_AnnData_NS_100_All_Cells_0222_v2.csv", prefix="jak2", file_header="Jak2")
sc_ET_V617L = get_mutations_all(sc_ET_V617L, mut_calls_dir+"0117/Python_AnnData_NS_0117_100_All_Cells.csv", prefix="jak2", file_header="Jak2")
sc_PV2 = get_mutations_all(sc_PV2, mut_calls_dir+"0329/Python_AnnData_NS_100_All_Cells_0329.csv", prefix="jak2", file_header="Jak2")
sc_PV2 = get_mutations_all(sc_PV2, mut_calls_dir+"0329/Python_AnnData_NS_100_All_Cells_0329_Tet2B.csv", prefix="tet2", file_header="Tet2")

mut_names1 = ["JAK2", "ASH1L", "MEF2C", "ABHD2", "FHIT", "MAML3", "TRAPPC11", "HSPA9", "PPIL3", "FRYL", "RSF1", "UPF1_O2"]

# list of amplicon mutation file locations for ET 1, in the same order as mut_names
# if all files are in the same directory, you can just add the directory location to mut_dir1.
mut_files1 = ["Python_AnnData_NS_Run2_100_All_Cells.csv", "Python_AnnData_NS_3000_All_Cells_0114_ASH1L.csv", "Python_AnnData_NS_5000_All_Cells_0114_MEF2C.csv", "Python_AnnData_NS_10000_All_Cells_0114_ABHD2.csv", "Python_AnnData_NS_10000_All_Cells_0114_FHIT.csv", "Python_AnnData_NS_10000_All_Cells_0114_MAML3.csv", "Python_AnnData_NS_1000_All_Cells_0114_TRAPPC11.csv", "Python_AnnData_NS_1000_All_Cells_0114_HSPA9.csv", "Python_AnnData_NS_5000_All_Cells_0114_PPIL3.csv", "Python_AnnData_NS_10000_All_Cells_0114_FRYL.csv", "Python_AnnData_NS_10000_All_Cells_0114_RSF1.csv", "Python_AnnData_NS_200_All_Cells_0114_UPF1_O2.csv"]
#mut_dir1 = INSERT DIRECTORY LOC HERE
for i in range(len(mut_files1)):
    sc_ET1 = get_mutations_all(sc_ET1, mut_files1[i], prefix=mut_names1[i], file_header=mut_names1[i])

mut_names2 = ["NSFP1", "CDC42", "NRROS", "ZNF22"]

# list of amplicon mutation file locations for ET 2, in the same order as mut_names. 
# if all files are in the same directory, you can just add the directory location to mut_dir2.
mut_files2 = ["Python_AnnData_NS_1000_All_Cells_0311_NSFP1.csv", "Python_AnnData_NS_5000_All_Cells_0311_CDC42.csv", "Python_AnnData_NS_1000_All_Cells_0311_NRROS.csv", "Python_AnnData_NS_4000_All_Cells_0311_ZNF22.csv"]
#mut_dir2 = INSERT DIRECTORY LOC HERE
for i in range(len(mut_files2)):
    sc_ET2 = get_mutations_all(sc_ET2, mut_dir2+mut_files2[i], prefix=mut_names2[i], file_header=mut_names2[i])

# FILTER, NORMALIZE, AND SAVE DATA
all_sc_data = [sc_ET1, sc_ET2, sc_ET3, sc_PV1, sc_PV2, sc_PV3, sc_ET_V617L, sc_healthy, sc_healthy2, sc_healthy3]
labels = ["ET1", "ET2", "ET3", "PV1", "PV2", "PV3", "ETV617L", "healthy", "healthy2", "healthy3"]
for i in range(len(all_sc_data)):
    sc_data = all_sc_data[i]
    sc_data.var_names_make_unique()
    sc_data = add_mito_cc(sc_data)
    sc_data = norm_and_dim_red(sc_data)
    all_sc_data[i] = sc_data
    sc_data.write("sc_data_"+labels[i]+"_final.h5ad")
    
