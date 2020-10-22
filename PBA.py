#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:05:32 2020

@author: dve
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import loompy, h5py, sys, os, random, csv, csv, io
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from functools import reduce
from statsmodels.stats.multitest import multipletests
import anndata as ad
from pathlib import Path

# HELPER FUNCTIONS (many from Weinreb et al)
def row_sum_normalize(A):
    d = np.sum(A,axis=1)
    A = A / (np.tile(d[:,None],(1,A.shape[1])))
    return(A)


def compute_Linv(adj_mat):
    L = np.identity(adj_mat.shape[0]) - row_sum_normalize(adj_mat)
    return(np.linalg.pinv(L))

def compute_V(Linv, R):
    V = np.dot(Linv, R)
    return(V)

def compute_R(sc_data, HBB_percentile, MEP_scaling, to_PBA, MEP_R=None):
    HBB_index = np.nonzero(sc_data.var.index == "HBB")[0][0]
    subset_data = sorted(sc_data.raw.X.todense()[:,HBB_index].tolist())
    HBB_threshold = subset_data[int(HBB_percentile * len(subset_data))]
    HBB_exp = np.array((to_PBA.raw.X.todense()[:,HBB_index] > HBB_threshold).tolist())
    terminally_diff = np.array([int(x) for x in HBB_exp])
    
    num_MEP = len(to_PBA[to_PBA.obs["cell_type"]== "MEP"]) + len(to_PBA[to_PBA.obs["cell_type"]== "erythroid"])
    num_HSC = len(to_PBA[to_PBA.obs["cell_type"]== "HSC"])
    num_Er = np.sum(HBB_exp)
    R = -1 * (terminally_diff)
    to_PBA.obs["is_sink_cell"] = terminally_diff
    
    if MEP_R is None:
        MEP_R = MEP_scaling * np.sum(num_Er)/np.sum(num_MEP)
    
    R = R + np.isin(to_PBA.obs["cell_type"], ["MEP", "erythroid"]) * MEP_R
    HSC_R = -np.sum(R)/num_HSC
    R = R + (to_PBA.obs["cell_type"]== "HSC") * HSC_R
    return(R, MEP_R)

def simulate_markov_T(start, trans_mat, T):
    total_trans = np.linalg.matrix_power(trans_mat, T)
    return(np.matmul(start, total_trans))

def simulate_markov_PBA(to_PBA, V, genes_to_measure, scale_factor, dummy_exp, exit_rate=1, D=1):
    stem_cell = np.nonzero(to_PBA.obs["is_stem_cell"])[0][0]
    all_sinks = np.nonzero(to_PBA.obs["is_sink_cell"])[0]
    A = to_PBA.uns['neighbors']['connectivities'].todense()
    A = A * scale_factor
    for i in range(np.shape(A)[0]):
        A[i,i] = 1
    V = V / D
    Vx,Vy = np.meshgrid(V,V)
    P = np.multiply(A,np.exp(Vy - Vx))
    P[all_sinks,:] = 0
    P = np.hstack((P, np.reshape(to_PBA.obs["is_sink_cell"].values, (np.shape(P)[0],1))))
    P = np.vstack((P, np.zeros(np.shape(P)[1])))
    P[-1,-1] = 1
    start_state = np.zeros(len(V)+1)
    start_state[stem_cell] = 1
    normed_mat = row_sum_normalize(P)
    curr_state = start_state
    for i in range(num_timesteps):
        curr_state = simulate_markov_T(curr_state, normed_mat, 1)
        if i == 0:
            gene_data = get_avg_exp(to_PBA, genes_to_measure, weights=curr_state, dummies=[dummy_exp])
        else:
            new_row = get_avg_exp(to_PBA, genes_to_measure, weights=curr_state, dummies=[dummy_exp])
            gene_data = pd.concat([gene_data, new_row], axis=0, ignore_index=True)
    return(gene_data)

def get_potentials(sc_data, prefix, stem_cell_idx, is_random=False, is_het=True, MEP_R=None):
    to_PBA = sc_data[np.isin(sc_data.obs["cell_type"], ["erythroid", "MEP", "HSC"])]
    to_PBA = to_PBA[np.logical_or(to_PBA.obs[prefix+"_WT_present"], to_PBA.obs[prefix+"_cancer_present"])]
    to_PBA.obs["is_stem_cell"] = range(len(to_PBA))==stem_cell_idx
    if is_het:
        to_PBA = mut_probs(to_PBA, [prefix])
        to_PBA.obs["sampled_mut"] = [np.random.choice([True, False], p=[x, 1-x]) for x in to_PBA.obs[prefix+"_mut_prob"]]
    else:
        to_PBA.obs["sampled_mut"] = to_PBA.obs[prefix+"_cancer_present"]
    if is_random:
        to_PBA.obs["sampled_mut"] = np.random.permutation(to_PBA.obs["sampled_mut"])
    to_PBA_mut = to_PBA[np.logical_or(to_PBA.obs["sampled_mut"], to_PBA.obs["is_stem_cell"])]
    to_PBA_WT = to_PBA[np.logical_or(~to_PBA.obs["sampled_mut"], to_PBA.obs["is_stem_cell"])]
    
    sc.pp.neighbors(to_PBA_mut, n_neighbors=5)
    R, MEP_R = compute_R(sc_data, HBB_percentile, MEP_scaling, to_PBA_mut, MEP_R)
    Linv = compute_Linv(to_PBA_mut.uns['neighbors']['connectivities'])
    mut_potential = compute_V(Linv, R)
    mut_R = R

    sc.pp.neighbors(to_PBA_WT, n_neighbors=5)
    R, MEP_R = compute_R(sc_data, HBB_percentile, MEP_scaling, to_PBA_WT, MEP_R)
    Linv = compute_Linv(to_PBA_WT.uns['neighbors']['connectivities'])
    WT_potential = compute_V(Linv, R)
    WT_R = R
    return(to_PBA_mut, to_PBA_WT, mut_R, WT_R, mut_potential, WT_potential)

# LOAD DATA (.h5ad scanpy files), change file locations
sc_PV1 = sc.read(...)
sc_PV2 = sc.read(...)
sc_PV3 = sc.read(...)
sc_ET1 = sc.read(...)
sc_ET2 = sc.read(...)
sc_ET3 = sc.read(...)
sc_ET_V617L = sc.read(...)

# PERFORM PBA ON ALL PATIENTS
MPN_data = [sc_ET1, sc_ET_V617L, sc_PV2, sc_ET2, sc_ET3, sc_PV3, sc_PV1]
MPN_labels = ["ET1", "ETV617L", "PV2", "ET2", "ET3", "PV3", "PV1"]
mut_prefixes = ["all", "jak2", "jak2", "all", "jak2", "jak2", "jak2"]
save_mut = True
mut_sc_datas = []
WT_sc_datas = []
mut_potentials = []
WT_potentials = []
mut_R = []
WT_R = []
stem_cells = []
dummy_exps = []
MEP_Rs = []
mut_trajs = []
WT_trajs = []
genes_to_measure = ["GATA1", "HBB", "KLF1"]

HBB_percentile = 0.9
MEP_scaling = 0.9

D = 1
scale_factor = 0.5
num_timesteps = 150

for i in range(len(MPN_data)):
    sc_data = MPN_data[i]
    prefix = mut_prefixes[i]
    to_PBA = sc_data[np.isin(sc_data.obs["cell_type"], ["erythroid", "MEP", "HSC"])]
    if prefix != "none":
        to_PBA = to_PBA[np.logical_or(to_PBA.obs[prefix+"_WT_present"], to_PBA.obs[prefix+"_cancer_present"])]
    sc.pp.neighbors(to_PBA, n_neighbors=10)
    
    R, MEP_R = compute_R(sc_data, HBB_percentile, MEP_scaling, to_PBA)
    MEP_Rs.append(MEP_R)
    Linv = compute_Linv(to_PBA.uns['neighbors']['connectivities'])
    V = compute_V(Linv, R)
    stem_cells.append(np.argmax(V))
    dummy_exp = to_PBA[to_PBA.obs["is_sink_cell"] > 0]
    dummy_exp = np.mean(dummy_exp.raw.X, axis=0)
    dummy_exps.append(dummy_exp)
    
    # heterozygote correction
    if MPN_labels[i] in ["ET1", "ETV617L", "ET2", "ET3", "PV3"]:
        to_PBA_mut, to_PBA_WT, mut_R_new, WT_R_new, mut_potential, WT_potential = get_potentials(sc_data, prefix, stem_cells[i], is_random=False, is_het=True, MEP_R=MEP_R)
    else: 
        to_PBA_mut, to_PBA_WT, mut_R_new, WT_R_new, mut_potential, WT_potential = get_potentials(sc_data, prefix, stem_cells[i], is_random=False, is_het=False, MEP_R=MEP_R)
    
    mut_potential = np.asarray(mut_potential)[0]
    WT_potential = np.asarray(WT_potential)[0]

    mut_sc_datas.append(to_PBA_mut)
    WT_sc_datas.append(to_PBA_WT)
    mut_potentials.append(mut_potential)
    WT_potentials.append(WT_potential)


    mut_R.append(mut_R_new)
    WT_R.append(WT_R_new)

    
    gene_data = simulate_markov_PBA(to_PBA_mut, mut_potential, genes_to_measure, scale_factor, dummy_exp, exit_rate=1, D=D)
    mut_trajs.append(gene_data)
    gene_data = simulate_markov_PBA(to_PBA_WT, WT_potential, genes_to_measure, scale_factor, dummy_exp, exit_rate=1, D=D)
    WT_trajs.append(gene_data)

# PERMUTE JAK2 LABELS TO GET NULL DISTRIBUTION OF TRAJECTORIES
# PLOT PERMUTED RANDOM LABEL DATA AND TRUE DATA
# Example given for ET 1, changing sc_data and idx (index of patient in MPN_data list) will allow you to use other patients
num_runs = 25
sc_data = sc_ET1
prefix = 'all'
idx = 0
dummy_exp = dummy_exps[idx]

D = 1
scale_factor = 0.9
num_timesteps = 150
genes_to_measure = ["GATA1", "HBB"]

WT_data = []
mut_data = []


for i in range(num_runs):
    print(i)
    to_PBA_mut, to_PBA_WT, mut_R_new, WT_R_new, mut_potential, WT_potential = get_potentials(sc_data, prefix, stem_cells[idx], is_random=True, is_het=True, MEP_R=MEP_Rs[idx])
    mut_potential = np.asarray(mut_potential)[0]
    WT_potential = np.asarray(WT_potential)[0]
    gene_data = simulate_markov_PBA(to_PBA_mut, mut_potential, genes_to_measure, scale_factor, dummy_exp, exit_rate=1, D=D)
    mut_data.append(gene_data["GATA1"].tolist())
    gene_data = simulate_markov_PBA(to_PBA_WT, WT_potential, genes_to_measure, scale_factor, dummy_exp, exit_rate=1, D=D)
    WT_data.append(gene_data["GATA1"].tolist())
    
WT_avg = np.mean(np.array(WT_data), axis=0)
mut_avg = np.mean(np.array(mut_data), axis=0)

WT_sd = np.std(np.array(WT_data), axis=0)
mut_sd = np.std(np.array(mut_data), axis=0)

plt.figure()
plt.fill_between(range(len(mut_avg)), mut_avg+mut_sd, mut_avg-mut_sd, color="red", alpha=0.1)
plt.plot(mut_avg, color="red", alpha=0.1)

plt.fill_between(range(len(WT_avg)), WT_avg+WT_sd, WT_avg-WT_sd, color="blue", alpha=0.1)
plt.plot(WT_avg, color="blue", alpha=0.1)

plt.plot(WT_trajs[idx]["GATA1"], color="navy")
plt.plot(mut_trajs[idx]["GATA1"], color="red")

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.ylabel("mean GATA1 expression (normalized counts)")
plt.tight_layout()
#plt.savefig()  
plt.show()