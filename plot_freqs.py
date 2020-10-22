#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 17:19:26 2020

@author: dve
"""
from scipy import stats, sparse
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import loompy, h5py, sys, os, random, csv
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import anndata as ad

# HELPER FUNCTIONS
def get_categorical_freq_counts(sc_data, groupby, prefix, is_counts=True):
    if is_counts:
        select_data = sc_data.obs[[prefix+"_WT", prefix+"_cancer", groupby]]
        grouped_means = select_data.groupby(by=groupby).mean()
        grouped_means[prefix+"_ratio"] = np.divide(grouped_means[prefix+"_cancer"], grouped_means[prefix+"_cancer"] + grouped_means[prefix+"_WT"])
        totals = select_data.groupby(by=groupby).sum()
        grouped_means['total_cancer'] = totals[prefix+"_cancer"]
        grouped_means['total_WT'] = totals[prefix+"_WT"]
    else:
        select_data = sc_data.obs[[prefix+"_WT_present", prefix+"_cancer_present", groupby]]
        grouped_means = select_data.groupby(by=groupby).mean()
        grouped_means[prefix+"_ratio"] = np.divide(grouped_means[prefix+"_cancer_present"], grouped_means[prefix+"_cancer_present"] + grouped_means[prefix+"_WT_present"])
        totals = select_data.groupby(by=groupby).sum()
        grouped_means['total_cancer'] = totals[prefix+"_cancer_present"]
        grouped_means['total_WT'] = totals[prefix+"_WT_present"]
    grouped_means["total_counts"] = grouped_means['total_cancer'] + grouped_means['total_WT']
    CIs = np.divide(np.multiply(grouped_means[prefix+"_ratio"], 1-grouped_means[prefix+"_ratio"]), grouped_means['total_counts'])
    CIs = np.sqrt(CIs)
    grouped_means['upper'] = grouped_means[prefix+"_ratio"] + 1.96*CIs
    grouped_means['lower'] = grouped_means[prefix+"_ratio"] - 1.96*CIs
    return(grouped_means)

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

sc_ET1.obs["jak2_cancer_present"] = sc_ET1.obs["JAK2_cancer_present"]
sc_ET1.obs["jak2_WT_present"] = sc_ET1.obs["JAK2_WT_present"]

sc_ET1.obs["jak2_cancer"] = sc_ET1.obs["JAK2_cancer"]
sc_ET1.obs["jak2_WT"] = sc_ET1.obs["JAK2_WT"]

# PLOT FREQUENCY OF JAK2 MUTANT TRANSCRIPTS IN EACH CELL TYPE (Fig. 2D)
all_data_ET = [sc_ET1, sc_ET2, sc_ET3]
all_data_PV = [sc_PV1, sc_PV2, sc_PV3, sc_ET_V617L]

to_plot_ET = []
uppers_ET = []
lowers_ET = []
prefix="jak2"

for i in range(len(all_data_ET)):
    sc_data = all_data_ET[i]
    select_data = sc_data[np.logical_or(sc_data.obs[prefix+'_cancer_present'],sc_data.obs[prefix+'_WT_present'])]
    cell_type_freqs = get_categorical_freq_counts(select_data, "cell_type", prefix)
    cell_type_freqs = cell_type_freqs.loc[["erythroid", "megakaryocyte", "MEP", "HSC", "GMP", "CD14+", "lymphoid"]]
    to_plot_ET.extend(cell_type_freqs[prefix+'_ratio']*2)
    uppers_ET.extend(cell_type_freqs["upper"]*2)
    lowers_ET.extend(cell_type_freqs["lower"]*2)
    
    
to_plot_ET = np.array(to_plot_ET)

to_plot_PV = []
uppers_PV = []
lowers_PV = []

for i in range(len(all_data_PV)):
    sc_data = all_data_PV[i]
    select_data = sc_data[np.logical_or(sc_data.obs[prefix+'_cancer_present'],sc_data.obs[prefix+'_WT_present'])]
    cell_type_freqs = get_categorical_freq_counts(select_data, "cell_type", prefix)
    cell_type_freqs = cell_type_freqs.loc[["erythroid", "megakaryocyte", "MEP", "HSC", "GMP", "CD14+", "lymphoid"]]
    if i == 2 or i == 3:
        to_plot_PV.extend(cell_type_freqs[prefix+'_ratio']*2)
        uppers_PV.extend(cell_type_freqs["upper"]*2)
        lowers_PV.extend(cell_type_freqs["lower"]*2)
    else:
        to_plot_PV.extend(cell_type_freqs[prefix+'_ratio'])
        uppers_PV.extend(cell_type_freqs["upper"])
        lowers_PV.extend(cell_type_freqs["lower"])

to_plot_PV = np.array(to_plot_PV)

all_data_ET = np.reshape(to_plot_ET, newshape=(3,7))
all_data_PV = np.reshape(to_plot_PV, newshape=(4,7))
all_data = np.concatenate((all_data_ET, all_data_PV), axis=0)

ax = sns.heatmap(all_data, vmin=0, vmax=1, cmap="viridis",linewidths=0.2,linecolor='black')
plt.xticks(np.arange(8)+0.5, ["Erythroid\nprogenitor", "Megakaryocyte\nprogenitor", "MEP", "HSC", "GMP", "CD14+", "Lymphoid\nprogenitor"], rotation=90, fontsize=12)
plt.yticks(np.arange(7) + 0.4, ["ET 1", "ET 2", "ET 3", "PV 1", "PV 2 (TET2)", "PV 3 (TET2)", "ET V617L"], rotation=0, fontsize=14)
ax.tick_params(axis="both", which="both",length=0)
plt.tight_layout()
#plt.savefig()
plt.show()

# PLOT PERIPHERAL BLOOD JAK2 FREQUENCY (Fig. 2E)
all_pb_ET = np.array([.166, .253, .106])
all_pb_PV = np.array([.627, .68, .409])
all_pb_num_ET = np.array([295, 233, 311])
all_pb_num_PV = np.array([640, 244, 251+362])

all_data_PV = [sc_PV1, sc_PV2, sc_PV3]

to_plot_ET = []
uppers_ET = []
lowers_ET = []
prefix="jak2"

for i in range(len(all_data_ET)):
    sc_data = all_data_ET[i]
    select_data = sc_data[np.logical_or(sc_data.obs[prefix+'_cancer_present'],sc_data.obs[prefix+'_WT_present'])]
    cell_type_freqs = get_categorical_freq_counts(select_data, "cell_type", prefix)
    cell_type_freqs = cell_type_freqs.loc[["erythroid", "megakaryocyte", "MEP", "HSC", "GMP", "CD14+", "lymphoid"]]
    to_plot_ET.extend(cell_type_freqs[prefix+'_ratio'])
    uppers_ET.extend(cell_type_freqs["upper"])
    lowers_ET.extend(cell_type_freqs["lower"])

    
    to_plot_ET.append(all_pb_ET[i])
    CI = 1.96*np.sqrt(all_pb_ET[i]*(1-all_pb_ET[i])/all_pb_num_ET[i])
    uppers_ET.append(all_pb_ET[i]+CI)
    lowers_ET.append(all_pb_ET[i]-CI)
    
    
to_plot_ET = np.array(to_plot_ET)

to_plot_PV = []
uppers_PV = []
lowers_PV = []
prefix="jak2"

for i in range(len(all_data_PV)):
    sc_data = all_data_PV[i]
    select_data = sc_data[np.logical_or(sc_data.obs[prefix+'_cancer_present'],sc_data.obs[prefix+'_WT_present'])]
    cell_type_freqs = get_categorical_freq_counts(select_data, "cell_type", prefix)
    cell_type_freqs = cell_type_freqs.loc[["erythroid", "megakaryocyte", "MEP", "HSC", "GMP", "CD14+", "lymphoid"]]

    to_plot_PV.extend(cell_type_freqs[prefix+'_ratio'])
    uppers_PV.extend(cell_type_freqs["upper"])
    lowers_PV.extend(cell_type_freqs["lower"])

    
    to_plot_PV.append(all_pb_PV[i])
    CI = 1.96*np.sqrt(all_pb_PV[i]*(1-all_pb_PV[i])/all_pb_num_PV[i])
    uppers_PV.append(all_pb_PV[i]+CI)
    lowers_PV.append(all_pb_PV[i]-CI)
    
to_plot_PV = np.array(to_plot_PV)

all_data_ET = np.reshape(to_plot_ET, newshape=(3,8))
all_data_PV = np.reshape(to_plot_PV, newshape=(3,8))
all_data = pd.DataFrame(all_data, index=["ET 1", "ET 2", "ET 3", "PV 1", "PV 2", "PV 3"], columns=["erythroid", "megakaryocyte", "MEP", "HSC", "GMP", "CD14+", "lymphoid", "PB"])

all_CIs_ET = np.reshape(to_plot_ET-np.array(lowers_ET), newshape=(3,8))
all_CIs_PV = np.reshape(to_plot_PV-np.array(lowers_PV), newshape=(3,8))
all_CIs = np.concatenate((all_CIs_ET, all_CIs_PV), axis=0)
all_CIs = pd.DataFrame(all_CIs, index=["ET 1", "ET 2", "ET 3", "PV 1", "PV 2", "PV 3"], columns=["erythroid", "megakaryocyte", "MEP", "HSC", "GMP", "CD14+", "lymphoid", "PB"])

x_label = "erythroid"
x_label2 = "HSC"

sizes = 150
border_width = 1

plt.figure()
plt.errorbar(x=all_data["PB"], y=all_data[x_label], xerr=all_CIs["PB"], yerr=all_CIs[x_label], color="black", fmt="none", zorder=1)
plt.scatter(all_data.loc["ET 1"]["PB"], all_data.loc["ET 1"][x_label], color="red", zorder=2, s=sizes, marker="d", label="ET 1", edgecolors='black', linewidth=border_width)
plt.scatter(all_data.loc["ET 2"]["PB"], all_data.loc["ET 2"][x_label], color="red", zorder=2, s=sizes, marker="s", label="ET 2", edgecolors='black', linewidth=border_width)
plt.scatter(all_data.loc["ET 3"]["PB"], all_data.loc["ET 3"][x_label], color="red", zorder=2, s=sizes, label="ET 3", edgecolors='black', linewidth=border_width)
plt.scatter(all_data.loc["PV 1"]["PB"], all_data.loc["PV 1"][x_label], color="red", zorder=2, s=sizes, marker="v", label="PV 1", edgecolors='black', linewidth=border_width)
plt.scatter(all_data.loc["PV 2"]["PB"], all_data.loc["PV 2"][x_label], color="red", zorder=2, s=sizes, marker="^", label="PV 2", edgecolors='black', linewidth=border_width)
plt.scatter(all_data.loc["PV 3"]["PB"], all_data.loc["PV 3"][x_label], color="red", zorder=2, s=sizes, marker=">", label="PV 3", edgecolors='black', linewidth=border_width)

plt.errorbar(x=all_data["PB"], y=all_data[x_label2], xerr=all_CIs["PB"], yerr=all_CIs[x_label2], color="black", fmt="none", zorder=1)
plt.scatter(all_data.loc["ET 1"]["PB"], all_data.loc["ET 1"][x_label2], color="steelblue", zorder=2, s=sizes, marker="d", edgecolors='black', linewidth=border_width)
plt.scatter(all_data.loc["ET 2"]["PB"], all_data.loc["ET 2"][x_label2], color="steelblue", zorder=2, s=sizes, marker="s", edgecolors='black', linewidth=border_width)
plt.scatter(all_data.loc["ET 3"]["PB"], all_data.loc["ET 3"][x_label2], color="steelblue", zorder=2, s=sizes, edgecolors='black', linewidth=border_width)
plt.scatter(all_data.loc["PV 1"]["PB"], all_data.loc["PV 1"][x_label2], color="steelblue", zorder=2, s=sizes, marker="v", edgecolors='black', linewidth=border_width)
plt.scatter(all_data.loc["PV 2"]["PB"], all_data.loc["PV 2"][x_label2], color="steelblue", zorder=2, s=sizes, marker="^", edgecolors='black', linewidth=border_width)
plt.scatter(all_data.loc["PV 3"]["PB"], all_data.loc["PV 3"][x_label2], color="steelblue", zorder=2, s=sizes, marker=">", edgecolors='black', linewidth=border_width)


plt.plot([0,20], [0,20], color="black", linestyle=":", zorder=0)

#plt.xlabel("peripheral blood VAF")
#plt.ylabel("BM JAK2 mutant transcript fraction")
plt.xlim(0,1)
plt.ylim(0,1)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.legend()
plt.tight_layout()
#plt.savefig()
plt.show()
