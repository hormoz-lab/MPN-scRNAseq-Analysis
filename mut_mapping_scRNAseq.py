#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:13:40 2020

@author: dve
"""

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import loompy, h5py, sys, os, random, csv, gseapy
import pandas as pd
import seaborn as sns
import matplotlib.colors
from functools import reduce
from matplotlib.colors import LinearSegmentedColormap

# HELPER FUNCTIONS
def add_somatic_muts(mutMatrix, anno_rev, sc_data, to_drop=[], prefix="somatic", select_mut=None, select_HSC=None):
    mutMatrix = mutMatrix.copy().drop(to_drop, axis=1)
    
    all_HSCs = mutMatrix.columns[7:-3]
    mutant_HSCs = np.nonzero(mutMatrix[mutMatrix["gene"] == 'JAK2'][list(all_HSCs)].values)[1]
    mutant_HSCs = all_HSCs[mutant_HSCs]

    WT_HSCs = list(set(all_HSCs) - set(mutant_HSCs))

    mutMatrix["WT_count"] = np.sum(mutMatrix[WT_HSCs] > 0, axis=1)
    mutMatrix["mut_count"] = np.sum(mutMatrix[mutant_HSCs] > 0, axis=1)
    mutMatrix["counts"] = mutMatrix["WT_count"] + mutMatrix["mut_count"]
    mutMatrix = mutMatrix[np.isin(mutMatrix["V2"], anno_rev.columns)]
    
    not_intergenic = mutMatrix[~(mutMatrix["region"]=="intergenic")]
    if not select_mut is None:
        not_intergenic = mutMatrix[np.isin(mutMatrix["gene"], select_mut)]
        mut_pos = not_intergenic["V2"]
    elif not select_HSC is None:
        not_intergenic = not_intergenic[not_intergenic["counts"]==1]
        not_intergenic = not_intergenic[not_intergenic[select_HSC] > 0]
        mut_pos = not_intergenic["V2"]
    else:
        mut_clade = not_intergenic[not_intergenic["mut_count"] >= (len(mutant_HSCs)-1)]
        mut_clade = mut_clade[mut_clade["WT_count"] == 0]
        WT_all = not_intergenic[not_intergenic["WT_count"] > 0]
        mut_pos = mut_clade["V2"]
    
    try:
        sc_data.obs = sc_data.obs.drop(mut_pos, axis=1)
    except KeyError:
        pass
    sc_data.obs = sc_data.obs.join(anno_rev[mut_pos])
    sc_data.obs[mut_pos] = np.nan_to_num(sc_data.obs[mut_pos], copy=True)
    sc_data.obs[prefix+"_cancer_present"] = np.logical_or.reduce(np.isin(sc_data.obs[mut_pos], [2,3]), axis=1)
    sc_data.obs[prefix+"_WT_present"] = np.logical_and(np.logical_or.reduce(np.isin(sc_data.obs[mut_pos], [1]), axis=1), ~sc_data.obs[prefix+"_cancer_present"]) 
    sc_data.obs = sc_data.obs.drop(mut_pos, axis=1)
    return(sc_data)

def filter_by_reads(q_mat, all_bcs, read_threshold=1):
    all_muts = list(set(q_mat["end"]))
    mut_to_idx = dict(zip(all_muts, range(len(all_muts))))
    bc_to_idx = dict(zip(all_bcs, range(len(all_bcs))))
    to_return = np.zeros(shape=(len(all_bcs), len(all_muts)))
    count = 0
    for i in range(len(q_mat)):
        row = q_mat.iloc[i]
        if row["cov"] > read_threshold:
            try:
                bc_idx = bc_to_idx[row["barcode"]]
            except KeyError:
                continue
            if row["mutation_type"] == "WT":
                if to_return[bc_idx, mut_to_idx[row["end"]]] not in [1, 3]:
                    to_return[bc_idx, mut_to_idx[row["end"]]] += 1
            elif row["mutation_type"] == "mutant":
                if to_return[bc_idx, mut_to_idx[row["end"]]] not in [2, 3]:
                    to_return[bc_idx, mut_to_idx[row["end"]]] += 2
    to_return = pd.DataFrame(to_return, index=all_bcs, columns=all_muts)
    return(to_return)

def get_clades(bool_muts, filtered_matrix):
    nonzeros = np.nonzero(bool_muts.values)
    locs_df = pd.DataFrame({"mutation": nonzeros[0], "HSC": nonzeros[1]})
    clades = []
    mutations = []
    num_HSC = []
    for i in range(len(bool_muts)):
        only_mut = locs_df[locs_df["mutation"]==i]
        clade = ".".join([str(x) for x in only_mut["HSC"]])
        clades.append(clade)
        mut_pos = filtered_matrix.loc[bool_muts.iloc[i].name]["gene"]
        mutations.append(mut_pos)
        num_HSC.append(len(only_mut))
    clade_set = sorted(list(set(clades)))
    return(pd.DataFrame({"gene":mutations, "clade_ID":[clade_set.index(x) for x in clades], "clade_def":clades, "num_HSC":num_HSC}))

# LOAD DATA (.h5ad scanpy files and mutation files)
sc_ET1 = sc.read(...)
sc_ET2 = sc.read(...)

# WGS mutation matrix (p0114_ALL_mutations.csv and trees_ALL_mutations_no_filt.csv)
mutMatrix1 = pd.read_csv(..., sep='\t')
mutMatrix1.drop(['HSC.38', 'STM.00'], axis=1, inplace=True)

mutMatrix2 = pd.read_csv(..., sep='\t')
mutMatrix2.drop(['STM.00_1'], axis=1, inplace=True)
    
all_HSCs1 = mutMatrix1.columns[7:-1]
mutant_HSCs1 = np.nonzero(mutMatrix1[mutMatrix1["gene"] == 'JAK2'][list(all_HSCs1)].values)[1]
mutant_HSCs1 = all_HSCs1[mutant_HSCs1]
WT_HSCs1 = list(set(all_HSCs1) - set(mutant_HSCs1))

mutMatrix1["WT_count"] = np.sum(mutMatrix1[WT_HSCs1] > 0, axis=1)
mutMatrix1["mut_count"] = np.sum(mutMatrix1[mutant_HSCs1] > 0, axis=1)
mutMatrix1["counts"] = mutMatrix1["WT_count"] + mutMatrix1["mut_count"]

all_HSCs2 = mutMatrix2.columns[7:-1]
mutant_HSCs2 = np.nonzero(mutMatrix2[mutMatrix1["gene"] == 'JAK2'][list(all_HSCs2)].values)[1]
mutant_HSCs2 = all_HSCs2[mutant_HSCs2]
WT_HSCs2 = list(set(all_HSCs2) - set(mutant_HSCs2))

mutMatrix2["WT_count"] = np.sum(mutMatrix2[WT_HSCs2] > 0, axis=1)
mutMatrix2["mut_count"] = np.sum(mutMatrix2[mutant_HSCs2] > 0, axis=1)
mutMatrix2["counts"] = mutMatrix2["WT_count"] + mutMatrix2["mut_count"]

filter_out_UMIs = ["GCCGCTTTGTGC", "ATTATGCCTCGT"]

# scRNA-seq mutation mapping (mapping_p114_mutations_using_p0114_BM_BAM_3_runs.tsv)
filtered_reads = pd.read_csv(..., sep='\t')
# mutation calling quality info (reliable_mutations_mapping_p114_mutations_using_p0114_BM_BAM_3_runs.tsv)
reads_summary = pd.read_csv(..., sep='\t')
reads_summary1 = reads_summary[reads_summary["mutation"] != "intergenic_._LINC01858;ZNF730"]
filtered_reads = filtered_reads[~np.isin(filtered_reads["UMI"], filter_out_UMIs)]
sc_BM_anno = filter_by_reads(filtered_reads, sc_ET1.obs_names, read_threshold=0)
filtered_reads1 = set(reads_summary1["start"]+1).intersection(sc_BM_anno.columns)
sc_BM_anno1 = sc_BM_anno[filtered_reads1]

# scRNA-seq mutation mapping (mapping_p311_mutations_using_p0311_BM_BAM.tsv)
filtered_reads2 = pd.read_csv(..., sep='\t')
# mutation calling quality info (reliable_mutations_mapping_p0311_mutations_using_p0311_BM_BAM.tsv)
reads_summary2 = pd.read_csv(..., sep='\t')
sc_BM_anno = filter_by_reads(filtered_reads2, sc_ET2.obs_names, read_threshold=0)
filtered_reads2 = set(reads_summary2["start"]+1).intersection(sc_BM_anno.columns)
sc_BM_anno2 = sc_BM_anno[filtered_reads2]

# PLOTTING MEAN MUT TRANSCRIPT FRACTION VS WGS HSCs WITH MUTATION (Fig. 5C)
# example with ET 1, for ET 2 change suffixes 1 -> 2
mean_fracs = reads_summary1.groupby("mut_colony_count")["mut_rate"].mean()
WT_counts = reads_summary1.groupby("mut_colony_count")["WT"].sum()
mut_counts = reads_summary1.groupby("mut_colony_count")["mut"].sum()
summed_fracs = mut_counts/(WT_counts+mut_counts)
summed_fracs = pd.DataFrame({"frac": summed_fracs, "error":np.sqrt(summed_fracs * (1-summed_fracs)/(WT_counts+mut_counts))})

plt.figure(figsize=(6,3))
plt.scatter(x=summed_fracs.index, y=summed_fracs["frac"], color="black")
plt.errorbar(x=summed_fracs.index, y=summed_fracs["frac"], yerr=summed_fracs["error"], fmt="none", color="black")
plt.xlabel("number of WGS HSCs with mutation", fontsize=18)
plt.ylabel("mean mutant\ntranscript fraction", fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.tight_layout()
#plt.savefig()
plt.show()

# PLOTTING CLONE FRACTION (Fig. 5B)
# example with ET 1, for ET 2 change suffixes 1 -> 2
ordered_HSC_1_mut = ["HSC.5", "MPP.48", "HSC.50", "HSC.30", "HSC.48", "HSC.45", "HSC.49", "MPP.22", "HSC.34", "MPP.34", "MPP.1", "HSC.39", "HSC.33", "HSC.6", "HSC.32", "HSC.37", "HSC.35", "MPP.39", "HSC.10", "HSC.51", "HSC.47", "HSC.46"]
ordered_HSC_1_WT = ["MPP.16", "MPP.43", "HSC.13", "MPP.30", "MPP.73", "MPP.33", "MPP.53", "MPP.38", "HSC.21", "HSC.52", "MPP.28", "MPP.10", "MPP.120", "MPP.36", "MPP.95", "HSC.36", "HSC.18", "HSC.14", "HSC.41"]
ordered_HSC_2_mut = ["HSC.26_1", "HSC.20_1", "MPP.84_1", "HSC.2_1", "HSC.37_1", "HSC.14_1", "MPP.49_1", "MPP.18_1", "HSC.32_1", "HSC.10_1", "HSC.38_1", "HSC.12_1", "HSC.11_1"]
ordered_HSC_2_WT = ["HSC.13_1", "MPP.83_1", "HSC.15_1", "MPP.80_1", "MPP.73_1", "MPP.95_1", "MPP.33_1", "HSC.5_1", "MPP.57_1", "MPP.21_1", "MPP.61_1", "HSC.9_1", "MPP.48_1", "MPP.55_1", "MPP.87_1", "MPP.46_1", "HSC.21_1", "MPP.62_1", "MPP.72_1", "MPP.92_1", "MPP.51_1"]

reds = LinearSegmentedColormap.from_list(colors=["maroon", "red", "pink"], name="RdBlk")
blues = LinearSegmentedColormap.from_list(colors=["cornflowerblue", "blue", "navy"], name="BlBlk")

single_clones = mutMatrix1[mutMatrix1["counts"] == 1]

clone_fracs = []
clone_errors = []
total_calls = []
cancer_counts = []
HSC_ids = []
for HSC in ordered_HSC_1_mut + ordered_HSC_1_WT:
    mutations = single_clones[np.isin(single_clones["V2"], sc_BM_anno1.columns)]
    sc_ET1 = add_somatic_muts(mutations, sc_BM_anno1, sc_ET1, to_drop=[], prefix=HSC, select_HSC=HSC)
    total = np.sum(sc_ET1.obs[HSC+'_cancer_present']) + np.sum(sc_ET1.obs[HSC+'_WT_present'])
    total_cancer = np.sum(sc_ET1.obs[HSC+'_cancer_present'])

    HSC_ids.append(HSC)
    prob = total_cancer/(total)
    clone_fracs.append(prob)
    clone_errors.append(np.sqrt(prob * (1-prob)/total))
    total_calls.append(total)
    cancer_counts.append(total_cancer)
    
fig, ax = plt.subplots(figsize=(8,2))
plt.bar(ordered_HSC_1_mut, height=clone_fracs[:len(ordered_HSC_1_mut)], color=[reds(x/len(ordered_HSC_1_mut)) for x in range(len(ordered_HSC_1_mut))], edgecolor="black")
plt.bar(ordered_HSC_1_WT, height=clone_fracs[len(ordered_HSC_1_mut):], color=[blues(x/len(ordered_HSC_1_WT)) for x in range(len(ordered_HSC_1_WT))], edgecolor="black")

plt.errorbar(ordered_HSC_1_mut + ordered_HSC_1_WT, y=clone_fracs, yerr=clone_errors, color="black", fmt="none")
plt.xticks(ordered_HSC_1_mut + ordered_HSC_1_WT, [""]* len(ordered_HSC_1_mut + ordered_HSC_1_WT), rotation=90)
sns.despine(top=False, bottom=True)
plt.yticks(fontsize=14)
plt.ylim(0,0.7)
ax.invert_yaxis()
plt.tight_layout()

#plt.savefig()
plt.show()

# ASSIGNING CLADES TO scRNA-seq DATA (required for the remainder of Fig. 5)
# example with ET 1, for ET 2 change suffixes 1 -> 2
mutMatrix1["counts"] = np.sum(mutMatrix1[ordered_HSC_1_mut + ordered_HSC_1_WT] > 0, axis=1)
mutMatrix1["WT_count"] = np.sum(mutMatrix1[ordered_HSC_1_WT] > 0, axis=1)
mutMatrix1["mut_count"] = np.sum(mutMatrix1[ordered_HSC_1_mut] > 0, axis=1)
filtered_matrix = mutMatrix[np.isin(mutMatrix["V2"], list(filtered_reads))]
is_mut_clade = np.logical_and(np.logical_and(filtered_matrix["mut_count"] > 1, filtered_matrix["mut_count"] < 21), filtered_matrix["WT_count"]==0)
is_WT_clade = np.logical_and(np.logical_and(filtered_matrix["WT_count"] > 1, filtered_matrix["WT_count"] < 19), filtered_matrix["mut_count"]==0)
filtered_matrix = filtered_matrix[np.logical_or(is_mut_clade, is_WT_clade)]

bool_muts = filtered_matrix[ordered_HSC_1_mut+ordered_HSC_1_WT] > 0
bool_muts.sort_values(by=ordered_HSC_1_mut+ordered_HSC_1_WT, inplace=True)

clades1 = get_clades(bool_muts, filtered_matrix)

sc_ET1.obs["clade_ID"] = ["none"] * len(sc_ET1)
gene_to_clade = dict(zip(clades["gene"], clades["clade_ID"]))

cancer_counts = [0]*(np.amax(clades["clade_ID"])+1)
total_counts = [0]*(np.amax(clades["clade_ID"])+1)
for gene in filtered_matrix['gene']:
    sc_ET1 = add_somatic_muts(filtered_matrix, sc_BM_anno1, sc_ET1, to_drop=[], prefix=gene, select_mut=gene)
    clade = gene_to_clade[gene]
    sc_ET1.obs["clade_ID"] = [clade if sc_ET1.obs[gene+"_cancer_present"].iloc[i] else sc_ET1.obs["clade_ID"].iloc[i] for i in range(len(sc_ET1.obs["clade_ID"]))]
    if clade != "none":
        cancer_counts[clade] += np.sum(sc_ET1.obs[gene+"_cancer_present"])
        total_counts[clade] += np.sum(sc_ET1.obs[gene+"_cancer_present"]+sc_ET1.obs[gene+"_WT_present"])
clade_fracs1 = np.array(cancer_counts)/np.array(total_counts)
clade_errors1 = (np.sqrt(clade_fracs * (1-clade_fracs)/total_counts))

# PLOTTING CLONES/CLADES ON scRNA-seq UMAP (Fig. 5C)
# example with ET 1, for ET 2 change suffixes 1 -> 2
oranges = LinearSegmentedColormap.from_list(colors=["gold", "orange", "darkorange"], name="og")
greens = LinearSegmentedColormap.from_list(colors=["palegreen", "lawngreen", "green", "darkgreen"], name="gr")
yellows = LinearSegmentedColormap.from_list(colors=["fuchsia", "violet"], name="y")

# group clades with desired colors
green_num = [0,1,2,3,9,10]
yellow_num = [4,5]
orange_num = [6,7]
purple_num = [8]

to_plot = sc_ET1

embed_name = "X_umap"

plt.figure()
plt.scatter(to_plot.obsm[embed_name][:,0], to_plot.obsm[embed_name][:,1], c="lightgrey", s=3, zorder=1)

for i in range(len(ordered_HSC_1_mut)):
    HSC = ordered_HSC_1_mut[i]
    filters = np.nonzero(to_plot.obs[HSC+"_cancer_present"])[0]
    plt.scatter(to_plot.obsm[embed_name][filters,0], to_plot.obsm[embed_name][filters,1], color=reds(i/len(ordered_HSC_1_mut)), s=9, zorder=2)
for i in range(len(ordered_HSC_1_WT)):
    HSC = ordered_HSC_1_WT[i]
    filters = np.nonzero(to_plot.obs[HSC+"_cancer_present"])[0]
    plt.scatter(to_plot.obsm[embed_name][filters,0], to_plot.obsm[embed_name][filters,1], color=blues(i/len(ordered_HSC_1_WT)), s=9, zorder=2)

all_clades = list(set(to_plot.obs["clade_ID"]))
all_clades.remove("none")
for i in range(len(all_clades)):
    clade = all_clades[i]
    if clade == "none":
        continue
    filters = np.nonzero(to_plot.obs["clade_ID"]==clade)[0]
    if clade in green_num:
        curr_color = greens(green_num.index(clade)/(len(green_num)-1))
    elif clade in yellow_num:
        curr_color = yellows(yellow_num.index(clade)/(len(yellow_num)-1))
    elif clade in orange_num:
        curr_color = oranges(orange_num.index(clade)/(len(orange_num)-1))
    elif clade in purple_num:
        curr_color = "mediumslateblue"
    plt.scatter(to_plot.obsm[embed_name][filters,0], to_plot.obsm[embed_name][filters,1], color=curr_color, s=9, zorder=2)
plt.axis('off')
#plt.savefig()
plt.show()

# PLOTTING CLADE FREQUENCIES (Fig. 5C)
# example with ET 1, for ET 2 change suffixes 1 -> 2
clade_to_HSC = dict(zip(clades1["clade_ID"], clades1["num_HSC"]))
num_HSC = [clade_to_HSC[x] for x in range(len(set(clades1["clade_ID"])))]
clade_fracs = pd.DataFrame({"clade_ID":range(len(set(clades1["clade_ID"]))), "num_HSC":num_HSC, "frac_mut":clade_fracs1, "errors":clade_errors1})

clade_order = [0,1,2,3,9,10,4,5,6,7,8] #reorder if desired
plt.figure(figsize=(6,3))
for i in range(len(clade_order)):
    clade = clade_order[i]
    if clade in green_num:
        curr_color = greens(green_num.index(clade)/(len(green_num)-1))
    elif clade in yellow_num:
        curr_color = yellows(yellow_num.index(clade)/(len(yellow_num)-1))
    elif clade in orange_num:
        curr_color = oranges(orange_num.index(clade)/(len(orange_num)-1))
    elif clade in purple_num:
        curr_color = "mediumslateblue"
    plt.errorbar(x=i, y=clade_fracs.iloc[clade]["frac_mut"], yerr=clade_fracs.iloc[clade]["errors"], color="black", fmt="none", zorder=2)
    plt.bar(x=i, height=clade_fracs.iloc[clade]["frac_mut"], color=curr_color, zorder=1, edgecolor="black")
plt.xlabel("clade", fontsize=18)
plt.ylabel("fraction of cells\nwith mutant transcript", fontsize=18)
plt.xticks([])
plt.yticks(fontsize=14)
plt.tight_layout()
#plt.savefig()
plt.show()