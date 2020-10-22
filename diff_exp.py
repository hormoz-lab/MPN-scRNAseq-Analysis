#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 12:31:10 2020

@author: dve
"""

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import loompy, h5py, sys, os, random, csv, gseapy, string, csv, io
import pandas as pd
from statsmodels.stats.multitest import multipletests

# HELPER FUNCTIONS
def filter_by_pvals_scanpy(sc_data, group_id, alpha, filter_genes=None):
    pvals = sc_data.uns['rank_genes_groups']['pvals_adj'][group_id]
    raw_pvals = sc_data.uns['rank_genes_groups']['pvals'][group_id]
    names = sc_data.uns['rank_genes_groups']['names'][group_id]
    logchanges = sc_data.uns['rank_genes_groups']['logfoldchanges'][group_id]
    to_return = pd.DataFrame({"gene": names, "raw_pval": raw_pvals, "pval":pvals, "logfoldchange":logchanges})
    if not filter_genes is None:
        to_return = to_return[np.isin(to_return["gene"], filter_genes)]
        to_return["pval"] = to_return["raw_pval"]*len(to_return["raw_pval"])
    return(to_return[to_return["pval"] < alpha])

def plot_generic_volcano(diff_exp_df, alpha, labels=[], colors=[], add_text=[]):
    to_plot = diff_exp_df.copy()

    to_plot["sig"] = to_plot["pval"] < alpha
    
    cmap = {True:"dimgrey", False:"lightgrey"}
    fig,ax=plt.subplots(1,1)
    to_plot_y = -np.log10(to_plot["pval"])
    to_plot_y = np.minimum(to_plot_y, np.amax(to_plot_y[np.isfinite(to_plot_y)]))
    plt.scatter(to_plot["logfoldchange"], to_plot_y, c=[cmap[x] for x in to_plot["sig"]], s=30)
    x_offset = 0
    y_offset = 0
    for i in range(len(labels)):
        label_group = labels[i]
        for label in label_group:
            if label not in to_plot.index.tolist():
                continue
            row = to_plot.loc[label]
            if row["sig"]:
                plt.scatter(row["logfoldchange"], -np.log10(row["pval"]), s=30, color=colors[i])
                if add_text[i]:
                    if row["logfoldchange"] > 0:
                        plt.text(row["logfoldchange"]+x_offset, -np.log10(row["pval"])+y_offset, s=label, color=colors[i])
                    else:
                        plt.text(row["logfoldchange"]-x_offset, -np.log10(row["pval"])+y_offset, s=label, color=colors[i], horizontalalignment='right')
    plt.axvline(linestyle="--", color="black")
    plt.xlabel("log2 fold expression change")
    plt.ylabel("-log10 of adjusted p-value")
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    plt.ylim(ymin=0)
    return(fig)

def plot_gseapy_ontology(go_test, alpha=0.05, max_num=10):
    to_plot = go_test.results[go_test.results["Adjusted P-value"] < alpha]
    if len(to_plot) > max_num:
        to_plot = to_plot[:max_num]
    fig = plt.figure()
    plt.barh(-np.arange(len(to_plot)), -np.log10(to_plot["Adjusted P-value"]))
    plt.yticks(-np.arange(len(to_plot)), to_plot["Term"])
    plt.xlabel("-log10 of adjusted p-value")
    return(fig)

def combine_diff_exp_pvals(all_sc_data, group_id):
    all_results_dfs = []
    first = True
    for sc_data in all_sc_data:
        pvals = sc_data.uns['rank_genes_groups']['pvals_adj'][group_id]
        raw_pvals = sc_data.uns['rank_genes_groups']['pvals'][group_id]
        names = sc_data.uns['rank_genes_groups']['names'][group_id]
        logchanges = sc_data.uns['rank_genes_groups']['logfoldchanges'][group_id]
        all_results_dfs.append(pd.DataFrame({"gene": names, "raw_pval": raw_pvals, "pval":pvals, "logfoldchange":logchanges}))
        if first:
            genes_to_use = set(sc_data.uns['rank_genes_groups']['names'][group_id])
            first = False
        else:
            genes_to_use = genes_to_use.intersection(set(sc_data.uns['rank_genes_groups']['names'][group_id]))
    genes_to_use = list(genes_to_use)
    combined_pvals = []
    all_pvals = []
    for sc_df in all_results_dfs:
        sc_df.set_index("gene", inplace=True)
        sc_df = sc_df.loc[genes_to_use]
        all_pvals.append(sc_df["raw_pval"].values)
    all_pvals = np.array(all_pvals)
    for i in range(len(genes_to_use)):
        fisher_stat, pval = stats.combine_pvalues(all_pvals[:,i])
        combined_pvals.append(pval)
    to_return = pd.DataFrame({"gene":genes_to_use, "raw_pval":combined_pvals})
    return(to_return)

def combine_diff_exp_logfold(all_sc_data, groupbys, useraw=True):
    first = True
    for sc_data in all_sc_data:
        if first:
            genes_to_use = set(sc_data.var.index)
            first = False
        else:
            genes_to_use = genes_to_use.intersection(set(sc_data.var.index))
    genes_to_use = list(genes_to_use)
    
    all_diffs = []
    all_cell_num = np.array([len(x) for x in all_sc_data])
    all_cell_num = all_cell_num/np.sum(all_cell_num)
    for i in range(len(all_sc_data)):
        sc_data = all_sc_data[i]
        groupby = groupbys[i]
        sc_data_WT = sc_data[~sc_data.obs[groupby]]
        sc_data_mut = sc_data[sc_data.obs[groupby]]
        if useraw:
            avg_WT = np.mean(sc_data_WT.raw.X.todense(), axis=0)
            avg_mut = np.mean(sc_data_mut.raw.X.todense(), axis=0)
        else:
            avg_WT = np.mean(sc_data_WT.X, axis=0)
            avg_mut = np.mean(sc_data_mut.X, axis=0)
        avg_diffs = np.log2(np.divide(avg_mut,avg_WT))
        avg_diffs = pd.DataFrame(np.transpose(avg_diffs), index=sc_data.var.index, columns=["diffs"])
        avg_diffs = avg_diffs.loc[genes_to_use]
        all_diffs.append(avg_diffs["diffs"].values)
    all_diffs = np.transpose(np.array(all_diffs))
    to_return = pd.DataFrame(all_diffs.copy(), index=genes_to_use)
    all_diffs = np.multiply(all_diffs, all_cell_num)
    all_diffs = pd.DataFrame(np.sum(all_diffs, axis=1), index=genes_to_use, columns=["logfoldchange"])
    return(all_diffs, to_return)


# LOAD DATA (.h5ad scanpy files)
sc_PV1 = sc.read(...)
sc_PV2 = sc.read(...)
sc_PV3 = sc.read(...)
sc_ET1 = sc.read(...)
sc_ET2 = sc.read(...)
sc_ET3 = sc.read(...)
sc_ET_V617L = sc.read(...)

# INTRAPATIENT COMPARISONS, COMBINED BY DISEASE TYPES
all_cell_types = ["HSC", "MEP", "erythroid", "megakaryocyte", "CD14+", "GMP"]

PV_data = [sc_PV1, sc_PV2, sc_PV3]
PV_labels = ['PV1', 'PV2', 'PV3']
ET_data = [sc_ET1, sc_ET2, sc_ET3]
ET_labels = ['ET1', 'ET2', 'ET3']

all_combined_pval_PV = {}
all_indiv_changes_PV = {}
for cell_type in all_cell_types:
    all_cell_type_data = []
    for i in range(len(PV_data)):
        sc_data = PV_data[i]
        label = PV_labels[i]
        to_plot = sc_data[sc_data.obs["cell_type"]==cell_type]
        if label in ["ET1", "ET2"]:
            to_plot = to_plot[np.logical_or(to_plot.obs['all_cancer_present'], to_plot.obs['all_WT_present'])]
            to_plot.obs['group_diff'] = to_plot.obs['all_cancer_present'].astype('category')
        else:
            to_plot = to_plot[np.logical_or(to_plot.obs['jak2_cancer_present'], to_plot.obs['jak2_WT_present'])]
            to_plot.obs['group_diff'] = to_plot.obs['jak2_cancer_present'].astype('category')
        try:
            sc.tl.rank_genes_groups(to_plot, groupby='group_diff', rankby_abs=True, use_raw=True, n_genes=len(sc_data.var.index), method="wilcoxon")
            all_cell_type_data.append(to_plot)
        except:
            pass
    if len(all_cell_type_data) > 0:
        combined_logs, indiv_logs = combine_diff_exp_logfold(all_cell_type_data, ['jak2_cancer_present', 'jak2_cancer_present', 'jak2_cancer_present'], useraw=True)
        combined_pvals = combine_diff_exp_pvals(all_cell_type_data, "True")
        reject, pvals_corr, sidak, bonf = multipletests(combined_pvals["raw_pval"])
        combined_pvals["pval"] = pvals_corr
        combined_pvals = combined_pvals.set_index("gene")
        combined_pvals = combined_pvals.join(combined_logs)
        all_combined_pval_PV[cell_type] = combined_pvals
        all_indiv_changes_PV[cell_type] = indiv_logs
        
all_combined_pval_ET = {}
all_indiv_changes_ET = {}
for cell_type in all_cell_types:
    all_cell_type_data = []
    for i in range(len(ET_data)):
        sc_data = ET_data[i]
        label = ET_labels[i]
        to_plot = sc_data[sc_data.obs["cell_type"]==cell_type]
        if label in ["ET1", "ET2"]:
            to_plot = to_plot[np.logical_or(to_plot.obs['all_cancer_present'], to_plot.obs['all_WT_present'])]
            to_plot.obs['group_diff'] = to_plot.obs['all_cancer_present'].astype('category')
        else:
            to_plot = to_plot[np.logical_or(to_plot.obs['jak2_cancer_present'], to_plot.obs['jak2_WT_present'])]
            to_plot.obs['group_diff'] = to_plot.obs['jak2_cancer_present'].astype('category')
        try:
            sc.tl.rank_genes_groups(to_plot, groupby='group_diff', rankby_abs=True, use_raw=True, n_genes=len(sc_data.var.index), method="wilcoxon")
            all_cell_type_data.append(to_plot)
        except:
            pass
    if len(all_cell_type_data) > 0:
        combined_logs, indiv_logs = combine_diff_exp_logfold(all_cell_type_data, ['jak2_cancer_present', 'jak2_cancer_present', 'jak2_cancer_present'], useraw=True)
        combined_pvals = combine_diff_exp_pvals(all_cell_type_data, "True")
        reject, pvals_corr, sidak, bonf = multipletests(combined_pvals["raw_pval"])
        combined_pvals["pval"] = pvals_corr
        combined_pvals = combined_pvals.set_index("gene")
        combined_pvals = combined_pvals.join(combined_logs)
        all_combined_pval_ET[cell_type] = combined_pvals
        all_indiv_changes_ET[cell_type] = indiv_logs
        
        
# PLOTTING DIFFERENTIAL EXPRESSION VOLCANO PLOTS (Fig. 6)
for cell_type in all_cell_types:
    labels = all_combined_pval_ET[cell_type].index.to_list()
    to_plot = all_combined_pval_ET[cell_type].copy()
    fig = plot_generic_volcano(to_plot, 0.05, labels=[ribosome, antigen, proteasome], colors=["steelblue", "green", "red"], add_text=[False, False, False])
    
    to_save = all_combined_pval_ET[cell_type].copy()
    to_save = to_save[to_save["pval"] < 0.05]
    plt.xlim(-1,1.6)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    #plt.savefig()   
    plt.show()
    
for cell_type in all_cell_types:
    labels = all_combined_pval_PV[cell_type].index.to_list()
    to_plot = all_combined_pval_PV[cell_type].copy()
    fig = plot_generic_volcano(to_plot, 0.05, labels=[ribosome, antigen, proteasome], colors=["steelblue", "green", "red"], add_text=[False, False, False])
    
    to_save = all_combined_pval_ET[cell_type].copy()
    to_save = to_save[to_save["pval"] < 0.05]
    plt.xlim(-1,1.6)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    #plt.savefig()   
    plt.show()
    
# GSEA (supplement)
        
genes_to_test = all_combined_pval_ET["MEP"][all_combined_pval_ET["MEP"]["pval"] < 0.05].index.to_list()
#test = gseapy.enrichr(gene_list=genes_to_test, description='pathway', gene_sets='ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X')
test = gseapy.enrichr(gene_list=genes_to_test, description='pathway', gene_sets='KEGG_2019_Human')
fig = plot_gseapy_ontology(test, max_num=20)
plt.tight_layout()
plt.show()

genes_to_test = all_combined_pval_ET["MEP"][all_combined_pval_ET["MEP"]["pval"] < 0.05].index.to_list()
test = gseapy.enrichr(gene_list=genes_to_test, description='pathway', gene_sets='ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X')
#test = gseapy.enrichr(gene_list=genes_to_test, description='pathway', gene_sets='KEGG_2019_Human')
fig = plot_gseapy_ontology(test, max_num=20)
plt.tight_layout()
plt.show()

genes_to_test = all_combined_pval_ET["CD14+"][all_combined_pval_ET["CD14+"]["pval"] < 0.05].index.to_list()
#test = gseapy.enrichr(gene_list=genes_to_test, description='pathway', gene_sets='ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X')
test = gseapy.enrichr(gene_list=genes_to_test, description='pathway', gene_sets='KEGG_2019_Human')
fig = plot_gseapy_ontology(test, max_num=20)
plt.tight_layout()
plt.show()

genes_to_test = all_combined_pval_ET["CD14+"][all_combined_pval_ET["CD14+"]["pval"] < 0.05].index.to_list()
test = gseapy.enrichr(gene_list=genes_to_test, description='pathway', gene_sets='ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X')
#test = gseapy.enrichr(gene_list=genes_to_test, description='pathway', gene_sets='KEGG_2019_Human')
fig = plot_gseapy_ontology(test, max_num=20)
plt.tight_layout()
plt.show()

# INTRAPATIENT COMPARISONS, FOR INDIVIDUAL PATIENTS

all_sc_data = ET_data + PV_data + [sc_ET_V617L]
all_labels = ET_labels + PV_labels + ["V617L"]

all_num_genes = []
num_genes_types = []
num_genes_labels = []
num_cells = []
min_num_cells = []

all_diff_exp = dict(zip(all_labels, [{}, {}, {}, {}, {}, {}, {}]))

for cell_type in all_cell_types:
    all_celltype = [sc_data[sc_data.obs["cell_type"]==cell_type] for sc_data in all_sc_data]
    combined_logs, indiv_logs = combine_diff_exp_logfold(all_celltype, ['all_cancer_present', 'jak2_cancer_present', 'all_cancer_present', 'jak2_cancer_present', 'jak2_cancer_present', 'jak2_cancer_present', 'jak2_cancer_present'], useraw=True)
    for i in range(len(all_sc_data)):
        sc_data = all_sc_data[i]
        label = all_labels[i]
        to_plot =sc_data[sc_data.obs["cell_type"]==cell_type]
        if label in ["ET1", "ET2"]:
            to_plot = to_plot[np.logical_or(to_plot.obs['all_cancer_present'], to_plot.obs['all_WT_present'])]
            to_plot.obs['group_diff'] = to_plot.obs['all_cancer_present'].astype('category')
        else:
            to_plot = to_plot[np.logical_or(to_plot.obs['jak2_cancer_present'], to_plot.obs['jak2_WT_present'])]
            to_plot.obs['group_diff'] = to_plot.obs['jak2_cancer_present'].astype('category')
        try:
            sc.tl.rank_genes_groups(to_plot, groupby='group_diff', rankby_abs=True, use_raw=True, n_genes=len(sc_data.var.index), method="wilcoxon")
            temp = filter_by_pvals_scanpy(to_plot, "True", 5)
            temp.set_index("gene", inplace=True)
            temp.drop("logfoldchange", axis=1, inplace=True)
            temp = temp.join(indiv_logs[i])
            temp.rename(columns={i:"logfoldchange"}, inplace=True)
            all_diff_exp[label][cell_type] = temp
            all_num_genes.append(len(filter_by_pvals_scanpy(to_plot, "True", 0.05)))
            num_genes_types.append(cell_type)
            num_genes_labels.append(label)
            num_cells.append(len(to_plot))
            min_num_cells.append(min(len(to_plot[to_plot.obs["group_diff"] == False]), len(to_plot[to_plot.obs["group_diff"] == True])))
        except:
            pass
        
for cell_type in all_cell_types:
    for label in all_labels:
        try:
            to_save = all_diff_exp[label][cell_type]
            
            fig = plot_generic_volcano(to_save, 0.05, labels=[ribosome, antigen, proteasome], colors=["steelblue", "green", "red"], add_text=[False, False, False])

            plt.xlim(-2,2)
            plt.ylim(ymin=0)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel("")
            plt.ylabel("")
            plt.tight_layout()
            
            #plt.savefig()
            plt.show()
            
        except KeyError:
            pass

# WRITING LISTS OF DIFFERENTIALLY-EXPRESSED GENES (Tables S3-4)
alpha = 0.05

# PV
for cell_type in all_cell_types:
    print(cell_type)
    genes_to_use = set(all_combined_pval_PV[cell_type].index)
    labels_to_use = []
    for label in all_labels:
        try:
            genes_to_use = genes_to_use.intersection(set(all_diff_exp[label][cell_type].index))
            labels_to_use.append(label)
        except KeyError:
            pass
    cell_type_data = all_combined_pval_PV[cell_type].loc[list(genes_to_use)].copy()
    for label in labels_to_use:
        all_diff_exp[label][cell_type][label] = all_diff_exp[label][cell_type]["pval"] < alpha
        cell_type_data = cell_type_data.join(all_diff_exp[label][cell_type][label])
    cell_type_data["num_ET"] = np.sum(cell_type_data[set(labels_to_use).intersection(set(ET_labels))], axis=1)
    cell_type_data["num_PV"] = np.sum(cell_type_data[set(labels_to_use).intersection(set(PV_labels))], axis=1)
    def get_sig_string(gene):
        to_test = cell_type_data.loc[gene]
        to_test = to_test[labels_to_use]
        sig_labels = [labels_to_use[x] for x in np.nonzero(to_test.values)[0]]
        return(", ".join(sig_labels))
    cell_type_data["sig_pts"] = [get_sig_string(x) for x in cell_type_data.index]
    to_save = np.logical_or(~(cell_type_data["sig_pts"] == ""), cell_type_data["pval"] < alpha)
    to_save = cell_type_data[to_save]
    to_save = to_save[["pval", "logfoldchange", "num_ET", "num_PV", "sig_pts"]]
    to_save = to_save.sort_values(by=["pval"], ascending=True)
    #to_save.to_csv()

    
# ET
for cell_type in all_cell_types:
    print(cell_type)
    genes_to_use = set(all_combined_pval_ET[cell_type].index)
    labels_to_use = []
    for label in all_labels:
        try:
            genes_to_use = genes_to_use.intersection(set(all_diff_exp[label][cell_type].index))
            labels_to_use.append(label)
        except KeyError:
            pass
    cell_type_data = all_combined_pval_ET[cell_type].loc[list(genes_to_use)].copy()
    for label in labels_to_use:
        all_diff_exp[label][cell_type][label] = all_diff_exp[label][cell_type]["pval"] < alpha
        cell_type_data = cell_type_data.join(all_diff_exp[label][cell_type][label])
    cell_type_data["num_ET"] = np.sum(cell_type_data[set(labels_to_use).intersection(set(ET_labels))], axis=1)
    cell_type_data["num_PV"] = np.sum(cell_type_data[set(labels_to_use).intersection(set(PV_labels))], axis=1)
    def get_sig_string(gene):
        to_test = cell_type_data.loc[gene]
        to_test = to_test[labels_to_use]
        sig_labels = [labels_to_use[x] for x in np.nonzero(to_test.values)[0]]
        return(", ".join(sig_labels))
    cell_type_data["sig_pts"] = [get_sig_string(x) for x in cell_type_data.index]
    to_save = np.logical_or(~(cell_type_data["sig_pts"] == ""), cell_type_data["pval"] < alpha)
    to_save = cell_type_data[to_save]
    to_save = to_save[["pval", "logfoldchange", "num_ET", "num_PV", "sig_pts"]]
    to_save = to_save.sort_values(by=["pval"], ascending=True)
    #to_save.to_csv()
