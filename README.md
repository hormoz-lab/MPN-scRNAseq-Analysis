# MPN scRNAseq analysis

Scripts used for analyzing MPN patient scRNA-seq analysis in Van Egeren et al.

scRNA-seq inputs to the pipeline are 10X h5 filtered bc matrices. Amplicon calls for each patient (csv files) are required to build the scanpy objects that will be used for most of the analysis. HSC WGS data is required for the mutation_mapping_scRNAseq.py script. All data can be found on the ftp server.

Most of the python files require scanpy, numpy, and pandas, further requirements are listed in the scripts themselves. R/Seurat v3 was used for batch correction (seurat_integrate.R).

The workflow scheme is as follows:
preprocessing.py is used first to clean and filter the raw 10X data. The cleaned data is passed into Seurat to perform batch correction (seurat_integrate.R), and the integrated dataset is again loaded into Python to define cell type clusters using the integrated data (cluster_plot_UMAPs.py). The cluster IDs are then transferred back into the original scanpy data objects (no batch correction) for further analysis, including
1. Plotting UMAPs with cell type and mutation calls (Fig. 2A-C, cluster_plot_UMAPs.py)
2. Plotting general JAK2 mutation frequency statistics (Fig 2D-E, plot_freqs.py)
3. Mapping somatic mutations onto scRNA-seq data (Fig. 5, mutation_mapping_scRNAseq.py)
4. Performing differential expression analysis between the JAK2-mutant and WT cells (Fig. 6, diff_exp.py)
5. Kinetic analysis of differentiation using Population Balance Analysis (Supplement, PBA.py)

Questions? Email [Debra Van Egeren](mailto:dvanegeren@g.harvard.edu).