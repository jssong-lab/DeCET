# DeCET

### Repository Contents

* [DeCET_core_scripts](./DeCET_core_scripts) contains scripts for performing the core components of the DeCET method.
    * [prep_data_tensor.py](./DeCET_core_scripts/prep_data_tensor.py) formats histone modification ChIP-seq data sets (or other genomic signal data sets specified in bed format) into a PyTorch double tensor.
    * [DeCET_HOSVD.py](./DeCET_core_scripts/DeCET_HOSVD.py) code for performing the Higher Order Singular Value Decomposition (HOSVD) of the epigenetic data tensor.
    * [MOR_scaling_ChIPseq.py](./DeCET_core_scripts/MOR_scaling_ChIPseq.py) scaling function implementing the median of ratios (MOR) scaling of the epigenetic tensor. Specific to an order 4 tensor with the MOR performed over the first and second indices. This scaling function was used for the leiomyoma and breast cancer datasets.
    * [MOR_order3_scaling_ChIPseq.py](./DeCET_core_scripts/MOR_order3_scaling_ChIPseq.py) scaling function implementing the median of ratios (MOR) scaling of order three epigenetic tensors. This scaling function was used for the REMC and prostate cancer datasets.

* [DeCET_downstream_analysis](./DeCET_downstream_analysis) additional code used for analyzing the results from the [DeCET_core_scripts](./DeCET_core_scripts).
    * [analysis_of_HOSVD_projections.ipynb](./DeCET_downstream_analysis/analysis_of_HOSVD_projections.ipynb) a jupyter notebook containing the code for analyzing the projections onto the location vectors returned by the HOSVD. This notebook also contains the code for identifying regions with epigenetic alterations. Includes code for generating Figures 1B, 1C, 4B, 4C, 6A, S3A-F, S4A-F, S5B, S6D.
    * [classification_cross_validation.py](./DeCET_downstream_analysis/classification_cross_validation.py) a Python script for running the disease and subtype classification cross-validation described in the study.
    * [region_functional_characterization.ipynb](./DeCET_downstream_analysis/region_functional_characterization.ipynb) a jupyter notebook containing code for determining the regulatory function of regions with altered histone modifications. Includes code for generating Figures 5A-D, S6A-C.
    * [contact_domain_alterations.ipynb](./DeCET_downstream_analysis/contact_domain_alterations.ipynb) a jupyter notebook containing code for study the confinement of epigenetic alterations to chromatin contact domains. Includes code for generating Figures 6C-F, S6E-G.
    * [define_gene_regulatory_regions.ipynb](./DeCET_downstream_analysis/define_gene_regulatory_regions.ipynb) a jupyter notebook used to define regulatory regions around the TSS of each gene; the notebook also contains code for calculating and filtering genes by FPKM.
    * [epigenetically_regulated_genes.ipynb](./DeCET_downstream_analysis/epigenetically_regulated_genes.ipynb) a jupyter notebook used to identify genes near epigenetic alterations and filter for genes that are also differentially expressed.
    * [motif_calling_analysis.ipynb](./DeCET_downstream_analysis/motif_calling_analysis.ipynb) a jupyter notebook for analyzing the results of motif calling.
    * [super_enhancer_overlap.ipynb](./DeCET_downstream_analysis/super_enhancer_overlap.ipynb) use the results from calling super enhancers to analyze the overlap of super enhancers and regions with identified epigenetic alterations.
    * [auxiliary_bed_functions.py](./DeCET_downstream_analysis/auxiliary_bed_functions.py) some useful functions for working with bed formated (or bed like) files. These are used in the jupyter notebooks in this study.
    * [DeCET_REMC_adult_tissues_final.ipynb](./DeCET_downstream_analysis/DeCET_REMC_adult_tissues_final.ipynb) application of DeCET to REMC adult tissue dataset. Includes code for generating Figure 2A.
    * [DeCET_REMC_muscle_types_final.ipynb](./DeCET_downstream_analysis/DeCET_REMC_muscle_types_final.ipynb) application of DeCET to REMC adult muscle tissues. Includes code for generating Figure 2B.
    * [DeCET_REMC_Tcell_final.ipynb](./DeCET_downstream_analysis/DeCET_REMC_Tcell_final.ipynb) application of DeCET to REMC T cell samples. Includes code for generating Figure 2C.
    * [DeCET_BCCL_final.ipynb](./DeCET_downstream_analysis/DeCET_BCCL_final.ipynb) applicaton of DeCET to breast cancer. Includes code for generating Figures 3A and 3B.
    * [DeCET_PCa_final.ipynb](./DeCET_downstream_analysis/DeCET_PCa_final.ipynb) application of DeCET to prostate cancer. Includes code for generating Figure 3C.
