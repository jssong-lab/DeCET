# DeCET

## Repository Contents

* [DeCET_core_scripts](./DeCET_core_scripts) contains scripts for performing the core components of DeCET.
    * [DeCET_get_data_tensor.py](./DeCET_core_scripts/DeCET_get_data_tensor.py) formats histone modification ChIP-seq datasets (or other genomic signal datasets) into a PyTorch double tensor. The ChIP-seq data must be input as aligned single-end reads in bed format.
    * [DeCET_HOSVD.py](./DeCET_core_scripts/DeCET_HOSVD.py) obtains the Higher Order Singular Value Decomposition (HOSVD) of the ChIP-seq data tensor.
    * [MOR_scaling_ChIPseq.py](./DeCET_core_scripts/MOR_scaling_ChIPseq.py) scaling function implementing the median of ratios (MOR) scaling of the ChIP-seq data tensor. Specific to an order 4 tensor with the MOR performed over the first and second indices. This scaling function was used for the leiomyoma and breast cancer datasets.
    * [MOR_order3_scaling_ChIPseq.py](./DeCET_core_scripts/MOR_order3_scaling_ChIPseq.py) scaling function implementing the median of ratios (MOR) scaling of an order 3 ChIP-seq data tensor. The MOR scaling is performed over the first index separately for each value of the second index. This scaling function was used for the REMC and prostate cancer datasets.

* [DeCET_downstream_analysis](./DeCET_downstream_analysis) Python scripts and Jupyter notebooks used for analyzing the results from the [DeCET_core_scripts](./DeCET_core_scripts).
    * [analysis_of_HOSVD_projections.ipynb](./DeCET_downstream_analysis/analysis_of_HOSVD_projections.ipynb) a Jupyter notebook used for analyzing the HOSVD projections for the leiomyoma and myometrium dataset. This notebook contains the code used for identifying regions with epigenetic alterations and for generating Figures 1B, 1C, 2B, 2C, 4A, 8E, S3A-F, S4A-F, S5B, S6D.
    * [classification_cross_validation.py](./DeCET_downstream_analysis/classification_cross_validation.py) a Python script for running the disease and subtype classification cross-validation described in the study.
    * [region_functional_characterization.ipynb](./DeCET_downstream_analysis/region_functional_characterization.ipynb) a Jupyter notebook containing code for determining the regulatory function of regions with altered histone modifications in leiomyoma. Includes code for generating Figures 3A-D, S6A-C.
    * [contact_domain_alterations.ipynb](./DeCET_downstream_analysis/contact_domain_alterations.ipynb) a Jupyter notebook containing code for studying the confinement of epigenetic alterations to chromatin contact domains in leiomyoma. Includes code for generating Figures 4C-F, S6E-G.
    * [define_gene_regulatory_regions.ipynb](./DeCET_downstream_analysis/define_gene_regulatory_regions.ipynb) a Jupyter notebook used to define regulatory regions around the TSS of each gene; the notebook also contains code for calculating and filtering genes by FPKM.
    * [epigenetically_regulated_genes.ipynb](./DeCET_downstream_analysis/epigenetically_regulated_genes.ipynb) a Jupyter notebook used to identify genes near epigenetic alterations and filter for genes that are also differentially expressed in leiomyoma.
    * [ATACseq_merged_summit_pileup.ipynb](./DeCET_downstream_analysis/ATACseq_merged_summit_pileup.ipynb) a Jupyter notebook for analyzing ATAC-seq pileup in peak regions overlapping regions with altered histone modifications in leiomyoma. Includes code for generating Figures S7A and S7B.
    * [motif_calling_analysis.ipynb](./DeCET_downstream_analysis/motif_calling_analysis.ipynb) a Jupyter notebook for analyzing the results of motif calling in leiomyoma.
    * [Myo_HOXA13_ChIPseq.ipynb](./DeCET_downstream_analysis/Myo_HOXA13_ChIPseq.ipynb) a Jupyter notebook for analyzing HOXA13 ChIP-seq data from primary myometrium cells. Includes code for generating Figures 5E and 5F.
    * [super_enhancer_overlap.ipynb](./DeCET_downstream_analysis/super_enhancer_overlap.ipynb) a Jupyter notebook comparing the overlap of DeCET altered regions and called super and normal enhancers in leiomyoma.
    * [DeCET_REMC_adult_tissues_final.ipynb](./DeCET_downstream_analysis/DeCET_REMC_adult_tissues_final.ipynb) application of DeCET to REMC adult tissue dataset. Includes code for generating Figures 6A and 8A.
    * [DeCET_REMC_muscle_types_final.ipynb](./DeCET_downstream_analysis/DeCET_REMC_muscle_types_final.ipynb) application of DeCET to REMC adult muscle tissues. Includes code for generating Figures 6B and 8B.
    * [DeCET_REMC_Tcell_final.ipynb](./DeCET_downstream_analysis/DeCET_REMC_Tcell_final.ipynb) application of DeCET to REMC T cell samples. Includes code for generating Figures 6C and 8C.
    * [DeCET_BCCL_final.ipynb](./DeCET_downstream_analysis/DeCET_BCCL_final.ipynb) application of DeCET to breast cancer. Includes code for generating Figures 7A, 7B and 8D.
    * [DeCET_PCa_final.ipynb](./DeCET_downstream_analysis/DeCET_PCa_final.ipynb) application of DeCET to prostate cancer. Includes code for generating Figure 7C.
    * [auxiliary_bed_functions.py](./DeCET_downstream_analysis/auxiliary_bed_functions.py) some useful functions for working with bed formatted (or bed like) files. These are used in the Jupyter notebooks in this study.

* [DeCET_REMC_demo](./DeCET_REMC_demo) A demonstration of applying DeCET to a small dataset from the Roadmap Epigenomics Consortium (http://www.roadmapepigenomics.org).
    * [DeCET_REMC_demo](./DeCET_REMC_demo/DeCET_REMC_demo.ipynb) a Jupyter notebook demonstrating how to load and plot the projections obtained from the demo dataset. Also demonstrates how to identify genomic locations from the genomic location vectors.
    * [DeCET_REMC_demo](./DeCET_REMC_demo/DeCET_REMC_demo_input.txt) a text file specifying the input for the demo dataset.
    * [DeCET_REMC_demo](./DeCET_REMC_demo/DeCET_REMC_demo_HOSVD_projections.pt) the projections obtained from the demo dataset stored as a PyTorch tensor.
    * [DeCET_REMC_demo](./DeCET_REMC_demo/DeCET_REMC_demo_HOSVD_projections.txt) the projections obtained from the demo dataset stored as a text file.

## Dependencies

All Python scripts were run on a Linux cluster, Python version 3.6.1. The Jupyter notebooks were run on macOS: Catalina (10.15.6), Python version 3.7.3. 

### Core script dependencies

The Python dependencies and versions used for the [DeCET_core_scripts](./DeCET_core_scripts) are:
- numpy (v1.18.5)
- PyTorch (v0.4.0)
- scikit-learn (v0.21.2)
- TensorLy (v0.4.2)

These dependencies are also required for [classification_cross_validation.py](./DeCET_downstream_analysis/classification_cross_validation.py).

### Downstream analysis dependencies

Each Jupyter notebook in [DeCET_downstream_analysis](./DeCET_downstream_analysis) has it's own set of dependencies. The following dependencies are required for at least one of the notebooks (the versions the code was tested with are also provided):
- numpy (v1.16.4)
- PyTorch (v1.3.0)
- scikit-learn (v0.21.2)
- TensorLy (v0.4.3)
- scipy (v1.3.0)
- matplotlib (v3.1.0)
- seaborn (v0.9.0)
- pandas (v0.24.2)
- PyWavelets (v1.0.3)

## Description of core scripts
The initial and core components of DeCET are accomplished with the [DeCET_core_scripts](./DeCET_core_scripts). The two core steps of DeCET are to first construct a quantitative description of a heterogenous ChIP-seq dataset as a PyTorch tensor object and then to decompose this tensor to obtain interpretable features that can be used for biological discovery and prediction. These two steps are accomplished with [DeCET_get_data_tensor.py](./DeCET_core_scripts/DeCET_get_data_tensor.py) and [DeCET_HOSVD.py](./DeCET_core_scripts/DeCET_HOSVD.py), respectively. A description of how each of these scripts works, as well as the application to a mock example dataset can be found below. Information regarding how the HOSVD results are analyzed can be found in the downstream analysis scripts and notebooks, [DeCET_downstream_analysis](./DeCET_downstream_analysis), and the manuscript.
   
### Obtaining the data tensor
The first step of DeCET is to construct a data tensor from a heterogeneous ChIP-seq dataset. DeCET takes as input for each sample in the dataset a file storing aligned single-end reads in bed format. Each sample can optionally be matched with a corresponding control or input file that will be used to remove potential biases resulting from library preparation or other artifacts in the ChIP-seq signal. A PyTorch tensor is obtained from the provided aligned read files using [DeCET_get_data_tensor.py](./DeCET_core_scripts/DeCET_get_data_tensor.py). This script requires two arguments:
  * `-i/--file_list`: A text file specifying the ChIP-seq signal and control files containing aligned reads in bed format and the corresponding indices used to fill the data tensor. The input bed files do not need to be sorted. This should be a white space delimited file with the first entry specifying the ChIP-seq signal file, the second entry the ChIP-seq control file, and then the white space separated index of the subtensor that will store the binned signal for this sample. The second entry can be replaced with a `.` if no control file is available. In this case the ChIP-seq signal reads will be binned and the result returned with no subtraction of matched control.
  * `-o/--output`: Prefix to use for writing the output files.
   
Three additional arguments can be specified:
  *  `-g/--genome`: The genome to use for binning. This should match the genome the data was aligned to or specify a subset of the chromosomes for this genome. Options for hg19, hg38, and mm10 are provided, as well as an option to specify an arbitrary genome. See help for more information. Default is to use the hg19 genome.
  * `-b/--binsize`: Integer value specifying the size in base pairs of the genomic bins to use for binning the genome. Default is 2000.
  * `-p/--processors`: The number of processors to use for parallelization. If a value greater than 1 is specified, then the corresponding number of input files will be binned and scaled in parallel. Default is 1.
  
Four output files are returned:
  * `.pt`: A Pytorch double tensor storing the ChIP-seq dataset with the scaled controls subtracted. The last index corresponds to the genomic location. The tensor is filled according to the indices specified in the input.
  * `_net_control_tenosr.pt`: A Pytorch tensor specifying the total number of binned reads in the corresponding control file for each of the input samples (using the same indices). This can be useful for scaling visualizations of the input bed files, but is not needed for the remaining DeCET analysis.
  * `_scale_tensor.pt`: A Pytorch tensor specifying the scaling factor used to scale the corresponding control to each of the signal datasets. This file is not needed for the remaining DeCET analysis.
  * `_index_to_genomic.pkl`: A pickled Python list. This list contains the genomic coordinates (in 0-start, half-open format) corresponding to the genomic location index used in the tensor. For each index, the corresponding chromosome, start coordinate, and end coordinate of the bin is stored as a list (_e.g._,  `['chr1', 0, 2000]`).
  
### Decomposing the data tensor
[DeCET_HOSVD.py](./DeCET_core_scripts/DeCET_HOSVD.py) performs the HOSVD of the epigenetic data tensor. There are two required arguments:
   * `-t/--data_tensor`: The PyTorch tensor storing the ChIP-seq dataset. The tensor can be of any shape, but the last index should specify the genomic location.
   * `-o/--output`: Prefix to use for writing the output files.
   
Three additional arguments can be specified:
   * `--loc_cut`: An integer specifying the number of location vectors to return. This can be used to reduce the size of the output files. Default is the product of the dimensions of all but the last index spaces.
   * `--scale_function`: A method for scaling the input data tensor. If this argument is specified with no value then L1 scaling is performed along the last index (this corresponds to scaling each sample by the number of binned reads). Alternatively the name of a module can be specified. In this case the module should specify a function named `scale_function` that takes a PyTorch tensor as input and returns a PyTorch tensor of the same shape. By default the HOSVD will be applied directly to the input tensor.
   * `--cpu`: Run the HOSVD on CPU. Default is to run on GPU.

[DeCET_HOSVD.py](./DeCET_core_scripts/DeCET_HOSVD.py) returns three output files plus an additional output file for each index in the input tensor (_e.g._ for an order 4 tensor 7 output files will be returned):
   * `_projections.pt`: A PyTorch double tensor storing the projections onto the genomic location vectors.
   * `_projections.txt`: A text file storing the projections onto the genomic location vectors. This file contains the same information as `_projections.pt`. For each line in the file the comma separated values before the `:` specify the sample index, while the comma separated values after the `:` specify the projections onto the location vectors (in that order) for the corresponding sample. 
   * `_core_tensor.pt`: A PyTorch tensor. This is the core tensor obtained with the HOSVD.
   * `_factor_matrix_i.pt`: An order 2 PyTorch tensor. This is the left singular vector matrix of the ith index space (0-based). The columns of this matrix are the left singular vectors of the tensor unfolded along the corresponding mode and sorted by descending singular value. The largest i returned corresponds to the genomic location space. 
   
DeCET computes a truncated HOSVD that only returns the singular vectors for which the corresponding subtensor of the core tensor has non-zero entries. This is analogous to the truncated SVD for matrices. The truncated core tensor and factor matrices are sufficient to reconstruct the input tensor. For most genomic datasets all but the last factor matrix will be square (no truncation). For the genomic location space only genomic location vectors with non-trivial projections will be returned. The number of columns in this matrix should be no more than the product of the dimensions of the other three index spaces. This number can be reduced with the `--loc_cut` argument to reduce the output file sizes.

## Demonstration

### Demonstration for an order 3 tensor
We demonstrate applying DeCET to obtain the HOSVD of a small sample dataset from Roadmap Epigenomics Consortium (REMC). Aligned ChIP-sequencing reads were downloaded from the REMC website (http://www.roadmapepigenomics.org) for three consolidated samples (E067, E069, E071) for five histone modifications and matched input (H3K27ac, H3K27me3, H3K9me3, H3K4me3, H3K4me1, Input). To reproduce the demonstration, the aligned files should be placed in the DeCET_REMC_demo directory and the commands given below should be run from this directory. The example dataset can be downloaded with wget, e.g.

`wget https://egg2.wustl.edu/roadmap/data/byFileType/alignments/consolidated/E067-H3K27ac.tagAlign.gz`

The aligned read files must be unzipped before inputting to [DeCET_get_data_tensor.py](./DeCET_core_scripts/DeCET_get_data_tensor.py), e.g.

`gunzip E067-H3K27ac.tagAlign.gz`

Some additional filtering of reads aligning to repeat regions may be necessary to remove artifacts resulting from alignment bias. This step is not, however, necessary to obtain the data tensor and is skipped for this demonstration. To construct the tensor, instructions must be provided to match each sample to a corresponding tensor index. This information is provided with a white space delimited text file. For the demo dataset we name this file `DeCET_REMC_demo_input.txt` and its contents are:
```
E067-H3K27ac.tagAlign        E067-Input.tagAlign  0       0
E067-H3K27me3.tagAlign       E067-Input.tagAlign  0       1
E067-H3K9me3.tagAlign        E067-Input.tagAlign  0       2
E067-H3K4me3.tagAlign        E067-Input.tagAlign  0       3
E067-H3K4me1.tagAlign        E067-Input.tagAlign  0       4
E069-H3K27ac.tagAlign        E069-Input.tagAlign  1       0
E069-H3K27me3.tagAlign       E069-Input.tagAlign  1       1
E069-H3K9me3.tagAlign        E069-Input.tagAlign  1       2
E069-H3K4me3.tagAlign        E069-Input.tagAlign  1       3
E069-H3K4me1.tagAlign        E069-Input.tagAlign  1       4
E071-H3K27ac.tagAlign        E071-Input.tagAlign  2       0
E071-H3K27me3.tagAlign       E071-Input.tagAlign  2       1
E071-H3K9me3.tagAlign        E071-Input.tagAlign  2       2
E071-H3K4me3.tagAlign        E071-Input.tagAlign  2       3
E071-H3K4me1.tagAlign        E071-Input.tagAlign  2       4
```

The input datasets are used to remove a background read distribution from the matched signal dataset. A PyTorch tensor for the demo dataset using the hg19 genome and 2000bp bins is obtained by running
```
python ../DeCET_core_scripts/DeCET_get_data_tensor.py -i "DeCET_REMC_demo_input.txt" -o "DeCET_REMC_demo"
```  
The script took about 55 minutes to complete. If additional processors are available, this time can be reduced. For example with 15 processors,
```
python ../DeCET_core_scripts/DeCET_get_data_tensor.py -p 15 -i "DeCET_REMC_demo_input.txt" -o "DeCET_REMC_demo"
``` 
took about 7 minutes to complete. The parallelization bins the reads for each file separately, so providing more processors will not further reduce the time. Four outputs are returned in the DeCET_REMC_demo directory:
  * `DeCET_REMC_demo.pt`: A PyTorch double tensor of shape `(3, 5, 1547874)` storing the binned signal reads with the scaled binned controls subtracted. The first two indices specify the sample and assay, while the third index specifies the genomic location. The samples are ordered according to the indices provided. For example, the binned reads for sample `E069-H3K4me3.tagAlign` with the corresponding control subtracted is stored in the `[1,3,:]` index of the PyTorch tensor.
  * `DeCET_REMC_demo_net_control_tenosr.pt`: A PyTorch tensor of shape `(3, 5)` storing the number of binned reads in the corresponding control files.
  * `DeCET_REMC_demo_scale_tensor.pt`: A PyTorch tensor of shape `(3, 5)` storing the scaling factor for the corresponding signal and control files.
  * `DeCET_REMC_demo_index_to_genomic.pkl`: A pickled Python list of length 1547874 with the genomic coordinates of the hg19 genome for the 2kb bins corresponding to the location vector indices in `DeCET_REMC_demo.pt`.

The HOSVD with the MOR scaling of the demo data tensor is obtained by running the [DeCET_HOSVD.py](./DeCET_core_scripts/DeCET_HOSVD.py) with the following command:
```
python ../DeCET_core_scripts/DeCET_HOSVD.py -t "DeCET_REMC_demo.pt" --scale_function "MOR_order3_scaling_ChIPseq" -o "DeCET_REMC_demo_HOSVD"
```
The script completed in about 4 minutes. With the provided arguments this script requires a gpu processor. The script can also be run using a cpu processor by adding the `--cpu` option:
```
python ../DeCET_core_scripts/DeCET_HOSVD.py --cpu -t "DeCET_REMC_demo.pt" --scale_function "MOR_order3_scaling_ChIPseq" -o "DeCET_REMC_demo_HOSVD"
```
When run using a cpu processor the script completed in about 13 minutes. Note that the `--scale_function` argument will search the working directory for a Python script `MOR_order3_scaling_ChIPseq.py`, and hence this script must be in the same directory as [DeCET_HOSVD.py](./DeCET_core_scripts/DeCET_HOSVD.py). The `MOR_order3_scaling_ChIPseq.py` scaling approach assumes an order 3 tensor with the three indices specifying the sample, assay, and location in that order (see description above). The script produces six output files:
   * `DeCET_REMC_demo_HOSVD_projections.pt`: A PyTorch tensor of shape (3, 5, 15) specifying the projections of each dataset onto the 15 genomic location vectors.
   * `DeCET_REMC_demo_HOSVD_projections.txt`: A text file specifying the projections of each dataset onto the 15 genomic location vectors.
   * `DeCET_REMC_demo_HOSVD_core_tensor.pt`: A PyTorch tensor of shape (3, 5, 15). This is a subtensor of the core tensor obtained with the HOSVD.
   * `DeCET_REMC_demo_HOSVD_factor_matrix_0.pt`: A PyTorch tensor of shape (3, 3) with columns corresponding to the left singular vectors of the data tensor unfolded along the sample index. 
   * `DeCET_REMC_demo_HOSVD_factor_matrix_1.pt`: A PyTorch tensor of shape (5, 5) with columns corresponding to the left singular vectors of the data tensor unfolded along the assay index. 
   * `DeCET_REMC_demo_HOSVD_factor_matrix_2.pt`: A PyTorch tensor of shape (1547874, 15) with columns corresponding to the left singular vectors of the data tensor unfolded along the genomic location index. The genomic coordinates of the 2kb bins corresponding to the row indices can be obtained from the pickled Python list `DeCET_REMC_demo_index_to_genomic.pkl`.

A demonstration of loading and plotting the HOSVD projections can be found in the jupyter notebook [DeCET_REMC_demo.ipynb](./DeCET_REMC_demo/DeCET_REMC_demo.ipynb). This notebook also provides a demonstration of obtaining the genomic coordinates for location vector bins.

### Demonstration for an order 4 tensor
Below is an example input, `example_input.txt`, for a mock heterogenous histone modification dataset consisting of matched normal and tumor tissue from two patients for three histone modifications (H3K27ac, H3K4me3, H3K4me1).
  ```
  normal_pt1_H3K27ac.bed   normal_pt1_input.bed 0  0  0
  normal_pt1_H3K4me3.bed   normal_pt1_input.bed 0  0  1
  normal_pt1_H3K4me1.bed   normal_pt1_input.bed 0  0  2
  normal_pt2_H3K27ac.bed   normal_pt2_input.bed 0  1  0
  normal_pt2_H3K4me3.bed   normal_pt2_input.bed 0  1  1
  normal_pt2_H3K4me1.bed   normal_pt2_input.bed 0  1  2
  tumor_pt1_H3K27ac.bed tumor_pt1_input.bed  1  0  0
  tumor_pt1_H3K4me3.bed tumor_pt1_input.bed  1  0  1
  tumor_pt1_H3K4me1.bed tumor_pt1_input.bed  1  0  2
  tumor_pt2_H3K27ac.bed tumor_pt2_input.bed  1  1  0
  tumor_pt2_H3K4me3.bed tumor_pt2_input.bed  1  1  1
  tumor_pt2_H3K4me1.bed tumor_pt2_input.bed  1  1  2
  ```
  To obtain a PyTorch tensor for this dataset using the hg19 genome and 2000bp bins, run:
  ```
  python DeCET_get_data_tensor.py -i "example_input.txt" -o "example_output"
  ```
  This produces four output files:
  * `example_output.pt`: A PyTorch double tensor of shape `(2, 2, 3, 1547874)` storing the binned signal reads with the scaled binned controls subtracted. The first three indices specify the condition, patient and assay, while the fourth index specifies the genomic location. The samples are ordered according to the indices provided. For example, the binned reads for sample `normal_pt1_H3K4me3.bed` with the corresponding control subtracted is stored in the `[0,0,1,:]` index of the PyTorch tensor.
  * `example_output_net_control_tenosr.pt`: A PyTorch tensor of shape `(2, 2, 3)` storing the number of binned reads in the corresponding control files.
  * `example_output_scale_tensor.pt`: A PyTorch tensor of shape `(2, 2, 3)` storing the scaling factor for the corresponding signal and control files.
  * `example_output_index_to_genomic.pkl`: A pickled Python list of length 1547874 with the genomic coordinates of the hg19 genome for the 2kb bins corresponding to the location vector indices in `example_output.pt`.
  
The HOSVD of the median of ratios scaled data tensor can be obtained with the following command: 
```
python DeCET_HOSVD.py -t "example_output.pt" --scale_function "MOR_scaling_ChIPseq" -o "example_output_HOSVD"
```
Note that the `--scale_function` argument will search the working directory for a Python script `MOR_scaling_ChIPseq.py`, and hence this script must be in the same directory as [DeCET_HOSVD.py](./DeCET_core_scripts/DeCET_HOSVD.py). The `MOR_scaling_ChIPseq.py` scaling approach assumes an order 4 tensor with the first two indices specifying the condition and patient (see description above). As such, this scaling function is not generalizable to arbitrary datasets or ordering of indices.

Running this code will return seven files:
   * `example_output_HOSVD_projections.pt`: A PyTorch tensor of shape (2, 2, 3, 12) specifying the projections of each dataset onto the 12 genomic location vectors.
   * `example_output_HOSVD_projections.txt`: A text file specifying the projections of each dataset onto the 12 genomic location vectors.
   * `example_output_HOSVD_core_tensor.pt`: A PyTorch tensor of shape (2, 2, 3, 12). This is a subtensor of the core tensor obtained with the HOSVD.
   * `example_output_HOSVD_factor_matrix_0.pt`: A PyTorch tensor of shape (2, 2) with columns corresponding to the left singular vectors of the data tensor unfolded along the condition index. 
   * `example_output_HOSVD_factor_matrix_1.pt`: A PyTorch tensor of shape (2, 2) with columns corresponding to the left singular vectors of the data tensor unfolded along the patient index. 
   * `example_output_HOSVD_factor_matrix_2.pt`: A PyTorch tensor of shape (3, 3) with columns corresponding to the left singular vectors of the data tensor unfolded along the assay index. 
   * `example_output_HOSVD_factor_matrix_3.pt`: A PyTorch tensor of shape (1547874, 12) with columns corresponding to the left singular vectors of the data tensor unfolded along the genomic location index. The genomic coordinates of the 2kb bins corresponding to the row indices can be obtained from the pickled Python list `example_output_index_to_genomic.pkl
