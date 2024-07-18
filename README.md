# PathNetDRP: A Novel Biomarker Discovery Framework Using Pathway and Protein-Protein Interaction Networks for Immune Checkpoint Inhibitor Response Prediction

This source code is the implementation code of the PathNetDRP algorithm to accurately predict ICI response.

## System Requirements

python	3.8.18

pandas	2.0.3

numpy	1.23.5

scikit-learn	1.3.0

statsmodels	0.14.0

networkx	3.1

scipy	1.10.1



## Code Execution

### Create PathNetGene Score matrix

Create PathNetGene score matrix files for each cohort.

```bash
#first, 
python 1_get_score_by_pathway.py

# If the 1_get_score_by_pathway.py process has been completed,
python 2_calculate_score.py

```

If the 2_calculate_score.py process is completed, the PathNetGene score matrix will be created in the result folder.



### Biomarker selection

Select Biomarker related to immunotherapy responsiveness for each cohort.

```bash
python 3_select_feature.py
```



### Prediction

Check LOOCV prediction performance for selected biomarkers.

```bash
python 4_LOOCV.py
```



## What is the input data format?

- This source code calculates the PathNetGene Score based on pre-treatment RNA-seq data from patients treated with ICI.
- You may use raw count data, but please ensure it is normalized (e.g., FPKM, TPM, TMM, etc.).
- The gene expression data should be a matrix with genes as rows and sample IDs as columns.
- Our experiments were conducted on cohorts with melanoma, metastatic gastric cancer, and bladder cancer. Other cancer types can be used but may not exhibit superior predictive performance.
- Please refer to the paper for the data required for the experiments. Due to storage limitations, only one cohort is provided as a sample.
- The PPI network data was downloaded from https://string-db.org/ (human version 11.0). For detailed preprocessing steps, please refer to the paper.
- For pathway data, we downloaded it from msigDB (https://www.gsea-msigdb.org/gsea/msigdb) and extracted only the Reactome data.


## ETC

If you have any issues or questions about execution, please leave a comment through the repository and we will check.