# cnnGWAS
This is the repository for cnnGWAS, which is a method that captures complex biological patterns shared by GWAS risk variants. By utilizing convolutional neural network, the method prioritizes functional variants out of all candidate variants in a GWAS loci. 


## Environment setting
###Installing miniconda:  
We recommend using miniconda as a package manager when executing cnnGWAS. Miniconda can be installed from the following link: https://docs.conda.io/en/latest/miniconda.html#


###Environment test:   
All software tests were done under the following environment setting, and therefore we *strongly* recommend to set the miniconda environment via following commands: 
```
conda create --name cnnGWAS python=2.7
conda activate cnnGWAS
conda install theano=1.0.3 numpy=1.15.4 matplotlib=2.2.3 pandas=0.24.2 bedtools=2.28.0 bedops=2.4.36 mkl=2019.4
```


###Required R packages (R version 3.5.2):
```R
library(ggplot2)
library(gplots)
library(grid)
library(gridExtra)
library(plyr)
library(reshape2)
```


###Required tools
```
awk (GNU Awk 3.1.7)
wget (GNU Wget 1.12 built on linux-gnu)
```


###Tools that will be installed
```
impG-summary (http://bogdan.bioinformatics.ucla.edu/software/impg/)
```


## Epigenetic feature DB preparation  

Users can download the required files by executing the following commands:

```      
cd DB
bash Download_DB.sh 
cd ..
```


## Input file preparation   
###Input file format: 
Input summary statistics file should be located under directory "00_RawDatas/". 
Input file **must** be a tab delimited text file with 6 columns. 
Each of the columns should have a header row named as follows:

```
snpid  chr     bp      a1      a2      zscore
rs3131972       chr1    752721  A       G       0.618384984132729
rs3131969       chr1    754182  A       G       1.20540840742781
```

All coordinates are based on hg19 (GRCh37). Raw GWAS summary statistics files can include different information; in most cases, z-score can be calculated from the given information. Example input file "AUTsummary.hg19.txt" is provided. 

###Input file name: 
We recommend simple file prefix, preferably disease name (e.g. SLE), ALWAYS followed by ".txt" extension. Underbar "_" MUST NOT be included inside a file prefix(e.g. PSY_AUT_dis.txt), since it will cause string processing error. [Example file names: "SLE.txt", "MDD2019.txt"]


## 1. Imputation of association p-value using ImpG-summary

> Following command executes the imputation process. More summary statistics file can be given as a input (e.g. bash RUN.sh FILE1.txt FILE2.txt FILE3.txt).

```
cd 01_RunImpG
bash RUN.sh AUTsummary.hg19.txt   
cd ..
```

## 2. Make input pickled files for CNN model training

> In this stage, the input files (*.pkl.gz) for training are made.

```
cd 02_PreProc
bash RUN.sh AUTsummary.hg19.txt 
cd ..
```

## 3. Model training

> In this stage, the training itself take place. As an example, a pre-produced sample file is provided (02_PreProc/Data/AUTsummary.hg19_AE_30SNP.pkl.gz).

```
cd 03_RunCNN
bash RUN.sh AUTsummary.hg19.txt 
cd ..
```

## 4. Evaluation of the model performance

> The performance of the model is saved as a pdf file. Also, the scores of candidate SNPs are calculated and given as a text file. SNPs with causal score > 0.5 are classified as positive, and are putative causal variants. 

```
cd 04_Performance
bash RUN.sh AUTsummary.hg19.txt
cd ..
```

## Notes
> Each folder contains "NOTE" or "RUN_NOTE.txt". Please read the contents before executing the software. 


