# HCP Sex Difference Analysis

This repository contains code for analyzing sex differences in the Human Connectome Project (HCP) data using Krakencoder and ensemble model (logistic regression). The analysis spans multiple age ranges and examines sex prediction accuracy, site effects, and feature importance across various brain connectivity modalities.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/kehuang011/HCP-sex-difference.git
cd HCP-sex-difference
pip install -r requirements.txt

# 2. Configure paths
cp config_template.py config.py
# Edit config.py with your data paths

# 3. Run analysis (see Detailed Usage Examples below)
```

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Configuration](#configuration)
- [Instructions for Use](#instructions-for-use)
- [Detailed Usage Examples](#detailed-usage-examples)
- [File Descriptions](#file-descriptions)
- [Citation](#citation)
- [Contact](#contact)

## Project Structure

```
HCP-sex-difference/
├── logistic_regression/
│   ├── raw_logistic_ensemble.py
│   ├── raw_logistic_ensemble_site.py
│   ├── raw_logistic_ensemble_covbat_site.py
│   ├── raw_wholecohort_logistic_ensemble.py
│   └── covbat_modified.py
├── sensitivity_analysis_sex_site/
│   ├── sensitivity_kraken_nwlevel.py
│   ├── sensitivity_kraken_networkpair.py
│   ├── Krakencoder_demo_finetuning.sh
│   └── [other Krakencoder training utilities]
├── *.ipynb                    # Jupyter notebooks for visualization
├── *.py                       # Main analysis scripts
├── requirements.txt           # Python dependencies
├── config_template.py         # Configuration template
└── README.md                  # This file
```

**Key Directories:**
- **`logistic_regression/`** - Ensemble logistic regression analyses on raw connectomes
- **`sensitivity_analysis_sex_site/`** - Krakencoder sensitivity analyses and fine-tuning

## Requirements

### Software Dependencies

This project requires Python >= 3.8. All required Python packages are listed in `requirements.txt`.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kehuang011/HCP-sex-difference.git
   cd HCP-sex-difference
   ```

2. **Create a conda environment** (recommended):
   ```bash
   conda create -n sex_diff python=3.8.20
   conda activate sex_diff
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure paths** (see Configuration section below):
   ```bash
   cp config_template.py config.py
   # Edit config.py with your local paths
   ```

**Typical install time**: 5-10 minutes on a standard desktop computer with a stable internet connection.

## Data Requirements

This code expects HCP data in the following structure:

### Required Data Files:

1. **Subject information CSV**: 
   - File: `ls3_subjinfo_agesexcog_details_1419subj.csv`
   - Contains: age, sex, site
   
2. **Connectivity matrices**: 
   - Format: `.mat` files with connectivity matrices
   - Location: Configure in `config.py`
   - Types: Various connectome types (flavor-based)

3. **Data splits**:
   - File: `ls3_subject_splits_1419subj_403train_45val_838test_1286balanced.mat`
   - Contains: train/validation/test subject indices


### Data Access

- HCP data: Available through [Human Connectome Project](https://www.humanconnectome.org/)
- Post-processed connectome data: Contact the authors or refer to the paper

## Configuration

1. Copy `config_template.py` to `config.py`:
   ```bash
   cp config_template.py config.py
   ```

2. Edit `config.py` to set your local data paths:
   ```python
   DATA_DIR = "/path/to/your/data"
   OUTPUT_DIR = "/path/to/your/outputs"
   ```

## Instructions for Use

### Overview

This repository provides tools for analyzing sex differences in brain connectivity across the lifespan. There are two main analysis pipelines:

1. **Raw connectome analysis**: Ensemble model (logistic regression) on raw connectivity matrices
2. **Krakencoder-based analysis**: Sex/site prediction using learned low-dimensional representations

### Running on Your Own Data

#### Step 1: Prepare Your Data

Your data should include:
- **Connectivity matrices**: Structural (SC) or functional (FC) connectivity in `.mat` format
- **Subject metadata**: CSV file with columns for subject ID, age, sex, and site
- **Data splits**: Pre-defined train/validation/test splits in `.mat` format

#### Step 2: Configure Paths

Edit `config.py` to point to your data:
```python
DATA_DIR = "/path/to/your/connectivity/matrices"
SUBJECT_INFO_CSV = "/path/to/subject_info.csv"
RESULTS_DIR = "/path/to/save/results"
```

#### Step 3: Run Analysis

**Option A: Krakencoder-based Analysis** (requires pre-trained Krakencoder model)

```bash
python krakencoder_sex.py \
    --bins 8 11 14 18 22 36 45 55 65 80 101 \
    --dataname 'your_krakencoder_encoded_data' \
    --split0328 \
    --testsize 30 \
    --add_trainingYA 'newYA_train' \
    --downsample \
    --train_downsample_size 15 \
    --method 'logistic'
```

**Option B: Raw Connectome Analysis** (works with raw connectivity data)

```bash
cd logistic_regression
python raw_logistic_ensemble.py \
    --bins 8 15 22 29 36 50 65 80 101 \
    --contype your_connectome_type \
    --outer_folds 100 \
    --inner_folds 5 \
    --add_trainingYA newYA_train \
    --split0328 \
    --testsize 30\
    --train_downsample_size 15
```

#### Step 4: Visualize Results

Open the relevant Jupyter notebook (e.g., `sex_acc_allmodels.ipynb`) and update paths to point to your results files. Run the notebook to generate figures.

### Expected Outputs

Both scripts will generate `.mat` files in `RESULTS_DIR` containing:
- **Accuracy metrics**: Per-fold accuracies, mean, standard deviation
- **Predictions**: Predicted labels and probabilities for test subjects
- **Model coefficients**: Trained weights (Haufe-transformed for interpretability)
- **Subject indices**: Train/test splits used for each fold

### Runtime Estimates

On a standard desktop computer:
- **Krakencoder-based analysis**: ~10-30 minutes 

---

## Detailed Usage Examples

Below are specific command-line examples used in the paper. Modify the parameters according to your dataset and analysis goals.

### Sex / Site Prediction
#### Based on Latent Space from Original Krakencoder / Sex Fine-tuned Krakencoder

```bash
python krakencoder_sex.py \ 
   --bins 8 11 14 18 22 36 45 55 65 80 101 \
   --dataname 'krakenLS3dynICV403demofineSEX100_20251107_144040_ep000500_encoded' \
   --split0328 \
   --testsize 30 \
   --add_trainingYA 'newYA_train' \
   --downsample \
   --train_downsample_size 15 \
   --method 'logistic' 

```

#### Raw Data Analysis (logistic regression for each flavor)
```bash
cd logistic_regression
python raw_logistic_ensemble_covbat.py \
    --bins 8 15 22 29 36 50 65 80 101 \
    --contype flavor1 \
    --outer_folds 100 \
    --inner_folds 5 \
    --downsample \
    --add_trainingYA newYA_train \
    --split0328 \
    --testsize 30\
    --train_downsample_size 15
```

#### Common Arguments:

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--bins` | list[int] | Age bin boundaries for stratification (e.g., `8 11 14 18 22 36 45 55 65 80 101`) | Required |
| `--contype` | str | Connectome type/flavor (e.g., `FCcorr_shen268_hpf`) | Required for raw analysis |
| `--dataname` | str | Name of Krakencoder encoded data file for input and output identification | Required for Krakencoder analysis |
| `--outer_folds` | int | Number of repeated cross-validation folds | 100 |
| `--inner_folds` | int | Number of inner CV folds for hyperparameter tuning | 5 |
| `--add_trainingYA` | str | Add fine-tuned Krakencoder training data: `'newYA_train'` or `'no_addtraining'` | None |
| `--split0328` | flag | Use predefined train/val/test split (403 train, 45 val, 838 test) | False |
| `--testsize` | int | Number of subjects per sex in test set (e.g., 30 = 15 F + 15 M) | 20 |
| `--downsample` | flag | Enable downsampling to balance sample sizes across age bins | False |
| `--train_downsample_size` | int | Number of subjects per sex in downsampled training set (total = 2×value) | 15 |
| `--method` | str | Classifier type: `'logistic'` or `'krr'` (kernel ridge regression) or `'mlp'` | logistic |

### Krakencoder Sensitivity Analysis

Identify which brain networks contribute most to sex prediction:

```bash
cd sensitivity_analysis_sex_site
python sensitivity_kraken_nwlevel.py \
   --bins 8 11 14 18 22 36 45 55 65 80 101 \
   --split0328 \
   --testsize 30 \
   --add_trainingYA 'newYA_train' \
   --downsample \
   --train_downsample_size 15 \
   --model_kraken 'sex_Krakencoder' 
```

**Additional argument:**
- `--model_kraken`: Specifies which Krakencoder model to use (`'sex_Krakencoder'` or `'sex_site_Krakencoder'`)

### Fine-tuning Krakencoder

To fine-tune a pre-trained Krakencoder model on sex prediction:

```bash
cd sensitivity_analysis_sex_site
batch Krakencoder_demo_finetuning.sh
```

This script fine-tunes the original Krakencoder on the sex prediction task. 
## File Descriptions

### Main Analysis Scripts

#### Krakencoder-based Sex/Site Prediction

- **`krakencoder_sex.py`**: Predicts biological sex using Krakencoder latent representations. 

- **`krakencoder_site.py`**: Performs site prediction using Krakencoder latent space. 

- **`krakencoder_site_permute.py`**: Performs site prediction with permutated site response.

- **`krakencoder_wholecohort_logistic.py`**: Whole-cohort analysis without age binning. Trains a single model across all ages to evaluate overall sex prediction performance using Krakencoder features.

#### Raw Connectome Analysis (Ensemble model)

- **`logistic_regression/raw_logistic_ensemble.py`**: Sex prediction on raw connectivity . 

- **`logistic_regression/raw_logistic_ensemble_site.py`**: Site prediction using raw connectomes. 

- **`logistic_regression/raw_logistic_ensemble_covbat_site.py`**: Site prediction after CovBat harmonization.

- **`logistic_regression/raw_wholecohort_logistic_ensemble.py`**: Whole-cohort raw connectome analysis. Evaluates sex prediction across the entire age range for each connectivity flavor.

#### Sensitivity Analysis

- **`sensitivity_analysis_sex_site/sensitivity_kraken_nwlevel.py`**: Network-level sensitivity analysis for Krakencoder. 

- **`sensitivity_analysis_sex_site/sensitivity_kraken_networkpair.py`**: Pairwise network interaction analysis. 

### Jupyter Notebooks

**Note**: Notebooks require results files generated by the Python scripts. Update file paths in the notebooks to point to your results directory before running.

#### Visualization and Model Comparison

- **`sex_acc_allmodels.ipynb`**: Comprehensive visualization of sex prediction accuracy across all models.

- **`site_acc_allmodels.ipynb`**: Site prediction accuracy visualization across all models. 

- **`compare_across_classifiers.ipynb`**: Benchmarks different classifier types (logistic regression, SVM, kernel ridge regression, mlp) on the same Krakencoder latent space. Identifies optimal classifier choices for sex prediction tasks.

#### Feature Analysis

- **`feature_importance(haufe).ipynb`**: Computes and visualizes Haufe-transformed feature importance for sex prediction models. 

- **`sensitivity_Krakencoder_sex.ipynb`**: Visualization of Krakencoder sensitivity analysis results. 

#### Statistical Analysis

- **`distribution_trajectories.ipynb`**: Data distribution, trajectories across the lifespan.

- **`wholecohort_model_comparisons.ipynb`**: Statistical comparison of whole-cohort models. 

## Citation

If you use this code, please cite:

```
Huang, Ke, et al. "The progression of sex differences in brain networks across the lifespan." bioRxiv (2026): 2026-01.

@article{huang2026progression,
  title={The progression of sex differences in brain networks across the lifespan},
  author={Huang, Ke and Jamison, Keith Wakefield and Jacobs, Emily G and Miolane, Nina and Kuceyeski, Amy},
  journal={bioRxiv},
  pages={2026--01},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Contact

For questions about the code or data access, please contact:
- Ke Huang (keh4016@med.cornell.edu)

