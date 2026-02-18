# HCP Sex Difference Analysis

This repository contains code for analyzing sex differences in the Human Connectome Project (HCP) data using logistic regression and deep learning approaches.

## Project Structure

```
HCP/
├── logistic_regression/          # Logistic regression analyses
├── sensitivity_analysis_sex_site/ # Sensitivity analyses
├── *.ipynb                        # Analysis notebooks
├── *.py                           # Python scripts for batch processing
└── config_template.py             # Configuration template (copy and modify)
```

## Data Requirements

This code expects HCP data in the following structure:

### Required Data Files (NOT included in this repository):

1. **Subject information CSV**: 
   - File: `ls3_subjinfo_agesexcog_details_1419subj.csv`
   - Contains: age, sex, site
   
2. **Connectivity matrices**: 
   - Format: `.mat` files with connectivity matrices
   - Location: Configure in `config.py`
   - Types: Various connectome types (flavor-based)

3. **Data splits** (for split0328 option):
   - File: `ls3_subject_splits_1419subj_403train_45val_838test_1286balanced.mat`
   - Contains: train/validation/test subject indices

### Data Access

- HCP data: Available through [Human Connectome Project](https://www.humanconnectome.org/)
- Processed data: Contact the authors or refer to the paper

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

## Usage (to be updated)

### Logistic Regression Analysis

```bash
cd logistic_regression
python raw_logistic_ensemble_covbat.py \
    --bins 8 15 22 29 36 50 65 80 100 \
    --contype flavor1 \
    --outer_folds 100 \
    --inner_folds 5 \
    --add_trainingYA newYA_train \
    --split0328 \
    --testsize 15
```

### Arguments:
- `--bins`: Age bins for analysis
- `--contype`: Connectome type/flavor
- `--outer_folds`: Number of cross-validation folds (default: 100)
- `--inner_folds`: Inner CV folds (default: 5)
- `--add_trainingYA`: Add training data from fine-tuned Krakencoder
- `--split0328`: Use specific data split
- `--testsize`: Test set size
- `--downsample`: Enable downsampling
- `--train_downsample_size`: Training downsample size

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

