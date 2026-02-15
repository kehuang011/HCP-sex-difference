"""
Configuration template for HCP analysis paths.

INSTRUCTIONS:
1. Copy this file to 'config.py': cp config_template.py config.py
2. Edit config.py with your local paths
3. DO NOT commit config.py to git (it's in .gitignore)
"""

# Main data directories
DATA_DIR = "/path/to/your/data/directory"  # e.g., "/midtier/cocolab/scratch/kwj2001/LS3" or "/midtier/cocolab/scratch/kwj2001/LS3/rawdata_dynseed"
SUBJECT_INFO_CSV = "/path/to/ls3_subjinfo_agesexcog_details_1419subj.csv"
DATA_SPLITS_MAT = "/path/to/ls3_subject_splits_1419subj_403train_45val_838test_1286balanced.mat"

# HCP Young Adult (YA) data - for krakencoder models
HCP_DATA_DIR = "./HCP_data"  # Directory containing HCP YA krakencoder data
HCP_YA_SUBJINFO_CSV = "./HCP_data/hcp_subject_info_1206subj_age_sex_cog.csv"
HCP_YA_SPLITS_MAT = "./HCP_data/HCP_krakencoder/subject_splits_993subj_683train_79val_196test_retestInTest.mat"
HCP_YA_ENCODED_MAT = "./HCP_data/HCP_krakencoder/hcp_993subj_20240413_210723_ep002000_encoded.mat"
YEO_NETWORK_MAT = "./HCP_data/fc_merged_atlasconcat_justyeo.mat"

# Krakencoder checkpoint and transform files - for sensitivity analysis
IOXFM_DIR = "/path/to/ioxfm_dynseed"  # e.g., "/midtier/cocolab/scratch/kwj2001/LS3/ioxfm_dynseed"
CHECKPOINT_SEX_FINETUNED = "/path/to/sex_finetuned_checkpoint.pt"
CHECKPOINT_SEX_SITE_FINETUNED = "/path/to/sex_site_finetuned_checkpoint.pt"

# Output directories
OUTPUT_DIR = "./outputs"  # Relative to project root
RESULTS_DIR = "./outputs/results"
