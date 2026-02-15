# GitHub Setup Guide

## Current Status

✅ Git repository initialized
✅ .gitignore configured (excludes data files)
✅ README.md created
✅ config_template.py created for path configuration
✅ Main logistic regression script updated to use config
✅ Files staged for commit

## Next Steps

### 1. Configure Git Identity (ONE TIME SETUP)

You need to tell Git who you are. Choose ONE of the options below:

**Option A: Set globally (recommended, applies to all repositories)**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Option B: Set for this repository only**
```bash
cd /home/keh4016/ke_storage/sex_difference_github/HCP
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

Use the same email that's associated with your GitHub account (https://github.com/kehuang011).

### 2. Make Initial Commit

```bash
cd /home/keh4016/ke_storage/sex_difference_github/HCP
git commit -m "Initial commit: HCP sex difference analysis code"
```

### 3. Create GitHub Repository

**Option A: Using GitHub CLI (gh)**
```bash
# Install gh if not already installed
# Then:
gh auth login  # Follow prompts to authenticate
gh repo create HCP-sex-difference --public --source=. --remote=origin --push
```

**Option B: Using GitHub Web Interface**

1. Go to: https://github.com/new
2. Repository name: `HCP-sex-difference` (or your preferred name)
3. Description: "Analysis of sex differences in HCP connectome data"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### 4. Connect Local Repository to GitHub

After creating the repository on GitHub, connect it:

```bash
cd /home/keh4016/ke_storage/sex_difference_github/HCP
git remote add origin https://github.com/kehuang011/YOUR-REPO-NAME.git
git branch -M main
git push -u origin main
```

**For SSH (if you have SSH keys set up):**
```bash
git remote add origin git@github.com:kehuang011/YOUR-REPO-NAME.git
git branch -M main
git push -u origin main
```

### 5. Verify Upload

Go to your repository URL:
```
https://github.com/kehuang011/YOUR-REPO-NAME
```

You should see:
- README.md displayed
- All your Python scripts and notebooks
- NO data files (*.mat, *.csv)
- NO output directories

## Important Notes

### What IS uploaded:
✅ Python scripts (.py files)
✅ Jupyter notebooks (.ipynb files)
✅ README.md and documentation
✅ config_template.py (template only, not your actual paths)
✅ .gitignore file

### What is NOT uploaded (and should not be):
❌ Data files (*.mat, *.csv)
❌ Output directories (out_log/, results/)
❌ config.py (contains your local paths)
❌ Large generated files

## Future Updates

To update your repository after making changes:

```bash
cd /home/keh4016/ke_storage/sex_difference_github/HCP

# Check what changed
git status

# Stage changes
git add .

# Commit with message
git commit -m "Description of your changes"

# Push to GitHub
git push
```

## Updating Other Python Scripts

The main logistic regression script has been updated to use `config.py`. You should update other scripts similarly:

1. Add the config import at the top
2. Replace hardcoded paths with config variables
3. Test locally before committing

## Data Sharing Options

Since data cannot go on GitHub, consider:

1. **Institutional Repository**: Upload to your university's data repository
2. **Zenodo**: Free, gets you a DOI, integrates with GitHub
3. **Open Science Framework (OSF)**: Free, designed for research
4. **Dryad**: For published papers
5. **HCP Website**: If redistribution is allowed
6. **Supplement Material**: With journal publication

In your README, include:
- Description of data requirements
- Instructions for obtaining HCP data
- Link to where you've uploaded processed data (if applicable)
- Contact information for data access requests

## Troubleshooting

### Permission denied (publickey)
Set up SSH keys or use HTTPS with personal access token.

### Large files rejected
GitHub has 100MB file size limit. Use .gitignore to exclude them.

### Accidentally committed sensitive data
Use `git filter-branch` or BFG Repo-Cleaner to remove from history.

## Questions?

- GitHub Docs: https://docs.github.com
- Git Basics: https://git-scm.com/book/en/v2
- GitHub CLI: https://cli.github.com/
