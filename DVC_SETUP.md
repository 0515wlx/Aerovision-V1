# DVC Setup Guide for AeroVision Project

This guide explains how to use DVC (Data Version Control) in the AeroVision project for managing models, data, and other large files.

## Overview

DVC allows us to:
- Version control large files (models, datasets) without storing them in Git
- Track changes to data and models over time
- Share models and data across the team
- Reproduce experiments with specific data/model versions

## Current Configuration

- **DVC Cache Directory**: `/home/wlx/dvc-cache`
- **DVC Remote**: `/home/wlx/dvc-remote` (local remote)
- **Managed Directories**:
  - `models/` - ML model files (.pt, .pth, .onnx, etc.)
  - `dvc-data/` - Training data and outputs

## Quick Start

### 1. Check DVC Status
```bash
python dvc_workflow.py status
```

### 2. Add Model Files to DVC
```bash
# Add a single model file
python dvc_workflow.py add-model models/your_model.pt

# Or use DVC directly
dvc add models/your_model.pt
git add models/your_model.pt.dvc
```

### 3. Add Data Files to DVC
```bash
# Add data files
python dvc_workflow.py add-data dvc-data/training_data/your_data.pkl

# Or use DVC directly
dvc add dvc-data/training_data/your_data.pkl
git add dvc-data/training_data/your_data.pkl.dvc
```

### 4. Push to Remote
```bash
# Push all DVC tracked files to remote
python dvc_workflow.py push

# Or use DVC directly
dvc push
```

### 5. Pull from Remote
```bash
# Pull all DVC tracked files from remote
python dvc_workflow.py pull

# Or use DVC directly
dvc pull
```

## Workflow Examples

### Training a New Model

1. Add your training data to DVC:
```bash
dvc add dvc-data/training_data/processed_data.pkl
git add dvc-data/training_data/processed_data.pkl.dvc
git commit -m "Add processed training data"
```

2. Train your model and save it to the models directory

3. Add the trained model to DVC:
```bash
dvc add models/my_new_model.pt
git add models/my_new_model.pt.dvc
git commit -m "Add trained model version 1.0"
```

4. Push everything to remote:
```bash
dvc push
git push
```

### Sharing Models with Team

1. Pull latest changes:
```bash
git pull
dvc pull
```

2. Get specific model version:
```bash
git checkout <model-version-tag>
dvc checkout
```

## Directory Structure

```
AeroVision-V1/
├── models/                           # Model files (managed by DVC)
│   ├── aircraft_classifier.pt
│   ├── airline_classifier.pt
│   ├── registration_detector.pt
│   └── .gitkeeper
├── dvc-data/                         # Data files (managed by DVC)
│   ├── training_data/
│   │   ├── annotations.json
│   │   └── processed_data.pkl
│   └── outputs/
├── .dvc/                            # DVC configuration
├── .dvcignore                       # DVC ignore file
├── dvc_workflow.py                  # Helper script
└── DVC_SETUP.md                     # This file
```

## File Types to Manage with DVC

### Models
- `.pt` - PyTorch models
- `.pth` - PyTorch state dict
- `.onnx` - ONNX models
- `.pkl` - Pickled models
- `.bin` - Binary model files
- `.safetensors` - Safe tensors format

### Data
- `.pkl` - Processed datasets
- `.json` - Large annotation files
- `.csv` - Large data files
- `.npz` - NumPy arrays
- `.h5` - HDF5 files

### Outputs
- Training checkpoints
- Evaluation results
- Generated images/videos
- Log files

## Best Practices

1. **Always commit DVC files**: After running `dvc add`, always run `git add <file>.dvc` and commit

2. **Use meaningful commit messages**: When updating models/data, use clear commit messages like "Update aircraft classifier to v2.1" or "Add new training data for winter 2024"

3. **Tag important versions**: Use git tags for important model versions
```bash
git tag -a aircraft-classifier-v2.0 -m "Aircraft classifier v2.0 with 95% accuracy"
```

4. **Document your data**: Keep a README in your data directories explaining what each file contains

5. **Use DVC pipelines**: For complex workflows, consider using DVC pipelines to track the entire ML pipeline

## Troubleshooting

### DVC files are git-ignored
If you get an error like "bad DVC file name is git-ignored", check your `.gitignore` file and make sure the directory containing the DVC file is not ignored.

### Permission denied when creating cache directory
If you get permission errors, make sure you have write access to the cache directory:
```bash
sudo chown -R $USER:$USER /path/to/dvc-cache
```

### Model files not showing up after pull
Make sure you've run both:
```bash
git pull  # Get the DVC metadata
dvc pull  # Get the actual files
```

## Advanced Usage

### Multiple Remotes
You can configure multiple remotes for different purposes:
```bash
# Add a backup remote
dvc remote add backup /mnt/backup/dvc-remote

# Push to specific remote
dvc push -r backup
```

### DVC Metrics and Plots
Track model performance metrics:
```bash
dvc metrics add metrics.json
dvc plots add training_history.png
```

### DVC Pipelines
Define reproducible ML pipelines:
```bash
dvc run -n train -d data/train.csv -o model.pkl python train.py
```

## Useful Commands

```bash
# Check DVC version
dvc --version

# Show DVC configuration
cat .dvc/config

# List all DVC tracked files
dvc list . --dvc-only

# Show file hash
dvc status

# Get file statistics
dvc get . models/aircraft_classifier.pt -o /tmp/model.pt

# Import file from external DVC repo
dvc import https://github.com/example/repo data/file.csv
```

## Integration with Training Scripts

You can integrate DVC into your training scripts to automatically track model versions:

```python
import dvc.api
import torch

# Save model with DVC tracking
def save_model(model, path):
    torch.save(model.state_dict(), path)
    # DVC will automatically track this file if it's in a managed directory

# Load model with version info
def load_model(path):
    with dvc.api.open(path) as f:
        return torch.load(f)
```

## Support

For more information:
- [DVC Documentation](https://dvc.org/doc)
- [DVC User Guide](https://dvc.org/doc/user-guide)
- Run `python dvc_workflow.py --help` for available commands