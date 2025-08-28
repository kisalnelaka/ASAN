# ASAN: Adaptive Symmetry-Aware Network for 3D Molecular Prediction

This repository contains the implementation of the Adaptive Symmetry-Aware Network (ASAN), a novel deep learning model designed to predict HOMO-LUMO gaps from 3D molecular structures with enhanced symmetry awareness. ASAN leverages adaptive symmetry learning to outperform traditional invariant models on small and noisy datasets, making it suitable for quantum chemistry applications.

## Overview

ASAN addresses the challenge of predicting molecular properties under random 3D rotations by incorporating learnable symmetry parameters. Compared to baseline models and MIT-Inspired approaches, ASAN offers improved generalization, as demonstrated on a subset of the QM9 dataset. This project includes data generation scripts, model implementations, training pipelines, and evaluation tools.

## Features
- Adaptive symmetry learning for robust 3D molecular feature extraction.
- Support for datasets with up to 15 atoms per molecule (expandable to 29 for full QM9).
- Cross-validation with 5 folds for robust performance metrics.
- Visualization of training losses and test predictions.

## Installation

### Prerequisites
- Python 3.13
- PyTorch
- RDKit
- NumPy
- SciPy
- Pandas

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/asan.git
   cd asan
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Ensure RDKit is installed (e.g., via `pip install rdkit-pypi`).

## Usage

### Data Generation
Generate 3D molecular data from `qm9.csv`:
```bash
python src/data_generator.py
```
This produces `data.pt` with 800 training and 200 test samples.

### Training
Train all models (Baseline, MIT-Inspired, ASAN) with 5-fold cross-validation:
```bash
python src/train.py
```
Check `results/poc_results.txt` for validation losses.

### Evaluation
Evaluate trained models on the test set:
```bash
python src/evaluate.py > results/poc_results.txt
```

### Visualization
Plot training losses:
```bash
python src/plot_losses.py
```

### Testing
Run unit tests:
```bash
python tests/test_models.py
```

## Results
- **Baseline Average MAE**: 0.0527 eV
- **MIT-Inspired Average MAE**: 0.0468 eV
- **ASAN Average MAE**: 0.0460 eV
- ASAN shows the best test performance, with potential for further improvement on larger datasets.

## Directory Structure
```
asan/
├── data/              # Generated data files (e.g., data.pt)
├── src/               # Source code
│   ├── data_generator.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── plot_losses.py
├── tests/             # Unit tests
│   └── test_models.py
├── results/           # Output files (e.g., poc_results.txt)
├── notebooks/         # Jupyter notebooks for POC
│   └── poc_notebook.ipynb
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Contributing
We welcome contributions! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "description"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## Commercial License
This software and its associated files are provided under a commercial license. Unauthorized use, distribution, or modification is strictly prohibited. For licensing inquiries, please contact the author.

## Acknowledgments
- Inspired by research on 3D molecular modeling and quantum chemistry datasets.
- Thanks to the QM9 dataset contributors.

## IF YOU HAVE THE ABILITY TO TEST WITH A LARGER DATASET, PLEASE TEST IT AND SHARE THE RESULTS WITH ME AT kisalnelaka6@gmail.com .