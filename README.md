# DJS NOVA - Technical AI Tasks

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Status:** All recruitment tasks completed ✅

Comprehensive solutions for DJS Nova Co-Comm recruitment tasks, featuring exploratory data analysis, machine learning, and astronomical data processing.

---

## 📋 Table of Contents

- [About](#about)
- [Repository Structure](#repository-structure)
- [Completed Tasks](#completed-tasks)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Results & Evaluation](#results--evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 About

This repository contains complete solutions for DJS Nova technical AI tasks, demonstrating expertise in:
- **Exploratory Data Analysis (EDA)** on astronomical datasets (SDSS)
- **Machine Learning** for regression and classification
- **Data visualization** with matplotlib, seaborn, and plotly
- **Feature engineering** and selection techniques
- **Unsupervised learning** for clustering celestial objects

Each task is documented with detailed Jupyter notebooks including code, visualizations, explanations, and results.

---

## 📁 Repository Structure

```
DJS_NOVA/
├── Technical(AI)/
│   ├── Task_1/                    # Astronomical object classification & EDA
│   │   ├── task1.ipynb           # Main analysis notebook
│   │   ├── task1.md              # Task description
│   │   ├── Task 1.csv            # SDSS dataset
│   │   └── Metadata.pdf          # Dataset documentation
│   └── Task_2/                    # Stellar parameter prediction
│       ├── task2.ipynb           # Regression modeling notebook
│       ├── task2.md              # Task requirements
│       └── task2.csv             # Stellar spectra dataset
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## ✅ Completed Tasks

### Task 1: Astronomical Object Classification & EDA
**Dataset:** SDSS (Sloan Digital Sky Survey) - 100,000 celestial objects

**Objectives:**
- Perform comprehensive exploratory data analysis
- Visualize celestial coordinates on 3D sphere
- Analyze photometric bands (u, g, r, i, z)
- Create color-color diagrams
- Handle missing values and outliers
- Apply unsupervised learning (clustering)

**Key Findings:**
- Successfully visualized 100K objects on celestial sphere
- Identified patterns in STAR, GALAXY, and QSO classes
- Removed highly correlated features (g≈u≈z, r≈i)
- Created interactive 3D visualizations with Plotly

**Notebook:** [`Technical(AI)/Task_1/task1.ipynb`](Technical(AI)/Task_1/task1.ipynb)

---

### Task 2: Stellar Parameter Prediction
**Dataset:** Stellar spectra with physical parameters

**Objectives:**
- Predict **surface gravity (logg)** using regression
- Predict **effective temperature (Teff)** using regression
- Apply feature selection techniques
- Evaluate model performance (RMSE, R², MAE)

**Approach:**
- Baseline: Linear Regression
- Advanced: Random Forest Regressor with hyperparameter tuning
- Feature engineering: proper motion, metallicity, photometric errors
- Outlier handling: IQR capping and percentile winsorization

**Notebook:** [`Technical(AI)/Task_2/task2.ipynb`](Technical(AI)/Task_2/task2.ipynb)

---

## 🔧 Requirements

### Minimum Requirements
- Python 3.8+
- pip or conda

### Core Dependencies
```
jupyter
jupyterlab
numpy
pandas
matplotlib
seaborn
plotly
scikit-learn
scipy
```

### Optional (for enhanced features)
```
umap-learn          # Better dimensionality reduction
skimpy              # Enhanced data summaries
summarytools        # Statistical summaries
polars              # Fast dataframe operations
xgboost             # Gradient boosting (bonus)
```

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/N1KH1LT0X1N/DJS_NOVA.git
cd DJS_NOVA
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

# Or using conda
conda create -n djs_nova python=3.10
conda activate djs_nova
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter
```bash
jupyter lab
# or
jupyter notebook
```

### 5. Open Notebooks
Navigate to:
- `Technical(AI)/Task_1/task1.ipynb` for Task 1
- `Technical(AI)/Task_2/task2.ipynb` for Task 2

---

## 📖 Usage

### Interactive Exploration
Open notebooks in Jupyter Lab/Notebook and run cells sequentially. Each notebook is self-contained with:
- Clear section headers
- Inline comments
- Visualization outputs
- Result interpretations

### Headless Execution
Run notebooks without GUI:
```bash
# Execute and save output
jupyter nbconvert --to notebook --execute Technical\(AI\)/Task_1/task1.ipynb --output task1_executed.ipynb

# Convert to HTML
jupyter nbconvert --to html Technical\(AI\)/Task_1/task1.ipynb
```

### Key Functions & Utilities

**Task 1 - EDA:**
```python
# Load and explore SDSS data
df = pd.read_csv('Task 1.csv')
from skimpy import skim
skim(df)  # Enhanced summary statistics

# 3D celestial sphere visualization
import plotly.graph_objs as go
# ... (see notebook for full code)
```

**Task 2 - Regression:**
```python
# Standardization pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])
pipe.fit(X_train, y_train)
```

---

## 📊 Results & Evaluation

### Task 1 Highlights
- **Data Quality:** Handled 25% missing values in `cam_col`
- **Feature Reduction:** Dropped 7 highly correlated features
- **Clustering:** Applied KMeans, DBSCAN for unsupervised classification
- **Visualizations:** Interactive 3D sphere, correlation matrices, color-color diagrams

### Task 2 Highlights
- **Surface Gravity Model:** Random Forest with optimized hyperparameters
- **Temperature Model:** (results pending - see notebook)
- **Feature Importance:** Top predictors identified
- **Outlier Treatment:** IQR capping with multiplier=2.5

**Metrics:**
- Cross-validation RMSE, R², MAE
- Test set performance
- Residual plots and diagnostics

---

## 🤝 Contributing

This repository represents completed recruitment tasks. Suggestions for improvements are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 for Python code
- Use descriptive variable names
- Add comments for complex logic
- Include docstrings for functions

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Author:** Nikhil  
**GitHub:** [@N1KH1LT0X1N](https://github.com/N1KH1LT0X1N)  
**Repository:** [DJS_NOVA](https://github.com/N1KH1LT0X1N/DJS_NOVA)

---

## 🙏 Acknowledgments

- **DJS Nova** for providing the recruitment tasks
- **SDSS** for the astronomical dataset
- **scikit-learn** and **plotly** communities for excellent libraries

---

**⭐ If you find this repository helpful, please consider giving it a star!**
