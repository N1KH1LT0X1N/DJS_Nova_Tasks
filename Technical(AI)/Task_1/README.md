# Task 1: Exploratory Data Analysis - SDSS Astronomical Dataset

## 🎯 Objective

Conduct comprehensive exploratory data analysis on the SDSS (Sloan Digital Sky Survey) dataset containing 100,000 celestial objects to understand:
- Dataset structure and quality
- Class distribution (STAR, GALAXY, QSO)
- Photometric magnitude patterns
- Spatial distribution on the celestial sphere
- Temporal patterns in observations

---

## 📊 Dataset Overview

- **File:** `Task 1.csv`
- **Metadata:** `Metadata.pdf`
- **Size:** 100,000 observations
- **Features:** 18 columns including:
  - Celestial coordinates (RA, Dec)
  - Photometric magnitudes (u, g, r, i, z)
  - Spectroscopic data (redshift, class)
  - Survey metadata (MJD, plate, fiber)

---

## 🔍 Key Analyses Performed

### 1. **Data Quality Assessment**
- Missing value analysis (25.5% in `cam_col`)
- Duplicate detection in `obj_ID`
- Outlier identification using IQR method
- Data type verification

### 2. **Descriptive Statistics**
- Summary statistics using `pandas.describe()` and `skimpy.skim()`
- Distribution analysis for all numeric features
- Correlation matrix to identify redundant features

### 3. **Spatial Analysis**
- **3D Celestial Sphere Visualization**
  - Converted RA/Dec to Cartesian coordinates
  - Interactive Plotly 3D scatter with redshift coloring
  - Wireframe sphere overlay for context
- **Sky Coverage**
  - Declination range: -18.8° to +83.0°
  - Predominantly Northern Hemisphere observations

### 4. **Photometric Analysis**
- **Band Distributions:** Histograms for u, g, r, i, z magnitudes
- **Boxplots:** Compared spread and outliers across bands
- **Pairplot:** Visualized relationships between all magnitude pairs
- **Color-Color Diagrams:**
  - (u-g) vs (g-r) for stellar population studies
  - Distinct patterns for different object classes

### 5. **Class Distribution**
- Pie chart showing proportion of STAR/GALAXY/QSO
- Bar plots with different color schemes
- Magnitude distributions by class (boxplots)

### 6. **Temporal Analysis**
- MJD (Modified Julian Date) distribution
- Observation timeline patterns
- Plate/field coverage statistics

### 7. **Feature Engineering**
- Dropped highly correlated features:
  - `g` and `z` (identical to `u`)
  - `i` (highly correlated with `r`)
  - `obj_ID` (correlated with `run_ID`)
  - `spec_obj_ID`, `MJD` (correlated with `plate`)
- Final reduced dataset: 11 features (from 18)

### 8. **Unsupervised Learning (Bonus)**
- KMeans clustering to recover `class` labels
- DBSCAN for density-based clustering
- Evaluation using Adjusted Rand Index (ARI)
- Cluster-to-class mapping analysis

---

## 📈 Key Findings

1. **Data Quality:**
   - High-quality dataset with minimal missing values
   - `cam_col` column has ~25% missingness (systematic, not random)
   - No critical data integrity issues

2. **Feature Redundancy:**
   - Strong correlations found: u≈g≈z (ρ > 0.95)
   - r and i bands highly correlated (ρ > 0.98)
   - Successfully reduced dimensions without information loss

3. **Spatial Distribution:**
   - Northern Hemisphere bias (typical for SDSS)
   - Dense coverage in certain sky regions
   - Clear visualization of large-scale structure

4. **Object Classes:**
   - Distinct magnitude patterns for STAR/GALAXY/QSO
   - Redshift ranges vary significantly by class
   - Color indices effectively separate object types

5. **Outliers:**
   - Identified extreme values in proper motion (pmL, pmB)
   - Some magnitude measurements at detection limits
   - Treated using IQR capping (multiplier=2.5)

---

## 🛠️ Tools & Techniques Used

### Libraries:
- **pandas, numpy:** Data manipulation
- **matplotlib, seaborn:** Static visualizations
- **plotly:** Interactive 3D plots
- **scikit-learn:** Clustering, preprocessing
- **skimpy:** Enhanced data summaries

### Visualization Techniques:
- Histograms, boxplots, violin plots
- Correlation heatmaps (standard + advanced scatter)
- 3D scatter plots with spherical coordinates
- Pairplots with class coloring
- Color-color diagrams

### Statistical Methods:
- IQR-based outlier detection
- Pearson correlation analysis
- Principal Component Analysis (PCA)
- KMeans and DBSCAN clustering

---

## 📁 Files in This Directory

- `task1.ipynb` - Complete analysis notebook with code and visualizations
- `task1.md` - This file (task description and results)
- `Task 1.csv` - SDSS dataset (100K objects)
- `Metadata.pdf` - Dataset documentation

---

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn plotly scikit-learn skimpy
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter lab
   ```

3. **Open Notebook:**
   - Navigate to `task1.ipynb`
   - Run cells sequentially (Kernel → Restart & Run All)

4. **Explore Interactively:**
   - 3D plots are interactive (zoom, rotate, pan)
   - Hover over points to see obj_ID and coordinates

---

## 📊 Sample Visualizations

### Generated Plots:
- Missing values heatmap
- Class distribution pie chart
- 3D celestial sphere (interactive)
- Color-color diagrams
- Correlation matrices (standard + advanced)
- Magnitude vs. redshift scatter
- Pairplot of photometric bands

---

## 🎓 Key Insights for Machine Learning

1. **Feature Selection:**
   - Use reduced feature set (11 features) for modeling
   - Color indices (u-g, g-r, etc.) are more informative than raw magnitudes

2. **Class Imbalance:**
   - Check class distribution before classification
   - Consider stratified sampling or SMOTE if needed

3. **Outliers:**
   - Handle carefully - some may be genuine astrophysical phenomena
   - Use robust scalers (RobustScaler) or tree-based models

4. **Missing Data:**
   - `cam_col` missing systematically - investigate before imputation
   - Use KNN or iterative imputation for numeric features

---

## 📖 References

- [SDSS Documentation](https://www.sdss.org/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Plotly Python](https://plotly.com/python/)
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)

---

## ✅ Completion Status

- [x] Data loading and initial inspection
- [x] Missing value analysis
- [x] Descriptive statistics
- [x] Spatial visualizations (3D sphere)
- [x] Photometric analysis (histograms, boxplots, pairplots)
- [x] Color-color diagrams
- [x] Correlation analysis
- [x] Feature engineering (reduction from 18 → 11)
- [x] Temporal analysis (MJD)
- [x] Outlier detection and treatment
- [x] Unsupervised clustering (bonus)

---

**Analysis Complete!** 🎉  
See `task1.ipynb` for detailed code, visualizations, and interpretations.
