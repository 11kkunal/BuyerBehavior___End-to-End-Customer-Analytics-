# Customer Segmentation & Behavioral Clustering Analysis

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0+-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Mall%20Customers-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

An end-to-end unsupervised machine learning pipeline for customer segmentation using the authentic **Kaggle Mall Customer dataset**. This project applies **Exploratory Data Analysis (EDA)**, **K-Means Clustering**, **Silhouette & Davies-Bouldin evaluation**, and **Principal Component Analysis (PCA)** to uncover distinct shopper profiles and provide data-driven marketing strategies.

---

## Table of Contents
1. [Project Overview & Business Problem](#project-overview--business-problem)
2. [Real-World Kaggle Dataset](#real-world-kaggle-dataset)
3. [Architecture & Workflow Pipeline](#architecture--workflow-pipeline)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Clustering Methodology & Optimization](#clustering-methodology--optimization)
6. [Dimensionality Reduction (PCA)](#dimensionality-reduction-pca)
7. [Customer Personas & Business Recommendations](#customer-personas--business-recommendations)
8. [Visualizations Gallery](#visualizations-gallery)
9. [Project Directory Layout](#project-directory-layout)
10. [Quickstart & Reproducibility](#quickstart--reproducibility)
11. [Tech Stack](#tech-stack)

---

## Project Overview & Business Problem

Retail malls and marketing teams face high customer acquisition costs and diminished conversion rates when relying on generic mass-marketing campaigns. Shoppers differ significantly in age, earning capacity, and retail spending propensity.

### Objectives:
- **Real-World Data Ingestion**: Utilize the benchmark real-world Kaggle Mall Customers dataset.
- **Identify Natural Customer Cohorts**: Uncover behavioral patterns without predefined target labels using unsupervised machine learning.
- **Translate Math to Business Value**: Transform mathematical cluster centroids into actionable marketing personas with customized retention and monetization strategies.

---

## Real-World Kaggle Dataset

This project utilizes the authentic benchmark **Mall Customer Segmentation Data** originally hosted on Kaggle:
- **Kaggle Dataset**: [Customer Segmentation Tutorial in Python by Vijay Choudhary](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Local File**: `Mall_Customers.csv` (can be fetched anytime via `fetch_kaggle_dataset.py`)
- **Dimensions**: 200 customer records × 5 features
- **Data Quality**: 100% complete records (0 missing values, 0 duplicate records)

### Data Dictionary

| Column | Type | Description | Range / Values |
| :--- | :--- | :--- | :--- |
| `CustomerID` | Integer | Unique identifier for each shopper | $1 \dots 200$ |
| `Gender` | String | Customer biological sex | `Male` (44%), `Female` (56%) |
| `Age` | Integer | Age of customer in years | $18 \dots 70$ (Mean: 38.85) |
| `Annual Income (k$)` | Integer | Annual income of customer in thousands of dollars | $\$15\text{k} \dots \$137\text{k}$ (Median: $\$61.5\text{k}$) |
| `Spending Score (1-100)` | Integer | Mall-assigned behavioral score based on customer spend | $1 \dots 99$ (Median: 50.0) |

> **Outlier Note (IQR Method)**: Interquartile range analysis identifies two high-income outliers (CustomerID 199 and 200 with $\$137\text{k}$ income). Both are retained to accurately segment the ultra-affluent demographic.

---

## Architecture & Workflow Pipeline

```mermaid
flowchart LR
    A["Raw Data (Kaggle CSV)"] --> B["Data Validation & Cleaning"]
    B --> C["Exploratory Data Analysis"]
    C --> D["Feature Scaling (StandardScaler)"]
    D --> E["Optimal k Search (Elbow & Silhouette)"]
    E --> F["K-Means Clustering (k=5)"]
    F --> G["PCA 2D/3D Projection & Biplot"]
    G --> H["Persona Mapping & Strategic Playbook"]
```

1. **Ingestion & Validation**: Automated validation of column types, null completeness, duplicate elimination, and IQR outlier boundary estimation.
2. **Exploratory Data Analysis (EDA)**: Univariate distributions (KDE), bivariate cross-correlations, boxplots, pairplots, and gender stratification.
3. **Feature Scaling**: Standardization using `StandardScaler` so that features with varying numerical magnitudes contribute proportionally to Euclidean distance calculations.
4. **Cluster Optimization**: Evaluating $k \in [2, 10]$ across three complementary metrics:
   - **WCSS / Inertia** (Elbow method)
   - **Silhouette Score** (cohesion vs separation)
   - **Davies-Bouldin Index** (cluster similarity ratio)
5. **Model Fitting**: Optimal $k=5$ K-Means model trained with `k-means++` centroid initialization over 30 independent restarts.
6. **Dimensionality Reduction**: Principal Component Analysis (PCA) to project 3-dimensional standardized customer space into 2D principal components with annotated feature loading vectors.
7. **Business Strategy Formulation**: Segment persona profiling and strategic recommendations.

---

## Exploratory Data Analysis (EDA)

Key statistical and visual insights extracted from the dataset:

- **Demographics**: Mall shoppers skew female (56% female vs 44% male). Median customer age is 36 years old.
- **Bimodal Spending Behavior**: Spending score exhibits two distinct clusters around score 35-40 and score 75-80.
- **Income vs. Spending Independence**: Pearson correlation between Annual Income and Spending Score is virtually non-existent ($r \approx 0.01$), demonstrating that affluent shoppers do not automatically spend more.
- **Age vs. Spending Inversion**: A negative correlation ($r = -0.33$) exists between Age and Spending Score; younger shoppers consistently show a higher propensity for discretionary spending.

---

## Clustering Methodology & Optimization

### Finding the Optimal Number of Clusters ($k$)

We evaluated cluster counts from $k=2$ through $k=10$:
- **Elbow Point**: The WCSS curve displays clear inflection kinks at $k=3$ and $k=5$.
- **Silhouette Score**: Strong separation achieved at $k=5$ ($\text{score} = 0.4166$) and $k=6$ ($\text{score} = 0.4274$).
- **Business Interpretability**: At $k=5$, clusters map cleanly to the 4 quadrants of the Income-Spending plane plus the central mid-market hub.

### K-Means Model Performance ($k=5$)

- **Inertia (WCSS)**: `168.25`
- **Silhouette Coefficient**: `0.4166` (compact, well-isolated clusters)
- **Davies-Bouldin Index**: `0.8746` (lower score confirms high partition quality)

---

## Dimensionality Reduction (PCA)

To validate cluster geometry and inspect multivariate variance without losing information:
- **PC1 (44.27% of variance)**: Captures the primary axis contrasting high spending vs. low spending and income alignment.
- **PC2 (33.31% of variance)**: Captures customer age variance and life-stage divergence.
- **Cumulative Variance**: **77.57%** explained by the first two principal components, and **100%** across 3 components.

The PCA biplot confirms five distinct, non-overlapping clusters in transformed orthogonal space.

---

## Customer Personas & Business Recommendations

Based on the multi-dimensional centroid profiles, we established 5 actionable marketing personas:

```
                  High Spending Score
                          ▲
        Impulsive         │        High-Value
        Spenders          │           VIPs
      (Young, Trend)      │      (Affluent, Prime)
                          │
  Low ────────────────────┼────────────────────► High
  Income                  │                    Income
       Budget             │        Wealthy
       Conscious          │         Savers
      (Value-seekers)     │      (High Wealth, Frugal)
                          ▼
                  Low Spending Score
                          
            [ Mid-Market Hub in the Center ]
```

### Segment Breakdown Matrix

| Segment | Cluster | Size (%) | Avg Age | Avg Income | Avg Spend | % Female | Core Behavioral Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **High-Value / VIP** | `C2` | 40 (20.0%) | 32.9 yrs | **$86.1k** | **81.5** | 55.0% | High earnings, high mall spending. Prime demographic for luxury goods. |
| **Wealthy Savers** | `C3` | 39 (19.5%) | 39.9 yrs | **$86.1k** | **19.4** | 48.7% | High earnings, conservative spending. Prioritize utility, investment, quality over impulse. |
| **Impulsive Spenders** | `C1` | 54 (27.0%) | 25.2 yrs | **$41.1k** | **62.2** | 59.3% | Youngest cohort, lower income, disproportionately high retail spending. Trend-driven. |
| **Budget Conscious** | `C0` | 20 (10.0%) | 46.2 yrs | **$26.8k** | **18.4** | 60.0% | Low income, highly disciplined spending. Extremely price-sensitive shoppers. |
| **Mid-Market** | `C4` | 47 (23.5%) | 55.6 yrs | **$54.4k** | **48.9** | 57.4% | Mature cohort, median earnings, balanced spending habits. Reliable core mall traffic. |

---

### Tailored Marketing Playbooks

#### 1. High-Value / VIP
* **Target Objective**: Maximize lifetime customer value (LTV) and brand advocacy.
* **Campaign Tactics**:
  - Invitation-only shopping events and preview nights for new collections.
  - Dedicated personal stylists and concierge checkout services.
  - Premium tier loyalty perks (free valet, bespoke gift rewards).

#### 2. Wealthy Savers
* **Target Objective**: Convert dormant purchasing power into active shopping.
* **Campaign Tactics**:
  - Emphasize craftsmanship, product durability, and long-term value over fast-fashion discounts.
  - Showcase high-end electronics, premium home appliances, and timeless luxury staples.
  - High-touch consultative sales rather than aggressive impulse promotions.

#### 3. Impulsive Spenders
* **Target Objective**: Capitalize on trend enthusiasm while reducing cart abandonment.
* **Campaign Tactics**:
  - Limited-time flash sales and influencer-promoted product drops.
  - Integrated "Buy Now, Pay Later" (BNPL) checkout options (e.g., Klarna, Affirm).
  - Gamified shopping rewards on social channels and mobile apps.

#### 4. Budget Conscious
* **Target Objective**: Provide accessible value without damaging store margins.
* **Campaign Tactics**:
  - End-of-season clearance promotions, factory outlet discounts, and BOGO deals.
  - Digital coupon books and loyalty points exchangeable for essential grocery/pharmacy items.

#### 5. Mid-Market
* **Target Objective**: Maintain steady cadence of mall visits and basket size.
* **Campaign Tactics**:
  - Family-oriented weekend bundles (dining + movie + retail combos).
  - Seasonal holiday catalogs (Back-to-School, Black Friday, Mother's Day).
  - Tiered spending vouchers (e.g., "$15 off when you spend $75").

---

## Visualizations Gallery

The pipeline generates 16 high-resolution charts documenting the entire process:

| Visual Artifact | File | Description |
| :--- | :--- | :--- |
| **Annotated Segments** | [`plot_final_clusters.png`](plot_final_clusters.png) | 2D scatter of final customer personas with centroid coordinates |
| **Optimal k Evaluation** | [`plot_elbow.png`](plot_elbow.png) | 3-panel comparison: Elbow WCSS, Silhouette Score, and Davies-Bouldin Index |
| **Cluster Scatter** | [`plot_clusters_2d.png`](plot_clusters_2d.png) | Annual Income vs. Spending Score color-coded by cluster |
| **Radar Profiles** | [`plot_radar.png`](plot_radar.png) | Polar radar chart comparing normalized Age, Income, and Spend per cluster |
| **PCA Biplot** | [`plot_pca_biplot.png`](plot_pca_biplot.png) | Cluster separation in 2D principal component space with feature vectors |
| **PCA Scree Plot** | [`plot_scree.png`](plot_scree.png) | Individual & cumulative explained variance ratios |
| **PCA Loadings** | [`plot_pca_loadings.png`](plot_pca_loadings.png) | Heatmap of feature weights driving PC1 and PC2 |
| **Feature Violins** | [`plot_violin_cluster.png`](plot_violin_cluster.png) | Distribution of Age, Income, and Spend across all 5 clusters |
| **Gender by Cluster** | [`plot_gender_cluster.png`](plot_gender_cluster.png) | Stacked bar chart showing gender distribution per cluster |
| **Pairplot** | [`plot_pairplot.png`](plot_pairplot.png) | Full pairwise bivariate relationships colored by gender |
| **Correlation Matrix** | [`plot_correlation.png`](plot_correlation.png) | Lower-triangle correlation heatmap of numerical features |
| **Histograms & KDE** | [`plot_histograms.png`](plot_histograms.png) | Feature distribution curves with kernel density estimation |
| **Boxplots** | [`plot_boxplots.png`](plot_boxplots.png) | Median, interquartile ranges, and outlier points |
| **Gender Boxplots** | [`plot_gender_boxplots.png`](plot_gender_boxplots.png) | Feature boxplots broken down by male vs female |
| **Key Scatterplots** | [`plot_scatterplots.png`](plot_scatterplots.png) | Scatterplots of Income vs Spend and Age vs Spend |
| **Data Completeness** | [`plot_missing_values.png`](plot_missing_values.png) | Missing-value audit verifying 100% clean data |

---

## Project Directory Layout

```text
customer_segmentation/
├── Customer_Segmentation.ipynb    # Fully executed Jupyter notebook with interactive cell outputs
├── Mall_Customers.csv             # Authentic real-world Kaggle Mall Customers dataset (200 rows)
├── Mall_Customers_Segmented.csv   # Exported dataset with assigned cluster IDs and Persona labels
├── fetch_kaggle_dataset.py        # Automated script to fetch the authentic Kaggle dataset
├── run_pipeline.py                # Standalone end-to-end Python script (runs pipeline & exports plots)
├── build_notebook.py              # Programmatic notebook builder & nbconvert runner
├── README.md                      # Comprehensive project documentation
└── plot_*.png                     # 16 High-resolution visualization artifacts
```

---

## Quickstart & Reproducibility

### 1. Prerequisites
Python 3.10+

```bash
# Clone or navigate to the project directory
cd customer_segmentation
```

### 2. Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter nbformat nbconvert
```

### 3. Re-download Kaggle Dataset (Optional)
The real Kaggle dataset is already included in `Mall_Customers.csv`. If you ever wish to re-download it from source:
```bash
python fetch_kaggle_dataset.py
```

### 4. Run the Pipeline
To execute data cleaning, clustering, PCA, and generate all visualization PNGs:
```bash
python run_pipeline.py
```

### 5. Build & Execute Jupyter Notebook
To generate and pre-render the complete executed `Customer_Segmentation.ipynb` notebook:
```bash
python build_notebook.py
```

Or open the notebook interactively:
```bash
jupyter notebook Customer_Segmentation.ipynb
```

---

## Tech Stack

- **Data Wrangling**: `pandas 3.0+`, `numpy`
- **Machine Learning**: `scikit-learn` (`StandardScaler`, `KMeans`, `PCA`, `silhouette_score`, `davies_bouldin_score`)
- **Data Visualization**: `matplotlib`, `seaborn`
- **Notebook Environment**: `jupyter`, `nbformat`, `nbconvert`

---

## License & Acknowledgements

- **Dataset**: Vijay Choudhary on [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).
- **License**: MIT Open Source License.
