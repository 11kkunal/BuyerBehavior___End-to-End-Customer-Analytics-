import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

sns.set_theme(style='whitegrid', palette='Set2', font_scale=1.15)
plt.rcParams.update({'figure.dpi': 110, 'figure.autolayout': True})
RANDOM_STATE = 42

print("Running customer segmentation pipeline...")

# Load dataset
df_raw = pd.read_csv('Mall_Customers.csv')
print(f"Loaded dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Data cleaning and preparation
df = df_raw.copy()
num_cols = ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']

miss_pct = df.isnull().mean() * 100
fig, ax = plt.subplots(figsize=(7, 3.5))
miss_pct.plot(kind='bar', color='#5B9BD5' if not miss_pct.any() else '#E07B8A', ax=ax, edgecolor='white')
ax.set_title('Missing-Value Percentage by Column', fontweight='bold')
ax.set_ylabel('% Missing')
ax.set_ylim(0, max(10, miss_pct.max() + 5))
for i, v in enumerate(miss_pct):
    ax.text(i, v + 0.3, f"{v:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig('plot_missing_values.png', bbox_inches='tight')
plt.close()

for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

df = df.drop_duplicates().reset_index(drop=True)
df['Gender_Code'] = df['Gender'].map({'Male': 0, 'Female': 1})

def iqr_bounds(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

for col in num_cols:
    lo, hi = iqr_bounds(df[col])
    n_out = len(df[(df[col] < lo) | (df[col] > hi)])
    if n_out > 0:
        print(f"Outliers in {col}: {n_out} detected")

# Exploratory data analysis
palette = {'Male': '#5B9BD5', 'Female': '#E07B8A'}
plot_cfg = [
    ('Age', 'steelblue', 'Age (years)'),
    ('Annual Income (k$)', 'seagreen', 'Annual Income (k$)'),
    ('Spending Score (1-100)', 'tomato', 'Spending Score'),
]

# Histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (col, color, xlabel) in zip(axes, plot_cfg):
    sns.histplot(df[col], kde=True, color=color, bins=20, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {xlabel}')
fig.suptitle('Histogram + KDE for Numerical Features', fontsize=14, fontweight='bold', y=1.02)
plt.savefig('plot_histograms.png', bbox_inches='tight')
plt.close()

# Boxplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (col, color, xlabel) in zip(axes, plot_cfg):
    sns.boxplot(y=df[col], color=color, ax=ax, width=0.4,
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5))
    ax.set_ylabel(xlabel)
    ax.set_title(f'Boxplot: {xlabel}')
fig.suptitle('Boxplots — Median, IQR & Outliers', fontsize=14, fontweight='bold', y=1.02)
plt.savefig('plot_boxplots.png', bbox_inches='tight')
plt.close()

# Gender-stratified boxplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (col, _, xlabel) in zip(axes, plot_cfg):
    sns.boxplot(data=df, x='Gender', y=col, palette=palette, ax=ax, width=0.5)
    ax.set_title(f'{xlabel} by Gender')
    ax.set_xlabel('')
    ax.set_ylabel(xlabel)
fig.suptitle('Feature Distributions by Gender', fontsize=14, fontweight='bold', y=1.02)
plt.savefig('plot_gender_boxplots.png', bbox_inches='tight')
plt.close()

# Scatterplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Gender', palette=palette, alpha=0.8, s=60, ax=axes[0])
axes[0].set_title('Annual Income vs Spending Score')
sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)',
                hue='Gender', palette=palette, alpha=0.8, s=60, ax=axes[1])
axes[1].set_title('Age vs Spending Score')
fig.suptitle('Scatterplots: Key Relationships', fontsize=14, fontweight='bold', y=1.02)
plt.savefig('plot_scatterplots.png', bbox_inches='tight')
plt.close()

# Correlation matrix
corr_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Code']
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(6, 5))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.savefig('plot_correlation.png', bbox_inches='tight')
plt.close()

# Pairplot
pp = sns.pairplot(df[corr_cols + ['Gender']], hue='Gender', palette=palette,
                  plot_kws={'alpha': 0.6, 's': 40}, diag_kind='kde')
pp.figure.suptitle('Pair Plot of All Numerical Features', y=1.02, fontsize=13, fontweight='bold')
plt.savefig('plot_pairplot.png', bbox_inches='tight')
plt.close()

# Feature scaling
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Finding optimal k
k_range = range(2, 11)
wcss_vals = []
sil_vals = []
db_vals = []

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=500, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    wcss_vals.append(km.inertia_)
    sil_vals.append(silhouette_score(X_scaled, labels))
    db_vals.append(davies_bouldin_score(X_scaled, labels))

best_sil_k = list(k_range)[np.argmax(sil_vals)]
best_db_k = list(k_range)[np.argmin(db_vals)]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(list(k_range), wcss_vals, 'o-', color='steelblue', linewidth=2, markersize=7)
axes[0].axvline(x=5, color='red', linestyle='--', alpha=0.6, label='k=5')
axes[0].set_title('Elbow Method — WCSS vs k', fontweight='bold')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('WCSS (Inertia)')
axes[0].legend()

axes[1].plot(list(k_range), sil_vals, 's-', color='seagreen', linewidth=2, markersize=7)
axes[1].axvline(x=best_sil_k, color='red', linestyle='--', alpha=0.6, label=f'best k={best_sil_k}')
axes[1].set_title('Silhouette Score vs k', fontweight='bold')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()

axes[2].plot(list(k_range), db_vals, 'D-', color='tomato', linewidth=2, markersize=7)
axes[2].axvline(x=best_db_k, color='blue', linestyle='--', alpha=0.6, label=f'best k={best_db_k}')
axes[2].set_title('Davies-Bouldin Index vs k', fontweight='bold')
axes[2].set_xlabel('Number of Clusters (k)')
axes[2].set_ylabel('DB Index')
axes[2].legend()

fig.suptitle('Cluster-Count Evaluation Metrics', fontsize=14, fontweight='bold', y=1.02)
plt.savefig('plot_elbow.png', bbox_inches='tight')
plt.close()

# Fit K-Means with k=5
k_optimal = 5
kmeans = KMeans(n_clusters=k_optimal, init='k-means++', n_init=30, max_iter=500, random_state=RANDOM_STATE)
df['Cluster'] = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, df['Cluster'])
db = davies_bouldin_score(X_scaled, df['Cluster'])
print(f"K-Means (k=5) -> Inertia: {kmeans.inertia_:.2f}, Silhouette: {sil:.4f}, Davies-Bouldin: {db:.4f}")

centroids_scaled = kmeans.cluster_centers_
centroids_orig = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_orig, columns=features).round(1)
centroids_df['Size'] = df['Cluster'].value_counts().sort_index().values

cluster_colors = sns.color_palette('Set1', k_optimal)

# 2D cluster scatter
fig, ax = plt.subplots(figsize=(9, 6))
for cid in sorted(df['Cluster'].unique()):
    mask = df['Cluster'] == cid
    ax.scatter(df.loc[mask, 'Annual Income (k$)'], df.loc[mask, 'Spending Score (1-100)'],
               c=[cluster_colors[cid]], label=f'Cluster {cid}', s=60, alpha=0.8,
               edgecolors='white', linewidths=0.4)
ax.scatter(centroids_df['Annual Income (k$)'], centroids_df['Spending Score (1-100)'],
           c='black', marker='X', s=200, zorder=5, label='Centroids')
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.set_title('K-Means Clusters: Annual Income vs Spending Score', fontsize=13, fontweight='bold')
ax.legend()
plt.savefig('plot_clusters_2d.png', bbox_inches='tight')
plt.close()

# Radar chart
radar_data = centroids_df[features].copy()
radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())
n_angles = len(features)
angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for i, row in radar_norm.iterrows():
    vals = row.tolist() + row.tolist()[:1]
    ax.plot(angles, vals, 'o-', linewidth=2, color=cluster_colors[i], label=f'Cluster {i}')
    ax.fill(angles, vals, alpha=0.12, color=cluster_colors[i])
ax.set_thetagrids(np.degrees(angles[:-1]), features)
ax.set_title('Cluster Profiles — Radar Chart\n(normalised features)', fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15))
plt.savefig('plot_radar.png', bbox_inches='tight')
plt.close()

# Violin plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (col, color, xlabel) in zip(axes, plot_cfg):
    sns.violinplot(data=df, x='Cluster', y=col, palette='Set1', inner='box', ax=ax)
    ax.set_title(f'{xlabel} per Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel(xlabel)
fig.suptitle('Feature Distributions per Cluster (Violin Plot)', fontsize=13, fontweight='bold', y=1.02)
plt.savefig('plot_violin_cluster.png', bbox_inches='tight')
plt.close()

# Gender composition
gender_cluster = df.groupby(['Cluster', 'Gender']).size().unstack(fill_value=0)
gender_pct = gender_cluster.div(gender_cluster.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(8, 4))
gender_pct.plot(kind='bar', stacked=True, color=['#5B9BD5', '#E07B8A'], edgecolor='white', ax=ax)
ax.set_xlabel('Cluster')
ax.set_ylabel('Percentage (%)')
ax.set_title('Gender Composition per Cluster', fontweight='bold')
ax.legend(title='Gender', bbox_to_anchor=(1.01, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.savefig('plot_gender_cluster.png', bbox_inches='tight')
plt.close()

# Principal component analysis
pca = PCA(n_components=3, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_ * 100
cumulative = np.cumsum(explained)

comp_labels = [f'PC{i+1}' for i in range(3)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(comp_labels, explained, color=['#4C72B0', '#DD8452', '#55A868'], edgecolor='white')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance (%)')
axes[0].set_title('Scree Plot — Individual Variance')
axes[1].plot(comp_labels, cumulative, 'o-', color='purple', linewidth=2.5)
axes[1].axhline(y=90, color='red', linestyle='--', alpha=0.6, label='90% threshold')
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Cumulative Variance (%)')
axes[1].set_title('Scree Plot — Cumulative Variance')
axes[1].legend()
fig.suptitle('PCA Explained Variance', fontsize=13, fontweight='bold', y=1.02)
plt.savefig('plot_scree.png', bbox_inches='tight')
plt.close()

# PCA 2D biplot
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca2 = pca2.fit_transform(X_scaled)
fig, ax = plt.subplots(figsize=(9, 7))
for cid in sorted(df['Cluster'].unique()):
    mask = df['Cluster'] == cid
    ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
               c=[cluster_colors[cid]], label=f'Cluster {cid}', s=65, alpha=0.8,
               edgecolors='white', linewidths=0.4)
scale = 2.5
for j, feat in enumerate(features):
    ax.annotate('', xy=(pca2.components_[0, j] * scale, pca2.components_[1, j] * scale),
                xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='black', lw=1.8))
    ax.text(pca2.components_[0, j] * scale * 1.12, pca2.components_[1, j] * scale * 1.12,
            feat, fontsize=9, color='black', fontweight='bold')
ax.axhline(0, color='grey', linewidth=0.6, linestyle='--')
ax.axvline(0, color='grey', linewidth=0.6, linestyle='--')
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('PCA Biplot — Clusters in 2-D Principal Component Space', fontsize=12, fontweight='bold')
ax.legend()
plt.savefig('plot_pca_biplot.png', bbox_inches='tight')
plt.close()

# PCA loadings
loadings = pd.DataFrame(pca2.components_.T, index=features, columns=['PC1', 'PC2']).round(3)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(loadings, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            linewidths=0.8, cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('PCA Feature Loadings', fontsize=12, fontweight='bold')
plt.savefig('plot_pca_loadings.png', bbox_inches='tight')
plt.close()

# Persona assignment based on quadrant profiles
income_rank = centroids_df['Annual Income (k$)'].rank()
spending_rank = centroids_df['Spending Score (1-100)'].rank()

persona_map = {}
for cid in range(k_optimal):
    inc = income_rank[cid]
    sp = spending_rank[cid]
    if inc >= 4 and sp >= 4:
        persona_map[cid] = 'High-Value / VIP'
    elif inc >= 4 and sp <= 2:
        persona_map[cid] = 'Wealthy Savers'
    elif inc <= 2 and sp >= 4:
        persona_map[cid] = 'Impulsive Spenders'
    elif inc <= 2 and sp <= 2:
        persona_map[cid] = 'Budget Conscious'
    else:
        persona_map[cid] = 'Mid-Market'

df['Persona'] = df['Cluster'].map(persona_map)

# Final annotated cluster plot
fig, ax = plt.subplots(figsize=(10, 7))
for cid in sorted(df['Cluster'].unique()):
    mask = df['Cluster'] == cid
    persona = persona_map[cid]
    ax.scatter(df.loc[mask, 'Annual Income (k$)'], df.loc[mask, 'Spending Score (1-100)'],
               c=[cluster_colors[cid]], label=f'C{cid}: {persona}',
               s=70, alpha=0.85, edgecolors='white', linewidths=0.5)
for i, row in centroids_df.iterrows():
    ax.annotate(f'C{i}\n{persona_map[i].split()[0]}',
                xy=(row['Annual Income (k$)'], row['Spending Score (1-100)']),
                fontsize=9, fontweight='bold', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.7))
ax.scatter(centroids_df['Annual Income (k$)'], centroids_df['Spending Score (1-100)'],
           c='black', marker='X', s=220, zorder=5)
ax.set_xlabel('Annual Income (k$)', fontsize=12)
ax.set_ylabel('Spending Score (1-100)', fontsize=12)
ax.set_title('Customer Segments — Annotated', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
plt.savefig('plot_final_clusters.png', bbox_inches='tight')
plt.close()

# Save segmented output
export_cols = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)',
               'Spending Score (1-100)', 'Cluster', 'Persona']
df[export_cols].to_csv('Mall_Customers_Segmented.csv', index=False)
print("Pipeline execution complete. Output saved to Mall_Customers_Segmented.csv.")
