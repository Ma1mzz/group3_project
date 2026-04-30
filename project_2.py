"""
================================================================================
  CDS 2413 – Introduction to Data Science
  Project Deliverable 1 – Data Analysis Report
  Dataset : Shopping Trends (shopping_trends_updated.csv)
  Dependent  Variable : Purchase Amount (USD)
  Independent Variables: Age (numerical) | Category (categorical)
 ================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── colour palette ─────────────────────────────────────────────────────────────
C1 = '#2E86AB'   # steel blue
C2 = '#E84855'   # coral red
C3 = '#6A0572'   # purple
C4 = '#F18F01'   # amber
BG = '#F7F9FC'

def section(title):
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}")

def subsection(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")

# ══════════════════════════════════════════════════════════════════════════════
#  FIX 1 ─ ML ALGORITHM IDENTIFICATION (Task 3)

# ══════════════════════════════════════════════════════════════════════════════
section("ML ALGORITHM – IDENTIFICATION & JUSTIFICATION (Task 3)")
print("""
  Algorithm  : Multiple Linear Regression (primary) + Simple Linear Regression
  Type       : Supervised Machine Learning – Regression

  Purpose:
    The dependent variable 'Purchase Amount (USD)' is a continuous numerical
    variable. Regression algorithms are therefore the correct choice because
    they model the relationship between one or more independent variables
    (Age, Category after encoding) and a continuous output.

    • Simple Linear Regression  – models the effect of Age alone on purchase
      amount, producing an interpretable slope coefficient.
    • Multiple Linear Regression – incorporates all relevant independent
      variables (Age, encoded Category, Season, etc.) to improve predictive
      accuracy and explain more variance (higher R²).

  Justification:
    1. Output is numeric and unbounded → regression, not classification.
    2. Linear regression is transparent, fast, and provides statistical
       significance tests (p-values, confidence intervals) for each predictor.
    3. Pearson/Spearman correlations (Task 11) will confirm linearity
       assumptions before fitting the model.

  Deliverable 2 will implement, train, and evaluate both models using
  train/test split (80:20) and report R², MAE, and RMSE.
""")

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
section("DATASET OVERVIEW")
df = pd.read_csv('shopping_trends_updated.csv')
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")
print(f"\n  Column names:\n  {df.columns.tolist()}")
print(f"\n  Data Types:\n{df.dtypes.to_string()}")
print(f"\n  Missing Values:\n{df.isnull().sum().to_string()}")
print(f"\n  First 5 Rows:\n{df.head().to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIX 2 ─ DATA VALIDATION (missing from original code)
# ══════════════════════════════════════════════════════════════════════════════
section("DATA VALIDATION (Fix 2 – was missing in original)")

TARGET = 'Purchase Amount (USD)'

subsection("2A. Missing Value Check")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_report[missing_report['Missing Count'] > 0].to_string()
      if missing.sum() > 0 else "  ✔ No missing values found in any column.")

subsection("2B. Duplicate Row Check")
dups = df.duplicated().sum()
print(f"  Duplicate rows : {dups}")
if dups > 0:
    df = df.drop_duplicates()
    print(f"  ✔ Duplicates removed. New shape: {df.shape}")
else:
    print("  ✔ No duplicates found.")

subsection("2C. Data Type Validation")
expected_numeric = ['Age', TARGET, 'Review Rating', 'Previous Purchases']
for col in expected_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        coerced = df[col].isnull().sum()
        status = f"✔ OK" if coerced == 0 else f"⚠ {coerced} values could not be converted"
        print(f"  {col:<30}: {status}")

subsection("2D. Value Range Validation")

neg_purchase = (df[TARGET] <= 0).sum()
print(f"  Purchase Amount (USD) ≤ 0  : {neg_purchase} records "
      + ("✔ None" if neg_purchase == 0 else "⚠ Found – review data"))

invalid_age = ((df['Age'] < 0) | (df['Age'] > 120)).sum()
print(f"  Age out of range [0–120]   : {invalid_age} records "
      + ("✔ None" if invalid_age == 0 else "⚠ Found – review data"))

invalid_rating = ((df['Review Rating'] < 1) | (df['Review Rating'] > 5)).sum()
print(f"  Review Rating out of [1–5] : {invalid_rating} records "
      + ("✔ None" if invalid_rating == 0 else "⚠ Found – review data"))

subsection("2E. Categorical Consistency Check")
for cat_col in ['Category', 'Season', 'Gender']:
    if cat_col in df.columns:
        unique_vals = df[cat_col].unique().tolist()
        print(f"  {cat_col:<25}: {len(unique_vals)} unique values → {unique_vals}")

print(f"\n  ✔ Validation complete. Dataset ready for analysis.")
print(f"  Final shape after validation: {df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
#  DESCRIPTIVE STATS FUNCTION  (Task 6)
# ══════════════════════════════════════════════════════════════════════════════
subsection("Script for Descriptive Statistics Function (Task 6)")

def descriptive_stats(sample: pd.DataFrame, field: str) -> pd.Series:
    """
    Computes descriptive statistics for a specified numeric column.

    Parameters
    ----------
    sample : pd.DataFrame  – input dataset or sample
    field  : str           – column name to analyse

    Returns
    -------
    pd.Series containing: Count, Mean, Median, Std Dev, Variance,
                          Min, Q1, Q3, Max, Skewness, Kurtosis, Range, IQR
    """
    series = sample[field].dropna()
    result = pd.Series({
        'Count'    : int(series.count()),
        'Mean'     : round(series.mean(),     4),
        'Median'   : round(series.median(),   4),
        'Std Dev'  : round(series.std(),      4),
        'Variance' : round(series.var(),      4),
        'Min'      : round(series.min(),      4),
        'Q1 (25%)' : round(series.quantile(0.25), 4),
        'Q3 (75%)' : round(series.quantile(0.75), 4),
        'Max'      : round(series.max(),      4),
        'Skewness' : round(series.skew(),     4),
        'Kurtosis' : round(series.kurtosis(), 4),
        'Range'    : round(series.max() - series.min(), 4),
        'IQR'      : round(series.quantile(0.75) - series.quantile(0.25), 4),
    }, name=field)
    return result

print("  ✔ descriptive_stats(sample, field) function defined successfully.")
print("    Inputs  : sample (pd.DataFrame), field (str – column name)")
print("    Outputs : pd.Series with 13 statistical measures")


# ══════════════════════════════════════════════════════════════════════════════
#  RANDOM SAMPLING  (Task 7)
# ══════════════════════════════════════════════════════════════════════════════
subsection("Random Sampling – n = 150 (Task 7)")
n_sample = 150
replace  = len(df) < n_sample          # use replacement only if dataset < 150
np.random.seed(42)
random_sample = df.sample(n=n_sample, random_state=42, replace=replace)

print(f"  Dataset size              : {len(df)} records")
print(f"  Sample size requested     : {n_sample}")
print(f"  Sampling with replacement : {replace}")
print(f"  Random sample obtained    : {len(random_sample)} records")

random_stats = descriptive_stats(random_sample, TARGET)
print(f"\n  Descriptive Statistics – Random Sample ('{TARGET}'):")
print(random_stats.to_string())


# ══════════════════════════════════════════════════════════════════════════════
#  FIX 4 ─ SYSTEMATIC SAMPLING  (Task 8)
# ══════════════════════════════════════════════════════════════════════════════
subsection("Systematic Sampling – Fix 4 (Task 8)")

print("  CONDITION: Select every k-th record where k = floor(N / desired_n)")
print(f"  N (dataset size)  = {len(df)}")
print(f"  desired_n         = 150  (matches random sample size for fair comparison)")
print(f"  k = floor({len(df)} / 150) = {len(df) // 150}")
print()
print("  ── WHY THE ORIGINAL WAS WRONG ──────────────────────────────────────")
print("  The original code hardcoded N=61 and targeted ~20 records, which")
print("  does not match the actual dataset size (3900 rows) and produces a")
print("  sample too small for meaningful comparison against the random sample.")
print("  ─────────────────────────────────────────────────────────────────────")

df_reset = df.reset_index(drop=True)
k = max(1, len(df_reset) // n_sample)   
systematic_sample = df_reset.iloc[::k].copy()

print(f"\n  Step size (k)              : {k}")
print(f"  Systematic sample size     : {len(systematic_sample)} records")
print(f"  First 10 row indices       : {list(systematic_sample.index[:10])}")

sys_stats = descriptive_stats(systematic_sample, TARGET)
print(f"\n  Descriptive Statistics – Systematic Sample ('{TARGET}'):")
print(sys_stats.to_string())


# ══════════════════════════════════════════════════════════════════════════════
#  FULL DATASET DESCRIPTIVE STATISTICAL REPORT  (Task 9)
# ══════════════════════════════════════════════════════════════════════════════
subsection("Descriptive Statistical Report – Full Dataset (Task 9)")
full_stats = descriptive_stats(df, TARGET)

print(f"\n  ╔══════════════════════════════════════════════════════╗")
print(f"  ║   DESCRIPTIVE STATISTICS REPORT                     ║")
print(f"  ║   Variable : {TARGET:<38}║")
print(f"  ║   Dataset  : shopping_trends.csv                    ║")
print(f"  ╠══════════════════════════════════════════════════════╣")
for k_stat, v_stat in full_stats.items():
    print(f"  ║   {k_stat:<14}: {str(v_stat):<36}║")
print(f"  ╚══════════════════════════════════════════════════════╝")

skew_dir = ('right-skewed (positive tail)' if full_stats['Skewness'] > 0.5
            else 'left-skewed (negative tail)' if full_stats['Skewness'] < -0.5
            else 'approximately symmetric')
print(f"\n  Distribution shape: {skew_dir}  (Skewness = {full_stats['Skewness']})")


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALISATIONS  (Task 10)
# ══════════════════════════════════════════════════════════════════════════════
section("VISUALISATIONS (Task 10)")
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG,
                     'font.family': 'DejaVu Sans'})

# ── (a) SCATTER PLOT ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG)

sc_df = df[['Age', TARGET]].copy()
sc_df['Age']  = pd.to_numeric(sc_df['Age'],  errors='coerce')
sc_df[TARGET] = pd.to_numeric(sc_df[TARGET], errors='coerce')
sc_df = sc_df.dropna()

if 'Category' in df.columns:
    cats = df['Category'].dropna().unique()
    palette = plt.cm.tab10(np.linspace(0, 1, len(cats)))
    cat_col = dict(zip(cats, palette))
    for cat in cats:
        mask = df['Category'] == cat
        sub  = sc_df[mask]
        ax.scatter(sub['Age'], sub[TARGET],
                   color=cat_col[cat], alpha=0.55, s=30,
                   edgecolors='k', linewidths=0.2, label=cat)
    ax.legend(title='Category', bbox_to_anchor=(1.01, 1), loc='upper left',
              fontsize=8, title_fontsize=9)
else:
    ax.scatter(sc_df['Age'], sc_df[TARGET], color=C1, alpha=0.55, s=30,
               edgecolors='k', linewidths=0.2)

slope, intercept, r_val, p_val, _ = stats.linregress(sc_df['Age'], sc_df[TARGET])
x_line = np.linspace(sc_df['Age'].min(), sc_df['Age'].max(), 200)
ax.plot(x_line, slope * x_line + intercept, color=C2, linewidth=2.5,
        linestyle='--', label=f'Regression line  r={r_val:.3f}')

ax.set_xlabel('Age (years)', fontsize=12)
ax.set_ylabel('Purchase Amount (USD)', fontsize=12)
ax.set_title('(a) Scatter Plot: Age vs Purchase Amount (USD)\nColour = Product Category',
             fontsize=13, fontweight='bold', pad=12)
ax.grid(True, linestyle='--', alpha=0.35)
ax.tick_params(labelsize=10)
ax.annotate(f'Slope: {slope:.3f}\nIntercept: {intercept:.2f}\nr = {r_val:.3f}\np = {p_val:.4f}',
            xy=(0.02, 0.97), xycoords='axes fraction', fontsize=9,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='grey', alpha=0.8))
plt.tight_layout()
plt.savefig('scatter_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ Saved: scatter_plot.png")

# ── (b) BOX PLOT ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 2]})
fig.patch.set_facecolor(BG)

bp = axes[0].boxplot(df[TARGET].dropna(), patch_artist=True, widths=0.45,
                     boxprops=dict(facecolor='#FADADD', color=C2),
                     medianprops=dict(color=C2, linewidth=2.5),
                     whiskerprops=dict(color=C2, linewidth=1.5, linestyle='--'),
                     capprops=dict(color=C2, linewidth=2),
                     flierprops=dict(marker='D', color=C2, markersize=4,
                                     markerfacecolor='white', markeredgewidth=1))
q1m  = df[TARGET].quantile(0.25)
medm = df[TARGET].median()
q3m  = df[TARGET].quantile(0.75)
for val, label in [(q1m, f'Q1=${q1m:.0f}'), (medm, f'Med=${medm:.0f}'), (q3m, f'Q3=${q3m:.0f}')]:
    axes[0].axhline(val, color='grey', linestyle=':', linewidth=0.8)
    axes[0].text(1.35, val, label, va='center', fontsize=9, color='#555')
axes[0].set_xticks([1])
axes[0].set_xticklabels(['All Customers'])
axes[0].set_ylabel('Purchase Amount (USD)', fontsize=11)
axes[0].set_title('Full Dataset', fontsize=11, fontweight='bold')
axes[0].grid(True, axis='y', linestyle='--', alpha=0.4)

if 'Category' in df.columns:
    cat_order = df.groupby('Category')[TARGET].median().sort_values().index.tolist()
    data_by_cat = [df[df['Category'] == c][TARGET].dropna().values for c in cat_order]
    bp2 = axes[1].boxplot(data_by_cat, patch_artist=True, widths=0.55,
                           medianprops=dict(color='white', linewidth=2),
                           whiskerprops=dict(color='grey', linewidth=1.2),
                           capprops=dict(color='grey', linewidth=1.5),
                           flierprops=dict(marker='o', markersize=4,
                                           markerfacecolor='white', markeredgewidth=1))
    colours_box = [C1, C4, C3, C2, '#2ECC71']
    for patch, colour in zip(bp2['boxes'], colours_box):
        patch.set_facecolor(colour)
        patch.set_alpha(0.85)
    axes[1].set_xticks(range(1, len(cat_order)+1))
    axes[1].set_xticklabels(cat_order, fontsize=10, rotation=15)
    axes[1].set_title('By Product Category', fontsize=11, fontweight='bold')
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.4)

fig.suptitle('(b) Box Plot: Distribution of Purchase Amount (USD)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('box_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ Saved: box_plot.png")

# ── (c) HISTOGRAM ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)

data_hist = df[TARGET].dropna()
n_bins = 15
axes[0].hist(data_hist, bins=n_bins, color=C3, edgecolor='white',
             alpha=0.82, density=False, label='Frequency')
ax2 = axes[0].twinx()
from scipy.stats import gaussian_kde
kde = gaussian_kde(data_hist)
x_kde = np.linspace(data_hist.min(), data_hist.max(), 300)
ax2.plot(x_kde, kde(x_kde), color=C1, linewidth=2.5, label='KDE')
ax2.set_ylabel('Density', fontsize=10, color=C1)
ax2.tick_params(axis='y', colors=C1)

mean_v = data_hist.mean()
med_v  = data_hist.median()
axes[0].axvline(mean_v, color=C2,  linestyle='--', linewidth=2,
                label=f'Mean ${mean_v:.1f}')
axes[0].axvline(med_v,  color=C4,  linestyle='-',  linewidth=2,
                label=f'Median ${med_v:.1f}')
axes[0].set_xlabel('Purchase Amount (USD)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Frequency Histogram + KDE', fontsize=11, fontweight='bold')
lines1, labels1 = axes[0].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[0].legend(lines1 + lines2, labels1 + labels2, fontsize=9)
axes[0].grid(True, axis='y', linestyle='--', alpha=0.35)

if 'Season' in df.columns:
    seasons = df['Season'].dropna().unique()
    pal_s = [C1, C2, C4, C3]
    for i, season in enumerate(seasons):
        sub_s = df[df['Season'] == season][TARGET].dropna()
        axes[1].hist(sub_s, bins=10, color=pal_s[i % 4],
                     alpha=0.55, edgecolor='white', label=season)
    axes[1].set_xlabel('Purchase Amount (USD)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Histogram by Season', fontsize=11, fontweight='bold')
    axes[1].legend(title='Season', fontsize=9)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.35)

fig.suptitle('(c) Histogram: Distribution of Purchase Amount (USD)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('histogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ Saved: histogram.png")

# ── (d) HEAT MAP ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG)

numeric_df  = df.select_dtypes(include=[np.number])
corr        = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.6, ax=axes[0], cbar_kws={'shrink': 0.85},
            annot_kws={'size': 11, 'weight': 'bold'},
            vmin=-1, vmax=1)
axes[0].set_title('Numeric Feature Correlation Matrix', fontsize=11, fontweight='bold')
axes[0].tick_params(labelsize=9)

if 'Category' in df.columns and 'Season' in df.columns:
    pivot = df.pivot_table(values=TARGET, index='Category',
                           columns='Season', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                linewidths=0.5, ax=axes[1], cbar_kws={'shrink': 0.85},
                annot_kws={'size': 10})
    axes[1].set_title('Mean Purchase Amount\n(Category × Season)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Season', fontsize=10)
    axes[1].set_ylabel('Category', fontsize=10)
    axes[1].tick_params(labelsize=9)

fig.suptitle('(d) Heat Maps: Correlations & Category–Season Analysis',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✔ Saved: heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
#  HYPOTHESIS TESTS  (Task 11)
# ══════════════════════════════════════════════════════════════════════════════
section("HYPOTHESIS TESTS (Task 11)")

# ── A. Pearson Correlation ─────────────────────────────────────────────────────
subsection("A. Pearson Correlation – Age vs Purchase Amount (USD)")
clean = df[['Age', TARGET]].dropna()
clean = clean.copy()
clean['Age']  = pd.to_numeric(clean['Age'],  errors='coerce')
clean[TARGET] = pd.to_numeric(clean[TARGET], errors='coerce')
clean = clean.dropna()

pearson_r, pearson_p = stats.pearsonr(clean['Age'], clean[TARGET])
print(f"""
  H₀ : No linear correlation between Age and Purchase Amount (r = 0)
  H₁ : Linear correlation exists (r ≠ 0)
  α  = 0.05

  Pearson r  = {pearson_r:.4f}
  p-value    = {pearson_p:.4f}
  Decision   : {"✘ Reject H₀ – significant linear correlation." if pearson_p < 0.05
                else "✔ Fail to reject H₀ – no significant linear correlation detected."}
  Interpretation : A Pearson r of {pearson_r:.4f} indicates a
    {'strong' if abs(pearson_r) > 0.7 else 'moderate' if abs(pearson_r) > 0.4 else 'weak'}
    {'positive' if pearson_r > 0 else 'negative'} linear relationship between Age and Purchase Amount.
""")

# ── B. Spearman Correlation ────────────────────────────────────────────────────
subsection("B. Spearman Correlation – Age vs Purchase Amount (USD)")
spearman_r, spearman_p = stats.spearmanr(clean['Age'], clean[TARGET])
print(f"""
  H₀ : No monotonic correlation between Age and Purchase Amount (ρ = 0)
  H₁ : Monotonic correlation exists (ρ ≠ 0)
  α  = 0.05

  Spearman ρ = {spearman_r:.4f}
  p-value    = {spearman_p:.4f}
  Decision   : {"✘ Reject H₀ – significant monotonic correlation." if spearman_p < 0.05
                else "✔ Fail to reject H₀ – no significant monotonic correlation detected."}
  Interpretation : Spearman is more robust to outliers than Pearson. A ρ of {spearman_r:.4f}
    confirms the {'same direction' if np.sign(spearman_r)==np.sign(pearson_r) else 'different direction'}
    as the Pearson result.
""")

# ══════════════════════════════════════════════════════════════════════════════
#  FIX 3 ─ CHI-SQUARE TEST  (Task 11C)
# ══════════════════════════════════════════════════════════════════════════════
subsection("C. Chi-Square Test – Category vs Purchase Amount Bins (Fix 3)")

print("  ── WHAT WAS WRONG IN THE ORIGINAL CODE ─────────────────────────────")
print("  1. pd.cut with bins=3 on a ~uniform distribution creates empty or")
print("     extremely unbalanced cells in the contingency table.")
print("  2. No check for expected cell frequencies ≥ 5 (Chi-Square assumption).")
print("  3. Temporary 'Purchase_Bin' column added directly to df without a copy,")
print("     risking data contamination if an error occurs mid-test.")
print("  ─────────────────────────────────────────────────────────────────────")
print()
print("  ── FIX APPLIED ──────────────────────────────────────────────────────")
print("  • Use pd.qcut (quantile-based) → guarantees balanced bin frequencies.")
print("  • Validate expected frequencies before accepting test results.")
print("  • Work on a local copy (df_chi) so df is never modified.")
print("  ─────────────────────────────────────────────────────────────────────")


df_chi = df[['Category', TARGET]].dropna().copy()

df_chi['Purchase_Bin'] = pd.qcut(df_chi[TARGET], q=3,
                                  labels=['Low', 'Medium', 'High'])

contingency = pd.crosstab(df_chi['Category'], df_chi['Purchase_Bin'])
print(f"\n  Contingency Table (rows=Category, cols=Purchase Bin):")
print(contingency.to_string())

chi2_stat, chi_p, dof, expected = stats.chi2_contingency(contingency)


min_expected = expected.min()
cells_below_5 = (expected < 5).sum()
print(f"\n  Expected Frequency Check:")
print(f"    Minimum expected cell frequency : {min_expected:.2f}")
print(f"    Cells with expected count < 5   : {cells_below_5}")
if cells_below_5 > 0:
    print("    ⚠ WARNING: Chi-Square assumption violated. Consider merging categories")
    print("      or using Fisher's Exact Test for small samples.")
else:
    print("    ✔ All expected cell frequencies ≥ 5. Chi-Square assumption satisfied.")

print(f"""
  H₀ : Category and Purchase Amount level are independent.
  H₁ : Category and Purchase Amount level are NOT independent (associated).
  α  = 0.05

  Chi-Square Statistic (χ²) = {chi2_stat:.4f}
  Degrees of Freedom (df)   = {dof}
  p-value                   = {chi_p:.4f}
  Decision : {"✘ Reject H₀ – Category and Purchase Amount are significantly associated." if chi_p < 0.05
              else "✔ Fail to reject H₀ – No significant association detected."}

  Interpretation:
    {'A significant association exists between product Category and purchase'
     if chi_p < 0.05 else 'No significant association detected between Category and purchase'}
    amount level. {'This suggests different product categories attract customers'
                   ' with different spending levels.' if chi_p < 0.05
                   else 'Purchase amounts appear consistent across categories.'}
""")


# ══════════════════════════════════════════════════════════════════════════════
#  ONE-SAMPLE T-TEST  (Task 12)
# ══════════════════════════════════════════════════════════════════════════════
section("ONE-SAMPLE T-TEST (Task 12)")

pop_mean    = df[TARGET].mean()
sample_data = random_sample[TARGET].dropna()
sample_mean = sample_data.mean()
sample_std  = sample_data.std()
sample_n    = len(sample_data)
t_stat, t_p = stats.ttest_1samp(sample_data, popmean=pop_mean)
df_t        = sample_n - 1

ci_margin = stats.t.ppf(0.975, df=df_t) * (sample_std / np.sqrt(sample_n))
ci_lower  = sample_mean - ci_margin
ci_upper  = sample_mean + ci_margin

print(f"""
  Objective:
    Assess whether the random sample (n={sample_n}) is representative of the
    full population for the variable '{TARGET}'.

  ┌──────────────────────────────────────────────────────┐
  │  Population Mean  (µ₀)   : ${pop_mean:.4f}              │
  │  Sample Mean      (x̄)    : ${sample_mean:.4f}              │
  │  Sample Std Dev   (s)    : ${sample_std:.4f}              │
  │  Sample Size      (n)    : {sample_n:<6}                   │
  │  Degrees of Freedom      : {df_t:<6}                   │
  └──────────────────────────────────────────────────────┘

  H₀ (Null)    : x̄ = µ₀  (sample mean = population mean = ${pop_mean:.2f})
  H₁ (Alt.)    : x̄ ≠ µ₀  (two-tailed test)
  Significance : α = 0.05

  t-statistic     = {t_stat:.4f}
  p-value         = {t_p:.4f}
  95% CI for mean = [${ci_lower:.2f} , ${ci_upper:.2f}]

  Decision : {"✘ Reject H₀ – sample mean differs significantly from population mean." if t_p < 0.05
               else "✔ Fail to reject H₀ – sample mean is NOT significantly different from population mean."}

  Interpretation:
    The one-sample t-test compares the random sample mean (${sample_mean:.2f})
    against the population mean (${pop_mean:.2f}).
    t({df_t}) = {t_stat:.4f},  p = {t_p:.4f}

    Since p = {t_p:.4f} {'< 0.05, we reject H₀.' if t_p < 0.05 else '> 0.05, we fail to reject H₀.'}
    The 95% confidence interval [${ci_lower:.2f}, ${ci_upper:.2f}]
    {'does NOT contain' if (ci_lower > pop_mean or ci_upper < pop_mean) else 'contains'}
    the population mean (${pop_mean:.2f}).

    Conclusion: The random sampling method {'successfully produced' if t_p >= 0.05 else 'did not produce'}
    a representative sample of the customer population.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  SAMPLING COMPARISON CHART
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(BG)
fig.suptitle('Sampling Comparison: Full Dataset vs Random vs Systematic',
             fontsize=13, fontweight='bold')

labels = ['Full Dataset', 'Random\nSample (n=150)', 'Systematic\nSample (n≈150)']
means  = [full_stats['Mean'], random_stats['Mean'], sys_stats['Mean']]
stds   = [full_stats['Std Dev'], random_stats['Std Dev'], sys_stats['Std Dev']]
meds   = [full_stats['Median'], random_stats['Median'], sys_stats['Median']]

bars = axes[0].bar(labels, means, color=[C1, C2, C4], edgecolor='white',
                   alpha=0.85, yerr=stds, capsize=6, error_kw={'linewidth':1.5})
axes[0].set_title('Mean ± Std Dev', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Purchase Amount (USD)')
axes[0].grid(axis='y', linestyle='--', alpha=0.4)
for bar, val in zip(bars, means):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'${val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

bars2 = axes[1].bar(labels, meds, color=[C3, C1, C2], edgecolor='white', alpha=0.85)
axes[1].set_title('Median', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Purchase Amount (USD)')
axes[1].grid(axis='y', linestyle='--', alpha=0.4)
for bar, val in zip(bars2, meds):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'${val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

all_data  = df[TARGET].dropna().values
rand_data = random_sample[TARGET].dropna().values
sys_data  = systematic_sample[TARGET].dropna().values
bp3 = axes[2].boxplot([all_data, rand_data, sys_data],
                       labels=['Full', 'Random', 'Systematic'], patch_artist=True,
                       medianprops=dict(color='white', linewidth=2),
                       whiskerprops=dict(linewidth=1.2),
                       capprops=dict(linewidth=1.5))
for patch, colour in zip(bp3['boxes'], [C1, C2, C4]):
    patch.set_facecolor(colour)
    patch.set_alpha(0.8)
axes[2].set_title('Distribution Comparison', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Purchase Amount (USD)')
axes[2].grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('sampling_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  ✔ Saved: sampling_comparison.png")



print("\n" + "═"*70)
print("  ✔ ALL TASKS COMPLETE")
print("  Charts generated:")
print("    → scatter_plot.png")
print("    → box_plot.png")
print("    → histogram.png")
print("    → heatmap.png")
print("    → sampling_comparison.png")
print("═"*70)