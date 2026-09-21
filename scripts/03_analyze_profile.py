import pandas as pd
import numpy as np
import os

def main():
    print("Starting profiling pipeline...")
    
    PREPARED_CSV_PATH = 'data/processed/bengaluru_resale_real_prepared.csv'
    PROFILE_CSV_PATH = 'data/processed/synthetic_generation_profile.csv'
    CORR_CSV_PATH = 'data/processed/correlation_matrix.csv'
    SUMMARY_PATH = 'data/processed/synthetic_profile_summary.txt'
    
    try:
        df = pd.read_csv(PREPARED_CSV_PATH, low_memory=False)
    except FileNotFoundError:
        print(f"Error: {PREPARED_CSV_PATH} not found.")
        return

    num_vars = ['price', 'area_sqft', 'bhk', 'bathrooms', 'parking', 'latitude', 'longitude', 'floor_number', 'total_floors', 'age_years']
    cat_vars = ['location', 'city', 'furnishing', 'facing', 'possession_category', 'is_under_construction']

    # Temporarily calculate price_per_sqft for analysis
    df_analysis = df.copy()
    df_analysis['price_per_sqft'] = df_analysis['price'] / df_analysis['area_sqft']
    df_analysis.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 1. Numerical summary
    num_summary = []
    for col in num_vars:
        if col in df.columns:
            s = df[col]
            num_summary.append({
                'Variable': col,
                'Type': 'Numerical',
                'Count': s.notna().sum(),
                'Missing': s.isna().sum(),
                'Mean': s.mean(),
                'Median': s.median(),
                'Std': s.std(),
                'Min': s.min(),
                'Max': s.max(),
                'Q1': s.quantile(0.25),
                'Q3': s.quantile(0.75)
            })

    # 2. Categorical summary (We will just save a condensed version in profile csv, detailed in text)
    cat_summary = []
    cat_details = ""
    for col in cat_vars:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False)
            pct = df[col].value_counts(dropna=False, normalize=True) * 100
            
            cat_summary.append({
                'Variable': col,
                'Type': 'Categorical',
                'Count': df[col].notna().sum(),
                'Missing': df[col].isna().sum(),
                'Unique_Values': df[col].nunique(),
                'Top_Value': df[col].mode()[0] if not df[col].mode().empty else np.nan,
            })
            
            cat_details += f"\\n--- {col} ---\\n"
            for val, count in vc.items():
                cat_details += f"{val}: {count} ({pct[val]:.2f}%)\\n"

    # Combine profiles
    profile_df = pd.DataFrame(num_summary + cat_summary)
    profile_df.to_csv(PROFILE_CSV_PATH, index=False)

    # 3. Pearson Correlation
    corr_df = df_analysis[num_vars].corr(method='pearson')
    corr_df.to_csv(CORR_CSV_PATH)
    
    # Extract important correlations
    imp_corrs = []
    if 'bhk' in corr_df and 'area_sqft' in corr_df: imp_corrs.append(f"BHK vs area_sqft: {corr_df.loc['bhk', 'area_sqft']:.3f}")
    if 'bhk' in corr_df and 'bathrooms' in corr_df: imp_corrs.append(f"BHK vs bathrooms: {corr_df.loc['bhk', 'bathrooms']:.3f}")
    if 'bhk' in corr_df and 'parking' in corr_df: imp_corrs.append(f"BHK vs parking: {corr_df.loc['bhk', 'parking']:.3f}")
    if 'area_sqft' in corr_df and 'price' in corr_df: imp_corrs.append(f"area_sqft vs price: {corr_df.loc['area_sqft', 'price']:.3f}")
    if 'bhk' in corr_df and 'price' in corr_df: imp_corrs.append(f"BHK vs price: {corr_df.loc['bhk', 'price']:.3f}")
    if 'age_years' in corr_df and 'price' in corr_df: imp_corrs.append(f"age_years vs price: {corr_df.loc['age_years', 'price']:.3f}")

    # Categorical vs Numeric median prices
    furn_price = df_analysis.groupby('furnishing')['price'].median().to_dict()
    uc_price = df_analysis.groupby('is_under_construction')['price'].median().to_dict()

    # 5. Location analysis
    loc_analysis = ""
    if 'location' in df_analysis.columns:
        n_unique_loc = df_analysis['location'].nunique()
        loc_counts = df_analysis['location'].value_counts()
        loc_pct = df_analysis['location'].value_counts(normalize=True) * 100
        loc_med_price = df_analysis.groupby('location')['price'].median()
        loc_med_psqft = df_analysis.groupby('location')['price_per_sqft'].median()
        
        top20_locs = loc_counts.head(20).index
        loc_analysis += f"Unique locations: {n_unique_loc}\\n\\nTop 20 Locations:\\n"
        for loc in top20_locs:
            loc_analysis += f"{loc}: {loc_counts[loc]} records ({loc_pct[loc]:.2f}%) - Median Price: {loc_med_price[loc]:.0f} - Median Price/Sqft: {loc_med_psqft[loc]:.2f}\\n"
            
    # 6. Outliers (IQR)
    outliers_report = ""
    for col in ['price', 'area_sqft', 'bhk', 'bathrooms', 'parking', 'age_years']:
        if col in df.columns:
            s = df[col].dropna()
            Q1 = s.quantile(0.25)
            Q3 = s.quantile(0.75)
            IQR = Q3 - Q1
            out_count = ((s < (Q1 - 1.5 * IQR)) | (s > (Q3 + 1.5 * IQR))).sum()
            outliers_report += f"- {col}: {out_count} potential outliers\\n"

    # 7. Consistency Checks
    consist_report = ""
    if 'floor_number' in df.columns and 'total_floors' in df.columns:
        fails = df[(df['floor_number'] > df['total_floors']) & df['floor_number'].notna() & df['total_floors'].notna()].shape[0]
        consist_report += f"- floor_number <= total_floors: {fails} violations\\n"
    if 'floor_number' in df.columns:
        # Ground floor = 0, upper basement = -1. So >= -3 is typically fine. Let's check < 0 but not -1,-2
        fails = df[(df['floor_number'] < -3) & df['floor_number'].notna()].shape[0]
        consist_report += f"- floor_number >= 0 (or valid basement): {fails} violations\\n"
    if 'age_years' in df.columns:
        fails = df[(df['age_years'] < 0) & df['age_years'].notna()].shape[0]
        consist_report += f"- age_years >= 0: {fails} violations\\n"
    if 'price' in df.columns:
        fails = df[(df['price'] <= 0) & df['price'].notna()].shape[0]
        consist_report += f"- price > 0: {fails} violations\\n"
    if 'area_sqft' in df.columns:
        fails = df[(df['area_sqft'] <= 0) & df['area_sqft'].notna()].shape[0]
        consist_report += f"- area_sqft > 0: {fails} violations\\n"
    if 'bhk' in df.columns:
        fails = df[(df['bhk'] <= 0) & df['bhk'].notna()].shape[0]
        consist_report += f"- bhk > 0: {fails} violations\\n"
    if 'bathrooms' in df.columns:
        fails = df[(df['bathrooms'] < 0) & df['bathrooms'].notna()].shape[0]
        consist_report += f"- bathrooms >= 0: {fails} violations\\n"
    if 'parking' in df.columns:
        fails = df[(df['parking'] < 0) & df['parking'].notna()].shape[0]
        consist_report += f"- parking >= 0: {fails} violations\\n"

    # 8. Synthetic Generation Recommendations
    recommendations = """
Synthetic Generation Recommendations:
- `location`: Empirical/Categorical distribution. Highly skewed, must preserve Top 20 distribution.
- `bhk`: Empirical/Categorical distribution. Should act as a conditional base for area and bathrooms.
- `area_sqft`: Conditional numeric distribution dependent on `bhk`.
- `bathrooms`: Conditional distribution dependent on `bhk`.
- `parking`: Conditional distribution dependent on `bhk` and `area_sqft`.
- `price`: Conditional numeric distribution heavily dependent on `area_sqft` and `location`. (Maintain location ↔ price_per_sqft relationship).
- `age_years`: Numeric distribution. Mildly influences price. Missing-value pattern should be preserved (MCAR or MAR).
- `furnishing`: Conditional categorical distribution.
- `facing`: Categorical distribution.
- `floor_number` & `total_floors`: Coupled numeric distribution ensuring `floor_number <= total_floors`.

Variables that should NOT be synthesized (Keep as reference or drop):
- `listingId`, `title`, `project_name`, `propertyType`, `saleTag`, `data_source`, `posted_date`, `possession_date`
"""

    total_records = len(df)
    perc_real = 100.0 # currently 100%

    report = f"""Synthetic Profile Summary
==================================================
Total records: {total_records}
Percentage real data: {perc_real}%

Missing-Value Patterns (from summary logic above):
[Check synthetic_generation_profile.csv for exact counts]

Important Correlations:
{chr(10).join(imp_corrs)}

Categorical Price Medians:
- Furnishing: {furn_price}
- Under Construction: {uc_price}

Location Analysis:
{loc_analysis}

Potential Outliers (IQR Method):
{outliers_report}

Consistency Checks:
{consist_report}

{recommendations}

Note: Do NOT generate synthetic records yet.
"""
    with open(SUMMARY_PATH, 'w') as f:
        f.write(report)
        
    print("Profiling pipeline completed.")

if __name__ == "__main__":
    main()
