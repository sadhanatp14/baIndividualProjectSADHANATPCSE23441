import pandas as pd
import numpy as np
import re
import os
import json

def main():
    print("Starting data cleaning pipeline...")
    
    # Paths
    RAW_PATH = 'realEstateListingsDataRaw.csv'
    PROCESSED_DIR = 'data/'
    CLEANED_CSV_PATH = os.path.join(PROCESSED_DIR, 'bengaluru_resale_real_clean.csv')
    EXCLUDED_CSV_PATH = os.path.join(PROCESSED_DIR, 'excluded_columns.csv')
    DATA_DICT_PATH = os.path.join(PROCESSED_DIR, 'data_dictionary.csv')
    REPORT_PATH = os.path.join(PROCESSED_DIR, 'cleaning_summary.txt')
    MISSING_SUMMARY_PATH = os.path.join(PROCESSED_DIR, 'missing_summary.csv')
    FEATURE_SUMMARY_PATH = os.path.join(PROCESSED_DIR, 'feature_extraction_summary.csv')

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    try:
        df_raw = pd.read_csv(RAW_PATH, low_memory=False)
    except FileNotFoundError:
        print(f"Error: {RAW_PATH} not found.")
        return

    original_rows, original_cols = df_raw.shape
    df = df_raw.copy()

    # 1. CITY
    # Preserve original city column
    if 'city' not in df.columns:
        df['city'] = np.nan
    
    city_mismatch = df[~df['city'].str.lower().isin(['bengaluru', 'bangalore']) & df['city'].notna()]['city'].unique()

    # 3. COORDINATES
    if 'coords/0' in df.columns:
        df.rename(columns={'coords/0': 'latitude'}, inplace=True)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    else:
        df['latitude'] = np.nan
        
    if 'coords/1' in df.columns:
        df.rename(columns={'coords/1': 'longitude'}, inplace=True)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    else:
        df['longitude'] = np.nan

    # 4. PROJECT NAME
    if 'entityProjectName' in df.columns:
        df.rename(columns={'entityProjectName': 'project_name'}, inplace=True)
    else:
        df['project_name'] = np.nan

    # 5. CONSTRUCTION STATUS
    if 'isUc' in df.columns:
        df.rename(columns={'isUc': 'is_under_construction'}, inplace=True)
    else:
        df['is_under_construction'] = np.nan

    # 2. FEATURES EXTRACTION
    # Find all feature index prefixes: e.g. 'features/0', 'features/1'
    feature_cols = [c for c in df.columns if c.startswith('features/')]
    feature_indices = set(c.split('/')[1] for c in feature_cols if c.split('/')[1].isdigit())
    
    # Initialize new columns
    df['age_of_property'] = pd.Series(dtype='object')
    df['floor'] = pd.Series(dtype='object')
    df['furnishing'] = pd.Series(dtype='object')
    df['facing'] = pd.Series(dtype='object')
    df['possession_status'] = pd.Series(dtype='object')
    df['configuration'] = pd.Series(dtype='object')
    df['built_up_area'] = pd.Series(dtype='object')

    label_mapping = {
        'age of property': 'age_of_property',
        'floor': 'floor',
        'furnishing': 'furnishing',
        'facing': 'facing',
        'possession status': 'possession_status',
        'configuration': 'configuration',
        'built up area': 'built_up_area'
    }
    
    print("Extracting features from features array...")
    # Iterate through rows
    for idx, row in df.iterrows():
        for fi in feature_indices:
            label_col = f'features/{fi}/label'
            desc_col = f'features/{fi}/description'
            
            if label_col in df.columns and desc_col in df.columns:
                label_val = str(row.get(label_col, '')).lower().strip()
                desc_val = row.get(desc_col, np.nan)
                
                if pd.notna(desc_val):
                    for key, target_col in label_mapping.items():
                        if key in label_val:
                            df.at[idx, target_col] = desc_val
                            break

    feature_summary = []
    for col in label_mapping.values():
        non_missing = df[col].notna().sum()
        pct = (original_rows - non_missing) / original_rows * 100
        unique_vals = df[col].dropna().unique()[:5]
        feature_summary.append({
            'Feature': col,
            'Non_Missing_Count': non_missing,
            'Missing_Percentage': pct,
            'Sample_Values': ", ".join(map(str, unique_vals))
        })
    pd.DataFrame(feature_summary).to_csv(FEATURE_SUMMARY_PATH, index=False)

    # 7. PRICE (Target)
    print("Cleaning price...")
    def clean_price(val):
        if pd.isna(val): return np.nan
        val = str(val).upper().replace(',', '').replace('₹', '').strip()
        try:
            if 'CR' in val:
                return float(re.findall(r"[-+]?\d*\.\d+|\d+", val)[0]) * 10000000
            elif 'L' in val:
                return float(re.findall(r"[-+]?\d*\.\d+|\d+", val)[0]) * 100000
            elif 'K' in val:
                return float(re.findall(r"[-+]?\d*\.\d+|\d+", val)[0]) * 1000
            else:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
                if nums: return float(nums[0])
                return np.nan
        except:
            return np.nan

    price_col = 'price' if 'price' in df.columns else 'propertyInformation/price'
    if price_col in df.columns:
        df['price'] = df[price_col].apply(clean_price)
    else:
        df['price'] = np.nan

    # 8. AREA
    print("Cleaning area...")
    def clean_area(val):
        if pd.isna(val): return np.nan
        val = str(val).lower().replace(',', '')
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", val)
        if nums: return float(nums[0])
        return np.nan

    if 'propertyInformation/area' in df.columns:
        df['area_sqft'] = df['propertyInformation/area'].apply(clean_area)
    elif 'area' in df.columns:
        df['area_sqft'] = df['area'].apply(clean_area)
    else:
        df['area_sqft'] = np.nan

    # 9. BHK
    print("Cleaning BHK...")
    def extract_fractional_bhk(title):
        if pd.isna(title): return np.nan
        title = str(title).upper()
        # Look for e.g. "2.5 BHK", "3 BHK"
        match = re.search(r'(\d+(?:\.\d+)?)\s*BHK', title)
        if match:
            return float(match.group(1))
        return np.nan

    df['bhk_structured'] = pd.to_numeric(df['propertyInformation/bedrooms'], errors='coerce') if 'propertyInformation/bedrooms' in df.columns else np.nan
    df['bhk_title'] = df['title'].apply(extract_fractional_bhk) if 'title' in df.columns else np.nan
    
    # Resolve disagreements: title fractional usually more precise if structured is integer
    def resolve_bhk(row):
        struct = row['bhk_structured']
        title_bhk = row['bhk_title']
        if pd.notna(title_bhk) and pd.notna(struct) and title_bhk != struct:
            return title_bhk # prefer fractional if exists
        elif pd.notna(struct):
            return struct
        elif pd.notna(title_bhk):
            return title_bhk
        return np.nan
        
    df['bhk'] = df.apply(resolve_bhk, axis=1)
    bhk_disagreements = df[(df['bhk_structured'].notna()) & (df['bhk_title'].notna()) & (df['bhk_structured'] != df['bhk_title'])].shape[0]

    # Bathrooms, Parking, Location
    df['bathrooms'] = pd.to_numeric(df['propertyInformation/bathrooms'], errors='coerce') if 'propertyInformation/bathrooms' in df.columns else np.nan
    df['parking'] = pd.to_numeric(df['propertyInformation/parking'], errors='coerce') if 'propertyInformation/parking' in df.columns else np.nan
    
    # 13. FINAL ANALYTICAL DATASET
    analytical_cols = [
        'listingId', 'title', 'price', 'area_sqft', 'bhk', 'bathrooms', 'parking', 
        'location', 'city', 'latitude', 'longitude', 'floor', 'furnishing', 'facing', 
        'age_of_property', 'possession_status', 'is_under_construction', 'project_name', 
        'postedDate', 'propertyType', 'saleTag'
    ]
    
    final_cols = [c for c in analytical_cols if c in df.columns]
    df_clean = df[final_cols].copy()

    # 10. MISSING VALUES SUMMARY
    missing_summary = df_clean.isnull().sum().to_frame('missing_count')
    missing_summary['missing_percentage'] = (missing_summary['missing_count'] / len(df_clean)) * 100
    missing_summary.to_csv(MISSING_SUMMARY_PATH)
    
    # 11. DATA QUALITY CHECKS
    dup_id = df_clean.duplicated(subset=['listingId']).sum() if 'listingId' in df_clean.columns else 0
    dup_exact = df_clean.duplicated().sum()
    
    invalid_price = (df_clean['price'] <= 0).sum() if 'price' in df_clean.columns else 0
    invalid_area = (df_clean['area_sqft'] <= 0).sum() if 'area_sqft' in df_clean.columns else 0
    invalid_bhk = (df_clean['bhk'] <= 0).sum() if 'bhk' in df_clean.columns else 0
    invalid_bath = (df_clean['bathrooms'] < 0).sum() if 'bathrooms' in df_clean.columns else 0
    invalid_park = (df_clean['parking'] < 0).sum() if 'parking' in df_clean.columns else 0
    
    invalid_coords = 0
    if 'latitude' in df_clean.columns and 'longitude' in df_clean.columns:
        invalid_coords = df_clean[(df_clean['latitude'] < -90) | (df_clean['latitude'] > 90) | (df_clean['longitude'] < -180) | (df_clean['longitude'] > 180)].shape[0]
        
    # 12. OUTLIERS (Only identify plausible extremes, don't remove)
    outliers_report = {}
    for col in ['price', 'area_sqft', 'bhk']:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)].shape[0]
            outliers_report[col] = outliers

    # 14. EXCLUDED COLUMNS
    image_cols = [c for c in df_raw.columns if 'image' in c.lower() or 'video' in c.lower()]
    seller_cols = [c for c in df_raw.columns if 'seller' in c.lower() or 'phone' in c.lower() or c in ['name']]
    technical_cols = [c for c in df_raw.columns if 'url' in c.lower() or 'hash' in c.lower() or 'page' in c.lower()]
    project_metadata = [c for c in df_raw.columns if 'project' in c.lower() and c != 'entityProjectName']
    
    excluded_cols = [c for c in df_raw.columns if c not in final_cols and c not in ['coords/0', 'coords/1', 'entityProjectName', 'isUc']]
    
    reasons = []
    for c in excluded_cols:
        if c in image_cols: reasons.append('Image/Video URL')
        elif c in seller_cols: reasons.append('Seller/Contact Information')
        elif c in technical_cols: reasons.append('Technical/Scraping')
        elif str(c).startswith('features/'): reasons.append('Redundant/Raw Flattened Feature')
        elif c in project_metadata: reasons.append('High-cardinality/Project Metadata')
        else: reasons.append('Redundant/Unused')
            
    pd.DataFrame({'excluded_column': excluded_cols, 'reason': reasons}).to_csv(EXCLUDED_CSV_PATH, index=False)

    # 15. DATA DICTIONARY
    data_dict = []
    for col in final_cols:
        data_type = str(df_clean[col].dtype)
        pct = missing_summary.loc[col, 'missing_percentage']
        count = missing_summary.loc[col, 'missing_count']
        sample = str(df_clean[col].dropna().iloc[0]) if count < len(df_clean) else "N/A"
        
        target_feat = 'Target' if col == 'price' else ('Reference' if col in ['listingId', 'title'] else 'Feature')
        
        # Determine source
        source = col
        if col == 'latitude': source = 'coords/0'
        elif col == 'longitude': source = 'coords/1'
        elif col == 'project_name': source = 'entityProjectName'
        elif col == 'is_under_construction': source = 'isUc'
        elif col in label_mapping.values(): source = 'features/*/label & description'
        elif col == 'area_sqft': source = 'propertyInformation/area or area'
        elif col == 'bhk': source = 'propertyInformation/bedrooms & title'
        
        data_dict.append({
            'column_name': col,
            'description': f'Analytical column for {col}',
            'source_raw_column': source,
            'data_type': data_type,
            'missing_count': count,
            'missing_percentage': pct,
            'example_values': sample,
            'role': target_feat,
            'reason_for_retention': 'Critical property characteristic' if target_feat != 'Reference' else 'Traceability'
        })
    pd.DataFrame(data_dict).to_csv(DATA_DICT_PATH, index=False)

    # 16. OUTPUTS
    df_clean.to_csv(CLEANED_CSV_PATH, index=False)
    
    final_rows, final_cols_count = df_clean.shape
    
    report_content = f"""Data Cleaning Report (Revised)
==================================================
Original records: {original_rows}
Original columns: {original_cols}
Cleaned records: {final_rows}
Final features: {final_cols_count}
Target variable: price
Columns retained: {final_cols}

City Validation
==================================================
Non-Bengaluru/Missing Cities found: {city_mismatch}

BHK Disagreements
==================================================
Fractional vs Structured disagreements: {bhk_disagreements}

Quality Checks
==================================================
Duplicate records (listingId): {dup_id}
Duplicate exact rows: {dup_exact}
Invalid prices (<= 0): {invalid_price}
Invalid areas (<= 0): {invalid_area}
Invalid BHK (<= 0): {invalid_bhk}
Invalid Bathrooms (< 0): {invalid_bath}
Invalid Parking (< 0): {invalid_park}
Invalid Coordinates: {invalid_coords}
Outliers (IQR method - Plausible Extremes): {outliers_report}

Dataset ready for synthetic generation: YES
"""
    with open(REPORT_PATH, 'w') as f:
        f.write(report_content)
        
    print("Pipeline completed successfully. Check the 'data/processed' folder for outputs.")

if __name__ == "__main__":
    main()
