import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

REAL_PATH = DATA_DIR / "processed/bengaluru_resale_real_prepared.csv"
SYNTHETIC_PATH = DATA_DIR / "synthetic/controlled_synthetic_5001.csv"
OUTPUT_PATH = DATA_DIR / "bengaluru_resale_10000.csv"

SEED = 42

# =========================================================
# 1. Load datasets
# =========================================================

real = pd.read_csv(REAL_PATH)
synthetic = pd.read_csv(SYNTHETIC_PATH)

print("=" * 70)
print("FINAL DATASET CREATION")
print("=" * 70)

print("\nInput shapes:")
print("Real:", real.shape)
print("Synthetic:", synthetic.shape)


# =========================================================
# 2. Standardize ID column
# =========================================================

if "listingId" in real.columns:
    real["listingId"] = "REAL_" + real["listingId"].astype(str)

synthetic["listingId"] = synthetic["listingId"].astype(str)


# =========================================================
# 3. Make sure data_source is correct
# =========================================================

real["data_source"] = "Real"
synthetic["data_source"] = "Synthetic"


# =========================================================
# 4. Make columns consistent
# =========================================================

all_columns = list(
    dict.fromkeys(
        list(real.columns) + list(synthetic.columns)
    )
)

real = real.reindex(columns=all_columns)
synthetic = synthetic.reindex(columns=all_columns)


# =========================================================
# 5. Combine
# =========================================================

combined = pd.concat(
    [real, synthetic],
    ignore_index=True
)


# =========================================================
# 6. Shuffle the dataset
# =========================================================

combined = combined.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)


# =========================================================
# 7. Create final row ID
# =========================================================

combined.insert(
    0,
    "row_id",
    [f"ROW_{i:05d}" for i in range(1, len(combined) + 1)]
)


# =========================================================
# 8. Save
# =========================================================

Path(OUTPUT_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

combined.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# 9. Final validation
# =========================================================

print("\nFinal shape:")
print(combined.shape)

print("\nData source:")
print(combined["data_source"].value_counts())

print("\nMissing values:")
print(
    combined.isna()
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .to_string()
)

print("\nDuplicate rows:")
print(
    "Exact duplicate rows:",
    combined.drop(columns=["row_id"]).duplicated().sum()
)

print(
    "Duplicate listing IDs:",
    combined["listingId"].duplicated().sum()
)

print("\nBasic validity:")

print(
    "Price <= 0:",
    (combined["price"] <= 0).sum()
)

print(
    "Area <= 0:",
    (combined["area_sqft"] <= 0).sum()
)

print(
    "BHK <= 0:",
    (combined["bhk"] <= 0).sum()
)

print(
    "Bathrooms < 0:",
    (combined["bathrooms"] < 0).sum()
)

print(
    "Parking < 0:",
    (combined["parking"] < 0).sum()
)

valid_floor = (
    combined["floor_number"].notna()
    & combined["total_floors"].notna()
)

print(
    "Floor > total floors:",
    (
        combined.loc[valid_floor, "floor_number"]
        > combined.loc[valid_floor, "total_floors"]
    ).sum()
)


# =========================================================
# 10. Real vs synthetic summary
# =========================================================

print("\nReal vs Synthetic medians:")

numeric_cols = [
    "price",
    "area_sqft",
    "bhk",
    "bathrooms",
    "parking",
    "floor_number",
    "total_floors",
    "age_years"
]

summary = pd.DataFrame({
    "Real": real[numeric_cols].median(),
    "Synthetic": synthetic[numeric_cols].median(),
    "Combined": combined[numeric_cols].median()
})

print(
    summary.round(2).to_string()
)


# =========================================================
# 11. BHK distribution
# =========================================================

print("\nBHK distribution:")

bhk_summary = pd.DataFrame({
    "Real": real["bhk"].value_counts(),
    "Synthetic": synthetic["bhk"].value_counts(),
    "Combined": combined["bhk"].value_counts()
}).fillna(0).sort_index()

print(bhk_summary.to_string())


# =========================================================
# 12. Save validation summary
# =========================================================

validation_path = DATA_DIR / "processed" / "final_dataset_validation.txt"

with open(validation_path, "w") as f:

    f.write("FINAL DATASET VALIDATION\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Real records: {len(real)}\n")
    f.write(f"Synthetic records: {len(synthetic)}\n")
    f.write(f"Combined records: {len(combined)}\n\n")

    f.write("Data source:\n")
    f.write(
        combined["data_source"]
        .value_counts()
        .to_string()
    )

    f.write("\n\nExact duplicate rows: ")
    f.write(
        str(
            combined
            .drop(columns=["row_id"])
            .duplicated()
            .sum()
        )
    )

    f.write("\nDuplicate listing IDs: ")
    f.write(
        str(
            combined["listingId"]
            .duplicated()
            .sum()
        )
    )

    f.write("\n\nInvalid values:\n")

    f.write(
        f"Price <= 0: {(combined['price'] <= 0).sum()}\n"
    )

    f.write(
        f"Area <= 0: {(combined['area_sqft'] <= 0).sum()}\n"
    )

    f.write(
        f"BHK <= 0: {(combined['bhk'] <= 0).sum()}\n"
    )

    f.write(
        f"Floor > total floors: "
        f"{((combined.loc[valid_floor, 'floor_number'] > combined.loc[valid_floor, 'total_floors']).sum())}\n"
    )

print("\nValidation report saved to:")
print(validation_path)

print("\nFinal dataset saved to:")
print(OUTPUT_PATH)

print("\nDONE.")