import pandas as pd
import numpy as np

REAL_PATH = "data/processed/sdv_training_data.csv"
OUTPUT_PATH = "synthetic/controlled_synthetic_5001.csv"

N_SYNTHETIC = 5001
SEED = 42

rng = np.random.default_rng(SEED)

df = pd.read_csv(REAL_PATH)

# =========================================================
# PREPARATION
# =========================================================

df["price_per_sqft"] = df["price"] / df["area_sqft"]

# Use only valid positive values for price/sqft
df_psf = df[
    (df["price_per_sqft"] > 0)
    & np.isfinite(df["price_per_sqft"])
].copy()

# Real BHK distribution
bhk_probs = df["bhk"].value_counts(normalize=True)

# Location distribution
location_probs = df["location"].value_counts(normalize=True)

# Location-level price/sqft distributions
location_psf = (
    df_psf.groupby("location")["price_per_sqft"]
    .apply(list)
    .to_dict()
)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def sample_bhk():
    return rng.choice(
        bhk_probs.index.to_numpy(),
        p=bhk_probs.values
    )


def sample_location(bhk):
    subset = df.loc[
        df["bhk"] == bhk,
        "location"
    ].dropna()

    if len(subset) == 0:
        subset = df["location"].dropna()

    probs = subset.value_counts(normalize=True)

    return rng.choice(
        probs.index.to_numpy(),
        p=probs.values
    )


def sample_from_bhk(bhk, column):
    """Sample a value from the real distribution for a BHK."""
    values = df.loc[
        df["bhk"] == bhk,
        column
    ].dropna().to_numpy()

    if len(values) == 0:
        values = df[column].dropna().to_numpy()

    return rng.choice(values)


def sample_age(bhk):
    values = df.loc[
        df["bhk"] == bhk,
        "age_years"
    ].dropna().to_numpy()

    if len(values) == 0:
        values = df["age_years"].dropna().to_numpy()

    return float(rng.choice(values))


def sample_furnishing(bhk):
    subset = df[df["bhk"] == bhk]["furnishing"].dropna()

    if len(subset) == 0:
        subset = df["furnishing"].dropna()

    probs = subset.value_counts(normalize=True)

    return rng.choice(
        probs.index.to_numpy(),
        p=probs.values
    )


def sample_facing(bhk):
    subset = df[df["bhk"] == bhk]["facing"].dropna()

    if len(subset) == 0:
        subset = df["facing"].dropna()

    probs = subset.value_counts(normalize=True)

    return rng.choice(
        probs.index.to_numpy(),
        p=probs.values
    )


# =========================================================
# GENERATE SYNTHETIC RECORDS
# =========================================================

rows = []

for i in range(N_SYNTHETIC):

    # -----------------------------------------------------
    # 1. BHK
    # -----------------------------------------------------

    bhk = float(sample_bhk())

    # -----------------------------------------------------
    # 2. Select location
    # -----------------------------------------------------

    location = sample_location(bhk)

    # -----------------------------------------------------
    # 3. Select a real BHK template
    # -----------------------------------------------------

    bhk_group = df[df["bhk"] == bhk]

    if len(bhk_group) == 0:
        bhk_group = df

    template = bhk_group.iloc[
        rng.integers(0, len(bhk_group))
    ]

    # -----------------------------------------------------
    # 4. Generate area
    #
    # Use the BHK-specific distribution, then introduce
    # controlled variation.
    # -----------------------------------------------------

    area_values = bhk_group["area_sqft"].dropna().to_numpy()

    area = float(
        rng.choice(area_values)
    )

    # Small continuous perturbation
    area *= rng.normal(1.0, 0.035)

    # Keep area within the observed BHK range
    area_min = np.percentile(area_values, 2)
    area_max = np.percentile(area_values, 98)

    area = np.clip(area, area_min, area_max)

    area = round(area)

    # -----------------------------------------------------
    # 5. Bathrooms
    #
    # Strongly condition on BHK.
    # -----------------------------------------------------

    bathroom_values = bhk_group["bathrooms"].dropna().to_numpy()

    bathrooms = int(
        rng.choice(bathroom_values)
    )

    # Prevent obviously impossible bathroom counts
    bathrooms = max(1, min(bathrooms, int(np.ceil(bhk))))

    # -----------------------------------------------------
    # 6. Parking
    #
    # Sample from BHK-specific real distribution.
    # -----------------------------------------------------

    parking_values = bhk_group["parking"].dropna().to_numpy()

    parking = int(
        rng.choice(parking_values)
    )

    parking = max(0, parking)

    # -----------------------------------------------------
    # 7. Location-specific price/sqft
    #
    # If location has enough observations, use that
    # location's empirical distribution.
    # Otherwise use BHK-specific distribution.
    # -----------------------------------------------------

    psf_values = location_psf.get(location, [])

    if len(psf_values) < 5:
        psf_values = bhk_group[
            "price_per_sqft"
        ].dropna().to_numpy()

    if len(psf_values) == 0:
        psf_values = df_psf[
            "price_per_sqft"
        ].to_numpy()

    price_per_sqft = float(
        rng.choice(psf_values)
    )

    # Controlled price variation
    price_per_sqft *= rng.lognormal(
        mean=0,
        sigma=0.06
    )

    # -----------------------------------------------------
    # 8. Price
    # -----------------------------------------------------

    price = area * price_per_sqft

    # Round to realistic listing-price increments
    price = round(price / 10000) * 10000

    price = max(price, 500000)

    # -----------------------------------------------------
    # 9. Floor and total floors
    # -----------------------------------------------------

    total_floor_values = bhk_group[
        "total_floors"
    ].dropna().to_numpy()

    if len(total_floor_values) > 0:
        total_floors = int(
            round(rng.choice(total_floor_values))
        )
    else:
        total_floors = int(
            rng.integers(5, 30)
        )

    total_floors = max(1, total_floors)

    # Generate a valid floor
    floor_number = int(
        rng.integers(0, total_floors + 1)
    )

    # -----------------------------------------------------
    # 10. Age
    # -----------------------------------------------------

    age_years = sample_age(bhk)

    # Small variation
    if rng.random() < 0.7:
        age_years += rng.normal(0, 0.5)

    age_years = max(0, age_years)
    age_years = round(age_years, 1)

    # -----------------------------------------------------
    # 11. Furnishing
    # -----------------------------------------------------

    furnishing = sample_furnishing(bhk)

    # -----------------------------------------------------
    # 12. Facing
    # -----------------------------------------------------

    facing = sample_facing(bhk)

    # -----------------------------------------------------
    # 13. Possession status
    # -----------------------------------------------------

    possession_probs = (
        bhk_group["possession_category"]
        .dropna()
        .value_counts(normalize=True)
    )

    possession_category = rng.choice(
        possession_probs.index.to_numpy(),
        p=possession_probs.values
    )

    # -----------------------------------------------------
    # 14. Construction status
    # -----------------------------------------------------

    construction_probs = (
        bhk_group["is_under_construction"]
        .dropna()
        .value_counts(normalize=True)
    )

    is_under_construction = rng.choice(
        construction_probs.index.to_numpy(),
        p=construction_probs.values
    )

    # Make upcoming properties consistent
    if possession_category == "Upcoming":
        is_under_construction = True

    # -----------------------------------------------------
    # 15. Possession date
    # -----------------------------------------------------

    possession_date = np.nan

    if possession_category == "Upcoming":

        real_dates = pd.to_datetime(
            bhk_group.loc[
                bhk_group["possession_category"] == "Upcoming",
                "possession_date"
            ],
            errors="coerce"
        ).dropna()

        if len(real_dates) > 0:
            base_date = rng.choice(real_dates)

            # Add controlled date variation
            days_variation = int(
                rng.integers(-180, 181)
            )

            possession_date = (
                pd.Timestamp(base_date)
                + pd.Timedelta(days=days_variation)
            ).strftime("%Y-%m-%d")

    # -----------------------------------------------------
    # Create row
    # -----------------------------------------------------

    rows.append({
        "listingId": f"SYN_{i + 1:05d}",
        "price": price,
        "area_sqft": area,
        "bhk": bhk,
        "bathrooms": bathrooms,
        "parking": parking,
        "location": location,
        "floor_number": floor_number,
        "total_floors": total_floors,
        "furnishing": furnishing,
        "facing": facing,
        "age_years": age_years,
        "possession_category": possession_category,
        "possession_date": possession_date,
        "is_under_construction": bool(is_under_construction),
        "data_source": "Synthetic"
    })


# =========================================================
# DATAFRAME
# =========================================================

synthetic = pd.DataFrame(rows)

# =========================================================
# FINAL SAFETY CHECKS
# =========================================================

synthetic["price"] = synthetic["price"].round(0)
synthetic["area_sqft"] = synthetic["area_sqft"].round(0)

synthetic["bhk"] = synthetic["bhk"].astype(float)
synthetic["bathrooms"] = synthetic["bathrooms"].astype(int)
synthetic["parking"] = synthetic["parking"].astype(int)
synthetic["floor_number"] = synthetic["floor_number"].astype(int)
synthetic["total_floors"] = synthetic["total_floors"].astype(int)

# Ensure floor <= total floors
synthetic["floor_number"] = np.minimum(
    synthetic["floor_number"],
    synthetic["total_floors"]
)

# Ensure positive values
synthetic = synthetic[
    (synthetic["price"] > 0)
    & (synthetic["area_sqft"] > 0)
    & (synthetic["bhk"] > 0)
]

# =========================================================
# SAVE
# =========================================================

synthetic.to_csv(
    OUTPUT_PATH,
    index=False
)

print("=" * 60)
print("CONTROLLED SYNTHETIC DATA GENERATION")
print("=" * 60)

print(f"\nGenerated records: {len(synthetic)}")
print(f"Saved to: {OUTPUT_PATH}")

print("\nBHK distribution:")
print(
    synthetic["bhk"]
    .value_counts()
    .sort_index()
)

print("\nMedian comparison:")

comparison = pd.DataFrame({
    "real": df[
        ["price", "area_sqft", "bhk", "bathrooms", "parking"]
    ].median(),

    "synthetic": synthetic[
        ["price", "area_sqft", "bhk", "bathrooms", "parking"]
    ].median()
})

print(comparison.round(2).to_string())

print("\nInvalid values:")

print(
    "floor > total:",
    (
        synthetic["floor_number"]
        > synthetic["total_floors"]
    ).sum()
)

print(
    "price <= 0:",
    (synthetic["price"] <= 0).sum()
)

print(
    "area <= 0:",
    (synthetic["area_sqft"] <= 0).sum()
)

print(
    "bhk <= 0:",
    (synthetic["bhk"] <= 0).sum()
)

print("\nUnique synthetic rows:")

print(
    synthetic.drop(
        columns=["listingId"]
    ).drop_duplicates().shape[0]
)

print("\nSample:")
print(
    synthetic.head(10).to_string(index=False)
)
