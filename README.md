# Predictive Valuation Modeling for Detecting Price Anomalies in Bengaluru Residential Resale Apartments

## Overview

This Business Analytics project develops a predictive valuation model for Bengaluru residential resale apartment listings and uses the model to identify listings whose observed asking prices differ substantially from their model-predicted prices.

### Central Business Question

Can the expected listing price of a Bengaluru residential resale apartment be predicted from its property characteristics, and can substantial differences between actual and predicted prices be used to identify potentially underpriced and potentially overpriced listings?

### Overall Workflow

**Data Collection → Data Cleaning → Data Preparation → Exploratory Data Analysis → Predictive Modeling → Model Evaluation → Out-of-Fold Prediction → Price Deviation Analysis → IQR-Based Anomaly Detection → Business Insights**

---

## Business Problem

Residential apartment asking prices in Bengaluru vary according to factors such as area, BHK configuration, bathrooms, parking, location, floor, furnishing, property age, and construction status.

For buyers, sellers, and real-estate platforms, it can be useful to estimate an expected listing price and identify properties whose asking prices differ substantially from that expectation.

This project therefore develops a predictive pricing model and uses model-based deviations as a screening mechanism for potential price anomalies.

### Relevant Property Factors

The analysis considers:

- Area
- BHK
- Bathrooms
- Parking
- Location
- Floor number
- Total floors
- Property age
- Furnishing
- Facing
- Possession category
- Construction status

### Business Motivation

The resulting model can support property screening, pricing review, and location-level analysis. The anomaly labels are screening indicators and are not treated as definitive statements about the true market value of a property.

---

# 1. Project Objectives

1. Collect Bengaluru residential resale apartment listings.
2. Clean and prepare the collected property data.
3. Build a 10,000-record analytical dataset.
4. Perform exploratory data analysis to understand property-price relationships.
5. Develop regression models for apartment listing-price prediction.
6. Compare models using MAE, RMSE, and R².
7. Select the predictive model based on evaluation results.
8. Generate out-of-fold predictions for all listings.
9. Calculate deviations between observed and predicted prices.
10. Identify potentially underpriced and potentially overpriced listings.
11. Extract business insights from property and location characteristics.
12. Provide business recommendations based on the findings.

---

# 2. Data Collection

## Source

The property listings were collected from **Housing.com**, focusing on Bengaluru residential resale apartment listings.

## Collection Method

Data collection was performed using the **Apify Housing.com Property Scraper**.

The collected records contain property listing information such as price, area, BHK, location, bathrooms, parking, furnishing, project information, floor information, construction status, and other available listing attributes.

## Apify Collection Workflow

1. Select the Bengaluru residential property search.
2. Configure the Housing.com property scraper on Apify.
3. Request residential sale listings.
4. Collect the available property listing fields.
5. Export the collected records as CSV.
6. Perform cleaning and preparation locally before analysis.

## Data Collected

Important collected information includes:

- Listing ID
- Listing title
- Price
- Area
- BHK
- Bathrooms
- Parking
- Location
- Latitude and longitude
- Floor information
- Furnishing
- Facing
- Property age
- Possession information
- Construction status
- Project name
- Posted date
- Property type
- Sale classification

---

# 3. Dataset

The final analytical dataset contains **10,000 Bengaluru residential resale apartment listings**.

## Raw Dataset

The raw/collected dataset contains the property listing records obtained during the data collection process.

## Processed Dataset

The processed dataset contains the cleaned and prepared variables used for exploratory analysis and predictive modeling.

## Dataset Size

**10,000 records**

The processed dataset contains the cleaned property attributes required for analysis.

## Important Columns

| Variable | Description |
|---|---|
| `listingId` | Property listing identifier |
| `price` | Listing price in Indian Rupees |
| `area_sqft` | Property area in square feet |
| `bhk` | BHK configuration |
| `bathrooms` | Number of bathrooms |
| `parking` | Number of parking spaces |
| `location` | Property location |
| `latitude` | Latitude |
| `longitude` | Longitude |
| `floor_number` | Property floor |
| `total_floors` | Total floors in building |
| `furnishing` | Furnishing status |
| `facing` | Facing direction |
| `age_years` | Property age in years |
| `possession_category` | Possession classification |
| `is_under_construction` | Construction status |
| `project_name` | Project name where available |
| `posted_date` | Listing posted date |

---

# 4. Target Variable

## Target Definition

The target variable for predictive modeling is:

```text
price
```

It represents the observed **listing/asking price** of the apartment.

## Price Anomaly Definition

After predicting the expected listing price, the difference between actual and predicted price is calculated as a percentage deviation.

```text
Deviation (%) =
(Actual Price - Predicted Price)
-------------------------------- × 100
        Actual Price
```

### Interpretation

- **Negative deviation:** actual listing price is below the model prediction.
- **Positive deviation:** actual listing price is above the model prediction.

The anomaly categories are:

- **Potentially Underpriced**
- **Fairly Priced**
- **Potentially Overpriced**
- **Invalid Prediction**

These are model-based screening categories and do not represent verified market valuations.

---

# 5. Data Cleaning and Preparation

The collected property data was cleaned and prepared before analysis.

### Major Cleaning Activities

- Clean listing prices.
- Clean and standardize property area.
- Extract structured property characteristics from listing information.
- Resolve BHK values using available structured information and listing titles.
- Extract bathrooms and parking information.
- Convert coordinates into latitude and longitude.
- Parse floor information into floor number and total floors.
- Convert property age into numeric years.
- Standardize facing values.
- Standardize possession information.
- Convert listing dates into usable date fields.
- Perform duplicate and validity checks.
- Evaluate missing values.
- Preserve plausible extreme property values rather than automatically removing them.

### Missing-Value Handling

Missing values are handled within the machine-learning preprocessing pipeline.

- Numerical variables → median imputation
- Categorical variables → most-frequent imputation

Categorical variables are then transformed using one-hot encoding.

The preprocessing pipeline is fitted on the training data to reduce the risk of data leakage.

---

# 6. Exploratory Data Analysis

The exploratory analysis examines the structure and pricing patterns of the 10,000-property dataset.

### Areas of Analysis

- Listing price distribution
- Property area distribution
- BHK distribution
- Median price by BHK
- Area versus price
- Correlation analysis
- Bathrooms and price
- Parking and price
- Furnishing and price
- Property age and price
- Construction status and price
- Floor number and price
- Location-wise listing counts
- Location-wise median prices
- Price per square foot
- Location-wise price per square foot
- Missing-value analysis
- Final data-quality checks

The derived `price_per_sqft` variable is used for descriptive analysis only and is not used as a predictive feature because it is directly derived from the target price.

---

# 7. Predictive Modeling

## Feature Selection

The predictive model uses the following numerical features:

```text
area_sqft
bhk
bathrooms
parking
floor_number
total_floors
age_years
```

Categorical features:

```text
location
furnishing
facing
possession_category
is_under_construction
```

### Preprocessing Pipeline

The modeling pipeline performs:

1. Numerical missing-value imputation using the median.
2. Categorical missing-value imputation using the most frequent value.
3. One-hot encoding of categorical features.
4. Model fitting using the transformed feature set.

Fields such as identifiers, constant fields, listing text, dates, and other reference information were not used as initial model predictors.

---

# 8. Machine Learning Models

Three regression models were evaluated:

### Linear Regression

Used as a baseline regression model for predicting apartment listing prices.

### Random Forest Regression

Used to evaluate whether an ensemble of decision trees could capture nonlinear property-price relationships.

### Gradient Boosting Regression

Used as another tree-based ensemble model for comparison.

---

# 9. Model Evaluation

The models were evaluated using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

## Model Comparison

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | ₹4.089M | ₹6.695M | 0.7970 |
| Random Forest | ₹3.993M | ₹8.891M | 0.6419 |
| Gradient Boosting | ₹4.256M | ₹9.200M | 0.6166 |

---

# 10. Selected Model

**Linear Regression** was selected as the final predictive model.

It achieved the highest R² and the lowest RMSE among the evaluated models.

The Random Forest model had a slightly lower MAE, but its RMSE and R² were substantially worse than Linear Regression.

Therefore, Linear Regression was retained for the subsequent out-of-fold prediction and anomaly-detection stage.

---

# 11. Out-of-Fold Prediction

Price anomaly detection requires a prediction for every property in the 10,000-record dataset.

A **5-fold cross-validation** approach was used to generate out-of-fold predictions.

The process works as follows:

1. Divide the dataset into five folds.
2. Train the model on four folds.
3. Predict the remaining fold.
4. Repeat until every property has received a prediction.
5. Combine all out-of-fold predictions.

This ensures that each property's prediction comes from a model that was not trained on that property.

Eight out-of-fold predictions were negative and therefore were treated as invalid model predictions for anomaly scoring.

---

# 12. Price Deviation Analysis

The signed percentage deviation is calculated as:

```text
Deviation (%) =
(Actual Price - Predicted Price)
-------------------------------- × 100
        Actual Price
```

### Interpretation

A negative value means the observed listing price is below the model-predicted price.

A positive value means the observed listing price is above the model-predicted price.

The eight negative predicted-price cases were classified separately as **Invalid Prediction** and excluded from percentage-based anomaly scoring.

---

# 13. Price Anomaly Detection

The **Interquartile Range (IQR)** method was used to identify unusually large deviations.

```text
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

### Final Thresholds

```text
Q1  = -16.83%
Q3  =  13.65%
IQR =  30.48%

Potentially Underpriced:
Deviation < -62.56%

Fairly Priced:
-62.56% to +59.37%

Potentially Overpriced:
Deviation > +59.37%
```

The IQR method was selected because it derives anomaly boundaries from the observed deviation distribution rather than forcing a fixed percentage of listings into each category.

---

# 14. Key Results

## Model Results

Linear Regression achieved:

- **MAE:** ₹4.089 million
- **RMSE:** ₹6.695 million
- **R²:** 0.7970

## Anomaly Results

| Price Category | Listings | Percentage |
|---|---:|---:|
| Fairly Priced | 9,313 | 93.13% |
| Potentially Underpriced | 581 | 5.81% |
| Potentially Overpriced | 98 | 0.98% |
| Invalid Prediction | 8 | 0.08% |
| **Total** | **10,000** | **100%** |

The majority of listings fell within the IQR-based normal deviation range.

---

# 15. Business Insights

## Potentially Underpriced Listings

The 581 potentially underpriced listings had the following median characteristics:

- Actual price: **₹95 lakh**
- Predicted price: **approximately ₹1.85 crore**
- Area: **1,345 sq.ft.**
- BHK: **2.5**
- Bathrooms: **2**
- Parking: **1**
- Property age: **7 years**

These listings had observed asking prices substantially below their model-predicted prices.

## Potentially Overpriced Listings

The 98 potentially overpriced listings had:

- Actual price: **₹1.37 crore median**
- Predicted price: **approximately ₹38.3 lakh median**
- Area: **972.5 sq.ft. median**
- BHK: **2 median**
- Bathrooms: **2 median**
- Parking: **1 median**
- Property age: **approximately 4.1 years median**

These listings had observed asking prices substantially above their model-predicted prices.

## Location Patterns

Locations contributing relatively large numbers of potentially underpriced listings included:

- Phase I, Electronic City
- Dodsworth Layout, Whitefield
- Varthur
- Sarjapur
- Balagere
- Phase 2, Whitefield
- Attibele
- Kodigehalli, K R Puram
- 7th Phase, JP Nagar

These are model-identified concentrations and should not be interpreted as proof that the locations themselves are inherently underpriced.

## Property Characteristics

Additional findings include:

- Potentially underpriced listings had a higher median property age than fairly priced listings.
- Potentially overpriced listings had a higher proportion of 2-BHK properties.
- Semi-furnished properties were common across all three categories.
- Construction status showed relatively similar distributions across the categories.

---

# 16. Business Recommendations

### 1. Property Screening

A real-estate platform can use the model to compare an observed listing price with an expected price and flag listings for further investigation.

### 2. Investigate Potentially Underpriced Listings

Flagged listings can be reviewed for:

- Property condition
- Seller urgency
- Documentation and legal factors
- Maintenance requirements
- Building quality
- Amenities
- Exact micro-location
- Negotiation conditions

### 3. Review Potentially Overpriced Listings

Potentially overpriced listings can be reviewed to determine whether the asking price is justified by property characteristics not fully captured by the model.

### 4. Monitor Location-Level Patterns

Repeated anomaly patterns can be monitored by location to identify areas where observed listing prices frequently differ from model expectations.

### 5. Periodically Update the Model

Real-estate markets change over time. The predictive model should therefore be updated periodically with newer listing data.

---

# 17. Project Workflow

```text
Housing.com
     ↓
Apify Data Collection
     ↓
Raw Property Listings
     ↓
Data Cleaning
     ↓
Data Preparation
     ↓
10,000-Record Dataset
     ↓
Exploratory Data Analysis
     ↓
Feature Selection
     ↓
Missing-Value Handling
     ↓
One-Hot Encoding
     ↓
Train/Test Split
     ↓
Model Comparison
     ↓
Linear Regression Selected
     ↓
5-Fold Out-of-Fold Prediction
     ↓
Price Deviation
     ↓
IQR-Based Anomaly Detection
     ↓
Potentially Underpriced / Fairly Priced / Potentially Overpriced
     ↓
Business Insights and Recommendations
```

---

# 18. Repository Structure

```text
baIndividualProject/
│
├── README.md
├── .gitignore
├── analysis.ipynb
│
└── data/
    ├── realEstateListingsDataRaw.csv
    ├── realEstateListingsDataProcessed.csv
    ├── 01_clean_data.py
    └── 02_prepare_data.py
```

---

# 19. Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Data Collection:** Apify
- **Data Source:** Housing.com
- **Development Environment:** Jupyter Notebook

---

# 20. Python Libraries

The project uses:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

The machine-learning workflow uses scikit-learn components including:

- `train_test_split`
- `ColumnTransformer`
- `Pipeline`
- `SimpleImputer`
- `OneHotEncoder`
- `LinearRegression`
- `RandomForestRegressor`
- `GradientBoostingRegressor`
- `KFold`
- `cross_val_predict`
- Regression evaluation metrics

---

# 21. Reproducibility

The project maintains a structured workflow from raw data through data preparation and analysis.

To reproduce the analysis:

1. Use the collected raw dataset.
2. Run the data-cleaning process.
3. Run the data-preparation process.
4. Load the processed dataset into the analysis notebook.
5. Execute the exploratory analysis.
6. Run the preprocessing and predictive modeling pipeline.
7. Generate out-of-fold predictions.
8. Calculate price deviations.
9. Apply the IQR anomaly-detection method.

The random states used in model evaluation and cross-validation are fixed to support reproducibility.

---

# 22. How to Run the Project

## Step 1 — Clone or Download the Repository

Download or clone the project repository to your local machine.

## Step 2 — Install Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Step 3 — Run Data Processing

Navigate to the data directory and execute the cleaning and preparation scripts:

```bash
python 01_clean_data.py
python 02_prepare_data.py
```

## Step 4 — Run the Analysis

Open the notebook:

```bash
jupyter notebook analysis.ipynb
```

Run the notebook cells sequentially to reproduce the EDA, model evaluation, out-of-fold prediction, anomaly detection, and business analysis.

---

# 23. Business Interpretation

The model provides a data-driven screening mechanism for residential property listings.

### Potential Applications

- Property listing screening
- Price-review assistance
- Buyer-side property investigation
- Seller-side pricing review
- Real-estate platform analytics
- Location-level anomaly monitoring

The model should support decision-making rather than replace property-level due diligence.

A listing flagged as potentially underpriced or potentially overpriced should be investigated further because important factors may not be captured in the available dataset.

---

# 24. Limitations

### Listing Price vs Transaction Price

The target variable represents the advertised listing/asking price and not the final negotiated transaction price.

### Missing Property Information

Some property characteristics contain missing values and are handled through preprocessing.

### Model Limitations

The predictive model is based only on the available property characteristics and may not capture every factor influencing listing prices.

### Location Variation

Location is represented using the available listing location information. Very specific micro-location effects may not be fully captured.

### Temporal Variation

Property prices can change over time. The dataset represents listings collected during the data-collection period.

### Anomaly Interpretation

A potentially underpriced or potentially overpriced classification is a model-based screening result and does not prove that a listing is objectively underpriced or overpriced.

### Invalid Predictions

A small number of out-of-fold predictions were negative. These eight records were excluded from percentage-based anomaly scoring and reported separately as invalid predictions.

---

# 25. Ethical and Responsible Data Use

The project uses publicly available property listing information for academic Business Analytics purposes.

The analysis is intended for:

- Academic study
- Predictive analytics demonstration
- Business intelligence
- Decision-support research

Model outputs should not be treated as guaranteed property valuations or as a substitute for professional real-estate, legal, or financial due diligence.

---

# 26. References

1. **Housing.com** — Bengaluru residential property listings used as the source for the collected property data.

2. **Apify — Housing.com Property Scraper** — Used as the data-collection mechanism for obtaining structured Housing.com property listing information.

3. **Scikit-learn Documentation** — Used for preprocessing pipelines, regression models, cross-validation, and model evaluation.

---

# 27. Final Deliverables

The project deliverables include:

- Project README
- Raw property dataset
- Processed property dataset
- Data-cleaning script
- Data-preparation script
- Exploratory data analysis
- Predictive modeling
- Model evaluation
- Out-of-fold prediction
- Price anomaly detection
- Business insights
- Business recommendations

---

# 28. Project Summary

This project applies predictive analytics to **10,000 Bengaluru residential resale apartment listings** to estimate expected listing prices and identify substantial deviations from those predictions.

Three regression models were evaluated, with **Linear Regression** selected based on its evaluation performance. Five-fold out-of-fold predictions were then generated for all listings to calculate model-based price deviations.

Using an IQR-based anomaly detection approach, the final analysis identified:

- **581 potentially underpriced listings**
- **9,313 fairly priced listings**
- **98 potentially overpriced listings**
- **8 invalid predictions**

The project demonstrates how web-collected real-estate data, exploratory analysis, regression modeling, cross-validation, and statistical anomaly detection can be combined to support property-price screening and business decision-making.

---

# Author

**Sadhana T P**

Business Analytics Individual Case Study

**Predictive Valuation Modeling for Detecting Price Anomalies in Bengaluru Residential Resale Apartments**
