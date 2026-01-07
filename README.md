# Multimodal Real Estate Valuation Pipeline 🏠

A machine learning project that combines tabular housing data with satellite imagery to predict property market values using a hybrid regression approach.

## Overview

This project implements an advanced multimodal regression pipeline for real estate analytics that integrates two complementary data sources:
- **Tabular Data**: Traditional property attributes (bedrooms, bathrooms, lot size, etc.)
- **Visual Data**: Satellite imagery capturing environmental context and neighborhood characteristics

By combining these modalities, the system leverages both structural property features and environmental indicators to achieve superior valuation predictions.

## Project Architecture

### Data Pipeline

#### 1. **Data Fetching & Satellite Imagery** (`data-fetch.ipynb`)
- Uses Mapbox Static Images API to fetch satellite imagery
- Retrieves 224x224 pixel satellite images for each property using latitude and longitude coordinates
- Saves images in two separate folders: `images/` (training) and `imagesval/` (test)
- Handles API authentication and error management

**Key Components:**
- Mapbox API Integration
- Batch image downloading with error handling
- Coordinate-based spatial data acquisition

#### 2. **Data Preprocessing** (`preprocessing.ipynb`)
- **Dataset Sizes:**
  - Training data: 16,208 properties
  - Test data: 5,404 properties
  
- **Feature Engineering:**
  - Log transformation of target variable (price) for improved model training
  - House age calculation from year built
  - Renovation status binary encoding
  - Removal of outliers (e.g., bedrooms > 33)
  
- **Exploratory Data Analysis:**
  - Price distribution analysis
  - Correlation heatmap of features
  - Geospatial price visualization
  - Feature importance assessment
  
- **Visual Feature Extraction:**
  - ResNet-18 architecture (pre-trained on ImageNet)
  - 512-dimensional feature vectors extracted from satellite imagery
  - Per-property visual embeddings capturing curb appeal and neighborhood characteristics
  
**Dataset Description:**

| Feature | Type | Description |
|---------|------|-------------|
| sqft_living | Numeric | Total interior living space |
| sqft_above | Numeric | Interior space above ground |
| sqft_basement | Numeric | Interior space below ground |
| sqft_lot | Numeric | Total land area |
| sqft_living15, sqft_lot15 | Numeric | Average sizes of 15 nearest neighbors |
| condition | Ordinal (1-5) | Property maintenance level |
| grade | Ordinal (1-13) | Construction quality and design |
| view | Ordinal (0-4) | View rating from property |
| waterfront | Binary | Water view indicator |
| lat, long | Numeric | Geographic coordinates |

#### 3. **Model Training & Evaluation** (`model_training.ipynb`)

**Models Implemented:**

1. **Tabular Data Only (16 features + 49 engineered features)**
   - Linear Regression: R² = 0.6943, RMSE = $182,527.55
   - Random Forest (150 estimators): R² = 0.6634, RMSE = $164,427.07
   - XGBoost (500 estimators): R² = 0.7960, RMSE = $139,189.90

2. **Multimodal (Tabular + Image Features, 528 features total)**
   - Linear Regression: R² = 0.6943, RMSE = $182,527.55
   - Random Forest: R² = 0.6634, RMSE = $164,427.07
   - XGBoost: R² = 0.7960, RMSE = $139,189.90
   - **MLP Neural Network: R² = 0.7247, RMSE = $190,996.69**

**Best Performing Model:** XGBoost Regressor (Tabular Data)
- 500 boosting rounds
- Max depth: 6
- Learning rate: 0.05
- Subsample: 0.8
- Column sample by tree: 0.8

**Note:** While multimodal features provide visual context, the tabular XGBoost model achieved the highest R² score, suggesting that traditional property attributes are the primary value drivers in this dataset.

## Data Preprocessing Steps

```
Raw Data → Data Cleaning → Feature Engineering → 
Feature Scaling/Encoding → Train-Test Split (80-20) → Model Training
```

### Categorical Features Encoded:
- bedrooms, floors, waterfront, view, condition, grade, isrenovated

### Numerical Features Scaled:
- bathrooms, sqftlot, sqftabove, sqftbasement, lat, long, sqftliving15, sqftlot15, houseage, and all 512 image features

## Features

### Tabular Features (16 Core)
- Property dimensions: bedrooms, bathrooms, sqft_living, sqft_lot
- Location: latitude, longitude
- Condition: condition, grade, view, waterfront
- Structural: sqft_above, sqft_basement, yrbuilt, yrrenovated
- Neighborhood: sqft_living15, sqft_lot15

### Visual Features (512 Dimensional)
- Extracted from satellite imagery using ResNet-18
- Captures curb appeal, green space density, road infrastructure
- Provides neighborhood character encoding

### Engineered Features
- log_price: Log-transformed target variable
- house_age: Years since construction
- is_renovated: Binary renovation indicator

## Model Performance Comparison

| Model | Data Type | R² Score | RMSE ($) | Remarks |
|-------|-----------|----------|----------|---------|
| Linear Regression | Tabular | 0.6943 | 182,527 | Baseline |
| Random Forest | Tabular | 0.6634 | 164,427 | Lower R² despite lower RMSE |
| XGBoost | Tabular | **0.7960** | **139,190** | Best overall |
| Linear Regression | Multimodal | 0.6943 | 182,527 | No improvement with images |
| Random Forest | Multimodal | 0.6634 | 164,427 | No improvement with images |
| XGBoost | Multimodal | **0.7960** | **139,190** | Consistent with tabular |
| MLP Neural Net | Multimodal | 0.7247 | 190,997 | Deep learning baseline |

## Repository Structure

```
project/
├── README.md                          # Project documentation
├── data-fetch.ipynb                   # Satellite image fetching
├── preprocessing.ipynb                # Data cleaning & feature engineering
├── model_training.ipynb               # Model training & evaluation
├── data/
│   ├── train1.xlsx                    # Raw training data
│   ├── test2.xlsx                     # Raw test data
│   ├── tabulardata.csv                # Processed training features
│   ├── test.csv                       # Processed test features
│   ├── imagedata.csv                  # Training image embeddings
│   └── imagedataval.csv               # Test image embeddings
├── images/                            # Training satellite imagery
├── imagesval/                         # Test satellite imagery
└── models/
    └── xgb_best.pkl                   # Serialized best model
```

## Installation & Setup

### Requirements
```
Python 3.8+
pandas >= 1.0
numpy >= 1.18
scikit-learn >= 0.24
xgboost >= 1.5
torch >= 1.9
torchvision >= 0.10
pillow >= 8.0
requests >= 2.25
matplotlib >= 3.3
seaborn >= 0.11
```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd real-estate-valuation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up Mapbox API token
export MAPBOX_TOKEN="your_mapbox_token_here"
```

## Usage

### 1. Fetch Satellite Imagery
```python
# Run data-fetch.ipynb
# Requires: Mapbox API token
# Output: Satellite images in images/ and imagesval/ folders
```

### 2. Preprocess Data
```python
# Run preprocessing.ipynb
# Input: train1.xlsx, test2.xlsx, satellite images
# Output: Cleaned CSVs and image embeddings
```

### 3. Train Models
```python
# Run model_training.ipynb
# Trains: Linear Regression, Random Forest, XGBoost, MLP
# Output: Model metrics and best model
```

### 4. Make Predictions
```python
import joblib
import pandas as pd

# Load best model
model = joblib.load('xgb_best.pkl')

# Load test data
test_data = pd.read_csv('test.csv')

# Make predictions
predictions = model.predict(test_data)
print(predictions)
```

## Key Insights

### Data Characteristics
- **Price Range**: $75,000 - $7,700,000
- **Average Price**: $537,470
- **Properties**: 16,208 training, 5,404 test
- **Features**: 16 tabular + 512 visual embeddings

### Model Performance Insights
1. **XGBoost dominance**: Gradient boosting captures non-linear relationships in property values
2. **Tabular features sufficiency**: Despite multimodal approach, tabular data contains most predictive power
3. **Image features role**: While visual data provides contextual information, structured attributes are primary value drivers
4. **Potential improvements**: 
   - Feature engineering: Additional derived metrics from coordinates
   - Ensemble methods: Combining multiple model predictions
   - Hyperparameter tuning: Grid search across larger parameter space

### Geographic Patterns
- Latitude and longitude show strong correlation with price
- Neighborhood characteristics (sqft_living15, sqft_lot15) capture local market dynamics
- Waterfront properties command significant premium

## Technical Highlights

### Deep Learning for Feature Extraction
- **Architecture**: ResNet-18 (pre-trained)
- **Input**: 224x224 RGB satellite images
- **Output**: 512-dimensional feature vectors
- **Advantage**: Transfer learning captures general visual patterns applicable to real estate

### Hyperparameter Optimization
```python
# XGBoost Best Configuration
XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)
```

### Data Processing Pipeline
```
Raw Input → Missing Value Handling → Outlier Removal →
Feature Engineering → Scaling/Encoding → Train-Test Split →
Model Training → Cross-Validation → Hyperparameter Tuning →
Final Evaluation
```

## Future Enhancements

1. **Advanced Computer Vision**
   - Sentiment analysis of satellite imagery
   - Road network extraction
   - Green space quantification

2. **Ensemble Methods**
   - Stacking predictions from multiple models
   - Weighted averaging based on performance

3. **Time Series Analysis**
   - Historical price trends
   - Seasonal patterns in real estate markets

4. **External Data Integration**
   - School ratings and proximity
   - Crime statistics
   - Zoning information
   - Economic indicators

5. **Deployment**
   - REST API for real-time predictions
   - Web interface for property valuation
   - Batch prediction pipeline

## Project Metrics

| Metric | Value |
|--------|-------|
| Total Properties | 21,612 |
| Training Set | 16,208 (75%) |
| Test Set | 5,404 (25%) |
| Features (Tabular) | 16 |
| Features (Visual) | 512 |
| Best Model R² | 0.7960 |
| Best Model RMSE | $139,190 |
| Training Time | ~5 minutes |
| Prediction Time | <1ms per property |

## File Descriptions

### Notebooks

**data-fetch.ipynb**
- Purpose: Satellite image acquisition
- Key Function: `fetch_satellite_image(lat, lon, idx, zoom)`
- API: Mapbox Static Images
- Output: 224x224 PNG files

**preprocessing.ipynb**
- Purpose: Data cleaning, EDA, feature engineering
- Key Operations:
  - Data validation and cleaning
  - Outlier detection and removal
  - Feature engineering and transformation
  - ResNet-18 feature extraction from images
- Output: Processed CSV files and image embeddings

**model_training.ipynb**
- Purpose: Model development and evaluation
- Models: Linear Regression, Random Forest, XGBoost, MLP
- Metrics: R², RMSE, MAE
- Output: Trained models and performance comparisons

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| API rate limiting | Batch processing with delays |
| Missing values | Handled via pandas dropna() |
| Extreme outliers (33 bedrooms) | Removed based on domain knowledge |
| Feature scaling differences | StandardScaler for numerical, OneHotEncoder for categorical |
| Multimodal feature alignment | Concatenated processed features after normalization |
| Model overfitting | Regularization, early stopping, cross-validation |

## Contributing

For improvements or modifications:
1. Create a feature branch
2. Make changes with clear commits
3. Test on both tabular and multimodal pipelines
4. Submit pull request with documentation

## License

This project is for educational and analytical purposes.

## Contact & Support

For questions or collaboration opportunities:
- GitHub Issues: [Link to issues]
- Email: [Your contact]

## Acknowledgments

- Mapbox for satellite imagery API
- PyTorch and scikit-learn communities
- Real estate data provided by [Source]

---

**Last Updated**: January 2026  
**Python Version**: 3.8+  
**Project Status**: Complete
