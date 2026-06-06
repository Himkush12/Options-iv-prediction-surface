# Options Implied Volatility Surface Prediction

This project is an end-to-end pipeline for predicting the Implied Volatility (IV) surface of Nifty50 options. 

The initial goal was to build a purely machine-learning-based time-series forecasting model using LightGBM. However, during cross-validation, the tree-based model failed significantly on 0-DTE (Expiration Day) options due to boundary extrapolation limits. To solve this, the final production pipeline uses a hybrid approach, relying on a localized Cross-Sectional Cubic Spline to handle the 0-DTE boundary conditions.

## Project Structure

The pipeline is split into three notebooks that should be run sequentially:

### 1. `01_data_cleaning.ipynb` (Data Prep & Filtering)
This notebook handles the messy raw data:
* Takes the wide-format option chain data and melts it into a long-format structure.
* Parses the text strings to extract standard option metadata (Strike, Expiry, Call/Put).
* Applies cross-sectional filtering to remove extreme outlier IV values (the 1% and 99% quantiles) that occur due to low liquidity, ensuring the model trains on stable quotes.

### 2. `02_feature_engineering.ipynb` (The ML Feature Space)
This notebook builds the feature space originally designed for the LightGBM baseline model. 
* **Spatial Features:** Calculates standard options metrics like Log-Moneyness ($S/K$) to map the volatility smile, and inverse Days-to-Expiration (DTE).
* **Temporal Features:** Creates rolling statistical moments (like a 5-period rolling IV mean) and momentum indicators to capture the velocity of volatility changes over time. 
* *Note: While these features are highly predictive for standard trading days, the final pipeline bypasses them on expiration days in favor of spatial interpolation.*

### 3. `03_modeling.ipynb` (Modeling & The 0-DTE Pivot)
This is where the actual forecasting happens. 

**The Problem:** During a 5-Fold Walk-Forward Cross-Validation, the LightGBM model performed exceptionally well on standard days (Folds 1-4 RMSE ~0.002) but completely failed on Expiration Day data (Fold 5 RMSE ~0.38). Because 0-DTE IV spikes exponentially, and tree-based models cannot extrapolate beyond the maximum values they saw in training, LightGBM flatlined its predictions.

**The Solution:** To fix the 0-DTE boundary issue, I implemented a Cross-Sectional Cubic Spline interpolator. 
* At every timestamp during inference, the pipeline looks at the known strikes currently trading.
* It fits a smooth cubic polynomial curve through those strikes to perfectly recreate the Volatility Smile.
* It then uses that spatial curve to interpolate the IV for the missing prediction strikes.
* **Safeguard:** To prevent the spline from wildly oscillating on the deep Out-of-the-Money wings (Runge's phenomenon), query strikes are dynamically clamped to the minimum and maximum known strikes.

## Results

By switching to spatial interpolation for the boundary data, the overall error dropped significantly:

| Metric | Score |
| :--- | :--- |
| **Pooled OOF MSE** | 0.000056 |
| **Pooled OOF RMSE** | 0.007506 |
| **0-DTE Specific RMSE** | 0.023989 |

## Running the Code
To reproduce the environment:
1. Ensure your raw `dataset.csv` is located in the `data/` folder.
2. Run `01_data_cleaning.ipynb` to generate the base `.parquet` files.
3. Run `02_feature_engineering.ipynb` to append the feature space.
4. Run `03_modeling.ipynb` to run the cross-validation and output `submission_spline.csv`.

*(Note: The generated `.parquet` files are ignored in this repository due to file size limits, but the notebooks will regenerate them cleanly from the source data).*
