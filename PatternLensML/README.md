# PatternLens ML

A Flask prototype for intelligent pattern discovery and prediction from CSV data.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`, upload a CSV file, and enter the target column name.

The app automatically:

- imputes missing feature values and encodes categorical features;
- trains a Random Forest classifier or regressor based on the target type;
- reports evaluation metrics on a holdout set;
- discovers numeric clusters with K-Means;
- displays a prediction preview.

This is a demonstration model. Before production use, replace the Flask secret, add authentication, persist trained models, and use a time-aware validation strategy for time-series data.
