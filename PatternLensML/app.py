from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
)
app.secret_key = "change-this-secret-in-production"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _json_safe(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_model(features, target):
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]

    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )

    if pd.api.types.is_numeric_dtype(target):
        estimator = RandomForestRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        )
        task = "regression"
    else:
        estimator = RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced", n_jobs=-1
        )
        task = "classification"

    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            ("model", estimator),
        ]
    ), task


def discover_patterns(data):
    numeric = data.select_dtypes(include=np.number).dropna(axis=1, how="all")
    if numeric.shape[1] < 2 or len(numeric) < 3:
        return None

    sample = numeric.fillna(numeric.median())
    scaled = StandardScaler().fit_transform(sample)
    cluster_count = min(3, len(sample))
    labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(
        scaled
    )
    counts = pd.Series(labels).value_counts().sort_index()
    return {
        "feature_count": int(numeric.shape[1]),
        "cluster_count": cluster_count,
        "clusters": [
            {"name": f"Pattern {index + 1}", "records": int(count)}
            for index, count in counts.items()
        ],
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/sample-dataset", methods=["GET"])
def sample_dataset():
    return send_file(
        "sample_data.csv",
        mimetype="text/csv",
        as_attachment=True,
        download_name="patternlens_sample_data.csv",
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded = request.files.get("dataset")
    if not uploaded or not uploaded.filename:
        flash("Please choose a CSV file.")
        return redirect(url_for("index"))
    if not uploaded.filename.lower().endswith(".csv"):
        flash("Only CSV files are supported.")
        return redirect(url_for("index"))
    if request.content_length and request.content_length > MAX_UPLOAD_BYTES:
        flash("The file is too large. Maximum size is 10 MB.")
        return redirect(url_for("index"))

    try:
        data = pd.read_csv(uploaded)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        flash(f"Could not read the CSV file: {error}")
        return redirect(url_for("index"))

    data = data.dropna(how="all")
    if data.empty or len(data.columns) < 2:
        flash("The dataset must contain at least two columns and one data row.")
        return redirect(url_for("index"))

    target_name = request.form.get("target")
    if target_name not in data.columns:
        flash("Select a valid prediction target.")
        return redirect(url_for("index"))

    data = data.dropna(subset=[target_name])
    if len(data) < 4:
        flash("At least four rows with a target value are required.")
        return redirect(url_for("index"))

    features = data.drop(columns=[target_name])
    target = data[target_name]
    usable_features = features.dropna(axis=1, how="all")
    if usable_features.empty:
        flash("The dataset needs at least one usable feature column.")
        return redirect(url_for("index"))

    try:
        model, task = build_model(usable_features, target)
        split_at = max(2, int(len(data) * 0.8))
        train_x, test_x = usable_features.iloc[:split_at], usable_features.iloc[split_at:]
        train_y, test_y = target.iloc[:split_at], target.iloc[split_at:]
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        if task == "regression":
            metrics = {
                "Metric": ["Mean absolute error", "R² score"],
                "Value": [
                    round(float(mean_absolute_error(test_y, predictions)), 4),
                    round(float(r2_score(test_y, predictions)), 4),
                ],
            }
        else:
            metrics = {
                "Metric": ["Accuracy"],
                "Value": [round(float(accuracy_score(test_y, predictions)), 4)],
            }
        patterns = discover_patterns(data)
        preview = pd.DataFrame(
            {"Actual": test_y.to_numpy(), "Predicted": predictions}
        ).head(10)
    except (ValueError, TypeError) as error:
        flash(f"Analysis failed: {error}")
        return redirect(url_for("index"))

    result = {
        "filename": uploaded.filename,
        "rows": len(data),
        "columns": len(data.columns),
        "target": target_name,
        "task": task,
        "features": usable_features.columns.tolist(),
        "metrics": metrics,
        "patterns": patterns,
        "preview": preview.to_dict(orient="records"),
    }
    return render_template("results.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
