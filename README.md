# Mental Health Signal — Student Wellness Prediction

A machine-learning application that estimates a student mental-health score from self-reported lifestyle, academic, stress, and social-media usage features. It combines exploratory data analysis, a trained scikit-learn regression model, a FastAPI prediction endpoint, and a browser-based interface.
## Features

- Exploratory data analysis of student social-media and lifestyle data.
- Regression model experiments targeting the numerical `Mental_Health_Score`.
- Preprocessing with a scikit-learn `ColumnTransformer` pipeline.
- FastAPI endpoint for validating input and returning predicted scores.
- HTML, CSS, and JavaScript frontend for submitting inputs and displaying the result.

## Project structure

```text
mental-healthpredn/
├── ML_project .ipynb
├── Student Social Media And Mental Health Impact.csv
├── Mental_Health_Model.pkl
├── main.py
├── requirements.txt
├── index.html
├── style.css
└── script.js
```

Keep local environment folders such as `venv/` and generated files such as `__pycache__/` out of version control.

## Dataset

The included CSV has **5,000 rows** and 13 columns:

- **Profile:** `Age`, `Gender`, `Country`, `Academic_Level`
- **Digital habits:** `Most_Used_Platform`, `Purpose_Of_Use`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`
- **Lifestyle and stress:** `Study_Hours`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night`, `Stress_Level`
- **Target:** `Mental_Health_Score`

The notebook explores distributions and relationships, checks data quality, considers outliers, groups less common countries, and builds preprocessing and regression pipelines.

## Machine-learning approach

The notebook compares Linear Regression and a Random Forest Regressor. The preprocessing pipeline uses:

- Log transformation and standardization for `Study_Hours`
- Standardization for other numerical features
- Ordinal encoding for `Stress_Level`
- One-hot encoding for nominal categorical features

The notebook uses a 70/30 train-test split with `random_state=42`. Reported results for the default Random Forest model:

| Metric | Test result |
|---|---:|
| R² | 0.8776 |
| Mean Absolute Error (MAE) | 0.3472 |
| Root Mean Squared Error (RMSE) | 0.4637 |

These metrics reflect one dataset split and do not guarantee real-world performance. The training R² (about 0.9808) is notably higher than the test R² and may indicate overfitting. The tuned model's reported test R² is about 0.8650. Evaluation on independent data is needed before drawing broader conclusions.

## Requirements

- Python 3.10 or later recommended
- pip
- The model file `Mental_Health_Model.pkl`

Dependencies are listed in `requirements.txt`, including FastAPI, Uvicorn, Pydantic, pandas, joblib, and scikit-learn.

## Run locally

1. Clone the repository:

   ```bash
   git clone https://github.com/anuragmishra77/mental-healthpredn.git
   cd mental-healthpredn
   ```

2. Create and activate a virtual environment.

   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the API from the project root:

   ```bash
   uvicorn main:app --reload
   ```

5. Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation, or [http://127.0.0.1:8000/](http://127.0.0.1:8000/) for the root endpoint.

The model file must be in the project root because `main.py` loads it at startup. Only load joblib/pickle model files from trusted sources and use compatible library versions.

## Prediction API

Send a JSON request to `POST /predict`.

Example request:

```json
{
  "age": 21,
  "gender": "Male",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 4.5,
  "daily_unlocks": 80,
  "study_hours": 4.0,
  "physical_activity_hours": 1.0,
  "sleep_hours_per_night": 7.5,
  "stress_level": "Medium"
}
```

Example response shape (illustrative only):

```json
{
  "predicted_mental_health_score": 6.5
}
```

The displayed number is not a prediction for the sample input. Check `/docs` and the input model in `main.py` for accepted fields and categories.

## Frontend

The interface uses plain HTML, CSS, and JavaScript. Open `index.html` for the static page. In `script.js`, set `API_BASE` to `http://127.0.0.1:8000` to use a locally running API. For deployment, configure CORS to allow only the frontend origins you trust.

## Limitations and responsible use

- The output is a model estimate, not a mental-health diagnosis.
- Associations do not establish that screen time, sleep, stress, or other features cause a mental-health outcome.
- The data may not represent all student populations; model performance may differ across groups.
- Validate the pipeline on independent data before any consequential use.
- Treat mental-health-related data as sensitive and avoid collecting unnecessary personal information.
