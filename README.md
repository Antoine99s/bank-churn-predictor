# bank-churn-predictor

A simple end-to-end churn prediction project using scikit-learn, FastAPI, and Streamlit.

## Project structure

```text
bank-churn-predictor/
├── data/
│   └── bank_churn.csv
├── notebooks/
│   └── 01_eda.ipynb
├── training/
│   └── train.py
├── app/
│   ├── main.py
│   ├── schema.py
│   └── model.pkl            # generated after training
├── dashboard/
│   └── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train model

Run from repository root:

```bash
python training/train.py
```

This reads `data/bank_churn.csv` and writes `app/model.pkl`.

## Run API

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Female",
    "Age": 40,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 100000
  }'
```

## Run dashboard

```bash
streamlit run dashboard/app.py
```

Set custom API URL if needed:

```bash
API_URL=http://localhost:8000 streamlit run dashboard/app.py
```

## Docker (FastAPI app)

Build:

```bash
docker build -t bank-churn-api .
```

Run:

```bash
docker run -p 8000:8000 bank-churn-api
```

`app/model.pkl` is expected at runtime. If missing, run `python training/train.py` before starting the API.
