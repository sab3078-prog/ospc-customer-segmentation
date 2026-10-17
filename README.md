#  AI/ML CLUB PROJECT FOR RECRUITMENT

## Customer Segmentation Prediction

This project predicts customer cluster labels for the OSPC AI/ML Recruitment Round 2 Kaggle competition.

## Objective

The goal is to predict the `Cluster` value for every customer in `test.csv`.
The competition reference labels were generated using K-Means clustering.

## Dataset Features

The dataset contains:

- `CustomerID` — unique customer identifier
- `Gender` — customer gender
- `Age` — customer age
- `Annual Income (k$)` — annual income in thousands of dollars
- `Spending Score (1-100)` — customer spending score
- `Cluster` — target column available in `train.csv`

`CustomerID` is used only to identify customers and is not used as a model feature.

## Methodology

1. Loaded `train.csv` and `test.csv`.
2. Tested several feature combinations.
3. Used annual income and spending score as the primary clustering features.
4. Compared scaled and unscaled features.
5. Applied K-Means clustering.
6. Matched arbitrary K-Means cluster numbers to the competition labels.
7. Selected the configuration with the highest training agreement.
8. Generated predictions for all test customers.
9. Saved the final predictions to `submission.csv`.

## Project Structure

```text
oscp-customer-segmentation/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── src/
│   └── kmeans_model.py
├── submission.csv
├── README.md
└── requirements.txt
```

## How to Run

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install pandas scikit-learn scipy
```

Run the model:

```bash
python3 src/kmeans_model.py
```

The script creates `submission.csv` in the project folder.

## Submission Format

The submission file contains exactly two columns:

```text
CustomerID,Cluster
```

Every customer from `test.csv` appears exactly once.

## Result

The final model achieved a public Kaggle score of approximately **0.95**.

