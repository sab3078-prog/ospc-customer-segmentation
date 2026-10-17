import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment


# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

target = "Cluster"
id_column = "CustomerID"

y = train[target]
test_ids = test[id_column]

number_of_clusters = y.nunique()

# Different feature configurations to test
candidates = [
    (
        "income_spending_scaled",
        ["Annual Income (k$)", "Spending Score (1-100)"],
        True
    ),
    (
        "age_income_spending_scaled",
        ["Age", "Annual Income (k$)", "Spending Score (1-100)"],
        True
    ),
    (
        "income_spending_unscaled",
        ["Annual Income (k$)", "Spending Score (1-100)"],
        False
    ),
    (
        "all_features_scaled",
        [
            "Gender",
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ],
        True
    )
]

best_score = -1
best_name = None
best_test_predictions = None

for name, selected_features, use_scaling in candidates:

    train_features = train[selected_features].copy()
    test_features = test[selected_features].copy()

    # Convert Gender to numbers when it is included
    for dataframe in [train_features, test_features]:
        if "Gender" in dataframe.columns:
            dataframe["Gender"] = dataframe["Gender"].map({
                "Male": 0,
                "Female": 1
            })

    # Combine train and test so they use identical preprocessing
    combined = pd.concat(
        [train_features, test_features],
        ignore_index=True
    )

    combined = pd.get_dummies(combined)
    combined = combined.fillna(combined.median(numeric_only=True))

    if use_scaling:
        scaler = StandardScaler()
        combined_values = scaler.fit_transform(combined)
    else:
        combined_values = combined.values

    train_values = combined_values[:len(train)]
    test_values = combined_values[len(train):]

    # K-Means
    kmeans = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=50
    )

    all_predictions = kmeans.fit_predict(combined_values)

    train_clusters = all_predictions[:len(train)]
    test_clusters = all_predictions[len(train):]

    # Match arbitrary K-Means IDs to competition labels
    label_values = sorted(y.unique())

    confusion = pd.crosstab(
        train_clusters,
        y
    ).reindex(
        index=range(number_of_clusters),
        columns=label_values,
        fill_value=0
    )

    rows, columns = linear_sum_assignment(
        -confusion.to_numpy()
    )

    mapping = {
        int(row): int(confusion.columns[column])
        for row, column in zip(rows, columns)
    }

    aligned_train = pd.Series(train_clusters).map(mapping)
    aligned_test = pd.Series(test_clusters).map(mapping)

    score = accuracy_score(y, aligned_train)

    print(f"{name}: {score:.4f}")

    if score > best_score:
        best_score = score
        best_name = name
        best_test_predictions = aligned_test


# Create final submission
submission = pd.DataFrame({
    "CustomerID": test_ids,
    "Cluster": best_test_predictions.astype(int)
})

submission.to_csv("submission.csv", index=False)

print()
print("Best method:", best_name)
print("Best training accuracy:", best_score)
print("submission.csv created successfully")
print(submission.head())
print("Submission rows:", len(submission))