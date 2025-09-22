import pytest
import sqlite3
import pandas as pd
from vsql import VSQLApp  # assuming your main app class is VSQLApp

# ---------- 1. Database Connection ----------
def test_database_connection(tmp_path):
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(db_file)
    assert conn is not None
    conn.close()

# ---------- 2. SQL Query Execution ----------
def test_sql_query_execution():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER, name TEXT)")
    cursor.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.commit()

    cursor.execute("SELECT name FROM users WHERE id=1")
    result = cursor.fetchone()
    assert result[0] == "Alice"
    conn.close()

# ---------- 3. Load Data into Pandas ----------
def test_load_data_to_dataframe():
    conn = sqlite3.connect(":memory:")
    pd.DataFrame({"id": [1, 2], "value": ["A", "B"]}).to_sql("test_table", conn)
    df = pd.read_sql("SELECT * FROM test_table", conn)
    assert len(df) == 2
    conn.close()

# ---------- 4. Data Cleaning ----------
def test_data_cleaning():
    df = pd.DataFrame({"id": [1, 2, None], "value": ["A", None, "C"]})
    cleaned_df = df.dropna()
    assert cleaned_df.shape[0] == 1

# ---------- 5. Train ML Model (Regression) ----------
def test_train_regression_model():
    from sklearn.linear_model import LinearRegression
    X = [[1], [2], [3], [4]]
    y = [2, 4, 6, 8]
    model = LinearRegression().fit(X, y)
    prediction = model.predict([[5]])
    assert round(prediction[0]) == 10

# ---------- 6. Train ML Model (Classification) ----------
def test_train_classification_model():
    from sklearn.tree import DecisionTreeClassifier
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    model = DecisionTreeClassifier().fit(X, y)
    prediction = model.predict([[2]])
    assert prediction[0] == 1

# ---------- 7. Visualization Output ----------
def test_visualization_output(tmp_path):
    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3], [2, 4, 6])
    file_path = tmp_path / "plot.png"
    plt.savefig(file_path)
    assert file_path.exists()

# ---------- 8. Clustering ----------
def test_clustering_kmeans():
    from sklearn.cluster import KMeans
    import numpy as np
    X = np.array([[1], [2], [10], [11]])
    model = KMeans(n_clusters=2, random_state=42).fit(X)
    labels = model.labels_
    assert len(set(labels)) == 2

# ---------- 9. Anomaly Detection ----------
def test_anomaly_detection():
    from sklearn.ensemble import IsolationForest
    import numpy as np
    X = [[1], [2], [3], [100]]
    model = IsolationForest(random_state=42).fit(X)
    preds = model.predict(X)
    assert -1 in preds  # anomaly detected

# ---------- 10. Save & Load Model ----------
def test_model_save_load(tmp_path):
    from sklearn.linear_model import LogisticRegression
    import joblib

    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    model = LogisticRegression().fit(X, y)

    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)
    loaded_model = joblib.load(model_path)

    pred = loaded_model.predict([[2]])
    assert pred[0] == 1
