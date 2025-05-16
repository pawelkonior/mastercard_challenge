from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

warnings.filterwarnings("ignore")

# 1. Load data
transactions = pd.read_parquet(DATA_PROCESSED_DIR / "transactions.parquet")
users = pd.read_parquet(DATA_PROCESSED_DIR / "users.parquet")
merchants = pd.read_parquet(DATA_PROCESSED_DIR / "merchants.parquet")

# 2. Merge data
df = transactions.merge(users, on='user_id', how='left') \
                 .merge(merchants, on='merchant_id', how='left')

# 3. Quick exploration
print("Data shape:", df.shape)
print("Missing values:\n", df.isna().sum())
print("Target distribution:\n", df['is_fraud'].value_counts(normalize=True))

# 4. Feature Engineering
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Drop uninformative or leaked columns
drop_cols = ['transaction_id', 'timestamp', 'user_id', 'merchant_id']
df = df.drop(columns=drop_cols)

# Encode categorical features
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# 5. Train/test split
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 6. XGBoost model
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)

# 7. Predictions and evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# 8. Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9. Feature importance
xgb.plot_importance(model, max_num_features=10, importance_type='gain')
plt.title("Top 10 Feature Importances")
plt.show()
