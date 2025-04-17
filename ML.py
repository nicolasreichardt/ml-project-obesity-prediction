import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Use a relative path to load the data
# ⚠️ This assumes the script is run from the root of the repo (where ML.py is located) on your local machine
# and that the CSV file is inside the 'raw_data/' folder.

file_path = os.path.join("raw_data", "obesity_prediction.csv")
df = pd.read_csv(file_path)

categorical_cols = [
    "Gender", "family_history", "FAVC", "CAEC", "SMOKE",
    "SCC", "CALC", "MTRANS", "Obesity"
]

#perform hot-one encoding 
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Encoded DataFrame shape:", df_encoded.shape)
print(df_encoded.head())

X = df_encoded
y = df["Obesity"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns)
print(X_scaled_df.head())

#Split the dataset 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42
)

# Check the result
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

#python /Users/ashleyrazo/Desktop/ML.py