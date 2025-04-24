import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Use a relative path to load the data
# ⚠️ This assumes the script is run from the root of the repo (where ML.py is located) on your local machine
# and that the CSV file is inside the 'raw_data/' folder.

file_path = os.path.join("processed_data", "obesity_cleaned.csv")
df = pd.read_csv(file_path)

#Define target and features 
target_col = "obesity_level"
y = df[target_col] #keep as categorical for multi-class classification
X = df.drop(columns=[target_col])

categorical_cols = [
    "gender", "family_history_overweight", "high_caloric_food_freq", 
    "vegetables_freq", "main_meal_count", "snacking_freq", "smokes",
    "water_intake", "calorie_tracking", "physical_activity_freq",
    "screen_time_hours", "alcohol_consumption_freq", "transport_mode"
]

#perform one-hot encoding 
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

#Split the dataset 
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

#scale (fit only on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Convert scaled arrays to dataframes
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Check the result
print("Training set shape:", X_train_scaled_df.shape)
print("Testing set shape:", X_test_scaled_df.shape)

#Combine the features and labels for export 
train_data = pd.concat([X_train_scaled_df, y_train.reset_index(drop = True)], axis = 1)
test_data = pd.concat([X_test_scaled_df, y_test.reset_index(drop = True)], axis = 1)

#Save to CSV
train_data.to_csv("processed_data/train_data.csv", index=False)
test_data.to_csv("processed_data/test_data.csv", index=False)

print("Training and testing data saved to 'processed_data/' folder.")