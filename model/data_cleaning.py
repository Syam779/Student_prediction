# data_cleaning.py

import pandas as pd

# Load dataset
df = pd.read_csv("merged_dataset.csv")

# Display basic info
print("Initial shape:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)

# Handle missing values (example)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Convert categorical columns if needed
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

# Display cleaned info
print("\nCleaned shape:", df.shape)
print(df.head())

# Save cleaned data
df.to_csv("cleaned_dataset.csv", index=False)
print("\n✅ Cleaned dataset saved as cleaned_dataset.csv")
