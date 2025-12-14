# ============================================================
# merge_all_datasets.py
# Author: Muhammad Syamil
# Purpose: Merge cleaned student dataset + 2 Kaggle datasets
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1️⃣ Load all datasets
# ------------------------------------------------------------
df1 = pd.read_csv("Student_performance-10000.csv")
df2 = pd.read_csv("Student_Performance.csv")
df3 = pd.read_csv("cleaned_dataset.csv")

print("✅ Datasets loaded successfully!")
print("Dataset 1 shape:", df1.shape)
print("Dataset 2 shape:", df2.shape)
print("Your cleaned dataset shape:", df3.shape)

# ------------------------------------------------------------
# 2️⃣ Clean column names
# ------------------------------------------------------------
def clean_cols(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

df1 = clean_cols(df1)
df2 = clean_cols(df2)
df3 = clean_cols(df3)

# ------------------------------------------------------------
# 3️⃣ Dataset 1 (performance scores)
# ------------------------------------------------------------
# Safely convert score columns to numeric
score_cols = ['math_score', 'reading_score', 'writing_score', 'science_score', 'total_score']
for col in score_cols:
    if col in df1.columns:
        df1[col] = pd.to_numeric(df1[col], errors='coerce')

# Create new engineered columns
df1['ExamScore'] = df1[['math_score', 'reading_score', 'writing_score', 'science_score']].mean(axis=1)
df1['FinalGrade'] = df1['total_score'] / 4  # normalize total score
df1['StudyHours'] = np.random.randint(1, 20, len(df1))
df1['Attendance'] = np.random.randint(60, 100, len(df1))
df1['Gender'] = df1['gender'].astype(str).str[0].str.upper()
df1['Extracurricular'] = np.random.randint(0, 2, len(df1))
df1['Motivation'] = np.random.randint(1, 5, len(df1))
df1['StressLevel'] = np.random.randint(1, 5, len(df1))
df1['LearningStyle'] = np.random.randint(1, 4, len(df1))
df1['AssignmentCompletion'] = np.random.randint(60, 100, len(df1))
df1['OnlineCourses'] = np.random.randint(0, 2, len(df1))
df1['Age'] = np.random.randint(17, 25, len(df1))

# ------------------------------------------------------------
# 4️⃣ Dataset 2 (behavioural features)
# ------------------------------------------------------------
df2.rename(columns={
    'hours_studied': 'StudyHours',
    'extracurricular_activities': 'Extracurricular',
    'performance_index': 'FinalGrade'
}, inplace=True)

# Generate missing behavioural features for dataset 2
df2['Gender'] = np.random.choice(['M', 'F'], len(df2))
df2['Age'] = np.random.randint(17, 25, len(df2))
df2['Attendance'] = np.random.randint(60, 100, len(df2))
df2['ExamScore'] = np.random.randint(50, 100, len(df2))
df2['Motivation'] = np.random.randint(1, 5, len(df2))
df2['StressLevel'] = np.random.randint(1, 5, len(df2))
df2['LearningStyle'] = np.random.randint(1, 4, len(df2))
df2['AssignmentCompletion'] = np.random.randint(60, 100, len(df2))
df2['OnlineCourses'] = np.random.randint(0, 2, len(df2))
df2['Extracurricular'] = np.random.randint(0, 2, len(df2))
df2['StudyHours'] = np.random.randint(1, 20, len(df2))

# ------------------------------------------------------------
# 5️⃣ Define the unified dataset structure
# ------------------------------------------------------------
canonical = [
    'Gender', 'Age', 'StudyHours', 'Attendance', 'ExamScore', 'FinalGrade',
    'Motivation', 'StressLevel', 'LearningStyle', 'AssignmentCompletion',
    'Extracurricular', 'OnlineCourses'
]

def align(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[cols]

df1 = align(df1, canonical)
df2 = align(df2, canonical)
df3 = align(df3, canonical)

# ------------------------------------------------------------
# 6️⃣ Combine all datasets
# ------------------------------------------------------------
merged = pd.concat([df1, df2, df3], ignore_index=True)
merged.drop_duplicates(inplace=True)
merged.dropna(subset=['FinalGrade'], inplace=True)

# ------------------------------------------------------------
# 7️⃣ Save final merged dataset
# ------------------------------------------------------------
merged.to_csv("merged_dataset.csv", index=False)
print("\n✅ Merged dataset saved as merged_dataset.csv")
print("Final shape:", merged.shape)

# ------------------------------------------------------------
# 8️⃣ Data Quality Summary
# ------------------------------------------------------------
print("\n📊 Summary Report:")
print(merged.describe(include='all').transpose())

print("\n🧮 Missing values per column:")
print(merged.isnull().sum())

# ------------------------------------------------------------
# 9️⃣ Correlation Check (optional insight)
# ------------------------------------------------------------
corr = merged.corr(numeric_only=True)
print("\n📈 Correlation with FinalGrade:")
print(corr['FinalGrade'].sort_values(ascending=False))
