import pandas as pd

def clean_data(input_file, output_file):
    df = pd.read_csv(input_file)

    # Drop duplicates
    df = df.drop_duplicates()

    # Remove rows with too many missing values
    df = df.dropna(thresh=len(df.columns) - 2)

    # Fill missing with column median
    for col in df.select_dtypes(include=["float", "int"]).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Convert Gender to numeric if needed
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"male": 1, "female": 0, "Male": 1, "Female": 0})

    # Normalize StudyHours (example)
    if "StudyHours" in df.columns:
        df["StudyHours"] = df["StudyHours"].clip(0, 40)

    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned dataset saved to: {output_file}")


if __name__ == "__main__":
    clean_data("data/merged_dataset.csv", "data/cleaned_dataset.csv")
