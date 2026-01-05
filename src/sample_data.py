import pandas as pd
import numpy as np

def stratified_sample(input_csv: str, output_csv: str, target_n: int = 12000, min_per_category: int = 4):
    """
    Create a stratified sample of the cleaned complaints.
    Proportional sampling with a minimum per category.
    """
    df = pd.read_csv(input_csv)
    print(f"Total cleaned complaints: {len(df)}")
    print(df["Product"].value_counts())

    # Compute proportional fractions
    fractions = (df.groupby("Product").size() / len(df)) * (target_n / len(df))

    # Apply sampling with minimum per category
    df_sample = df.groupby("Product", group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), max(min_per_category, int(np.floor(fractions.loc[x.name] * len(x))))),
            random_state=42
        )
    )

    print("Sampled size by Product:")
    print(df_sample["Product"].value_counts())
    print("Total sample size:", len(df_sample))

    df_sample.to_csv(output_csv, index=False)
    print(f"Saved stratified sample to {output_csv}")

if __name__ == "__main__":
    stratified_sample(
        input_csv="data/processed/filtered_complaints.csv",
        output_csv="data/processed/sample_complaints.csv"
    )
