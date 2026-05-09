import pandas as pd

# Load original datasets
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Add labels
true_df["label"] = 0
fake_df["label"] = 1

# Merge datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Keep only required columns
df = df[["text", "label"]]

# Save final dataset
df.to_csv("fake_news.csv", index=False)

print("Dataset merged successfully!")
print(df.head())