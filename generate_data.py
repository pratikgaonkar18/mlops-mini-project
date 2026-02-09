from sklearn.datasets import load_iris
import pandas as pd
import os

# Load dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Create data directory if not exists
os.makedirs("data", exist_ok=True)

# Save dataset
df.to_csv("data/iris.csv", index=False)

print("Dataset saved to data/iris.csv")
