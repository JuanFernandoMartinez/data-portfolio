import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

data_dir = "cancerSeno_bw/test"
output_dir = "featuresTestFromTest"
classes = ["sano", "cancer"]

# Create output directories for unified data
for cls in classes:
    os.makedirs(os.path.join(output_dir, cls, "UnifiedFeatures"), exist_ok=True)

def read_csv_to_df(class_name, feature):
    """Read CSV file into a dataframe."""
    file_path = os.path.join(output_dir, class_name, feature, f"combined_{feature}.csv")
    return pd.read_csv(file_path)


data_frames = {cls: {} for cls in classes}

for cls in classes:
    data_frames[cls]['pixel_density'] = read_csv_to_df(cls, "pixelDensity")
    data_frames[cls]['brightness'] = read_csv_to_df(cls, "brightness")
    data_frames[cls]['std_deviation'] = read_csv_to_df(cls, "stdDeviation")

# Combine dataframes for each class
for cls in classes:
    combined_df = pd.concat([
        data_frames[cls]['pixel_density'],
        data_frames[cls]['brightness'],
        data_frames[cls]['std_deviation']
    ], axis=1)
    combined_df.columns = ['pixel_density', 'brightness', 'std_deviation']
    combined_df.to_csv(os.path.join(output_dir, cls, "UnifiedFeatures", "combined_features.csv"), index=False)


combined_sano_df = pd.read_csv(os.path.join(output_dir, "sano", "UnifiedFeatures", "combined_features.csv"))
combined_cancer_df = pd.read_csv(os.path.join(output_dir, "cancer", "UnifiedFeatures", "combined_features.csv"))

# Add diagnosis column
combined_sano_df['diagnosis'] = 0
combined_cancer_df['diagnosis'] = 1

# Concatenate both dataframes
final_df = pd.concat([combined_sano_df, combined_cancer_df], ignore_index=True)

# Drop rows with NaN values
final_df.dropna(inplace=True)

# Save the final dataframe
final_df.to_csv(os.path.join(output_dir, "combined_features_test.csv"), index=False)

print("done")