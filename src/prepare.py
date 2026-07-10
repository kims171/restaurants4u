import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def load_config():
    """Loads the centralized MLOps pipeline parameters."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def prepare_data():
    config = load_config()["prepare"]
    
    print("Ingesting raw tabular restaurant dataset records...")
    # Read the dataset from the path configured in params.yaml
    df = pd.read_csv(config["raw_input_path"])
    
    print("Enforcing explicit data-type casting and scrubbing corrupted entries...")
    # FIX: Force critical columns to numeric floats. 
    # Any text anomalies (like repeated headers or "None" strings) will safely turn into NaN.
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    
    # Drop rows that don't have valid ratings or geospatial coordinates
    df = df.dropna(subset=["Rating", "Latitude", "Longitude"])
    
    print("Engineering target and structural features...")
    # Target Construction: 1 if Rating >= 4.0 (Highly Rated), else 0
    df["is_highly_rated"] = (df["Rating"] >= 4.0).astype(int)
    
    # Convert presence of operational fields into binary flags
    df["has_website"] = df["Website"].notna().astype(int)
    df["has_phone"] = df["Phone"].notna().astype(int)
    df["has_images"] = df["Images"].notna().astype(int)
    
    # Frequency Capping: Group sparse categories into "Other" to prevent high cardinality explosion
    top_categories = df["Category"].value_counts().index[:20]
    df["clean_category"] = df["Category"].apply(lambda x: x if x in top_categories else "Other")
    
    # One-Hot Encode categorical cuisine types
    category_encoded = pd.get_dummies(df["clean_category"], prefix="cat")
    
    # Consolidate our target and numeric engineering matrix
    base_numeric = df[["Latitude", "Longitude", "has_website", "has_phone", "has_images", "is_highly_rated"]]
    features_df = pd.concat([base_numeric, category_encoded], axis=1)
    
    print("Executing Stratified Split partitions...")
    # Split features and labels using a stratified distribution based on our target flag
    train_df, test_df = train_test_split(
        features_df, 
        test_size=config["test_size"], 
        random_state=config["seed"],
        stratify=features_df["is_highly_rated"]
    )
    
    # Create target directory if it doesn't exist
    os.makedirs(config["processed_dir"], exist_ok=True)
    
    # Save train and test sets into high-performance parquet formats
    train_path = os.path.join(config["processed_dir"], "train.parquet")
    test_path = os.path.join(config["processed_dir"], "test.parquet")
    
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    
    print(f"Data compilation complete. Datasets cached successfully:")
    print(f"  - Train Shape: {train_df.shape} -> saved to {train_path}")
    print(f"  - Test Shape:  {test_df.shape} -> saved to {test_path}")

if __name__ == "__main__":
    prepare_data()