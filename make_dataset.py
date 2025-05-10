import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
from tqdm import tqdm
import ast  # For safely evaluating string representations of tuples

# --- Configuration ---
INPUT_ANNOTATIONS_FILE = './output_patches/annotations_extended.csv'
OUTPUT_DATASET_DIR = './dataset'
VAL_RATIO = 0.15  # 15% for validation
TEST_RATIO = 0.15  # 15% for testing
TRAIN_RATIO = 0.70  # 70% for training
RANDOM_SEED = 42

def create_dataset():
    """
    Creates a dataset structure with train, validation, and test splits
    while ensuring all unit_avatar frames stay in the training set.
    """
    print(f"Creating dataset in {OUTPUT_DATASET_DIR}")
    
    # Create directory structure
    dataset_path = Path(OUTPUT_DATASET_DIR)
    annotations_path = dataset_path / "annotations"
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"
    test_path = dataset_path / "test"
    
    for path in [dataset_path, annotations_path, train_path, val_path, test_path]:
        path.mkdir(parents=True, exist_ok=True)
        
    # Load the annotations file
    print(f"Loading annotations from {INPUT_ANNOTATIONS_FILE}")
    df = pd.read_csv(INPUT_ANNOTATIONS_FILE)
    
    # Fix Frame_ID: convert from float to integer
    if 'Frame_ID' in df.columns:
        df['Frame_ID'] = df['Frame_ID'].fillna(0).astype(int)
    
    # Extract Width and Height from Box column
    if 'Box' in df.columns:
        # Function to extract width and height from box tuple
        def extract_wh(box_str):
            try:
                if isinstance(box_str, str):
                    # Parse string representation of tuple "(x, y, w, h)"
                    box = ast.literal_eval(box_str)
                    return box[2], box[3]  # w, h
                return None, None
            except:
                return None, None
        
        # Apply the function to each row
        df[['Width', 'Height']] = df.apply(lambda row: pd.Series(extract_wh(row['Box'])), axis=1)
        
        # Drop Box column after extraction
        df = df.drop(columns=['Box'])
    
    # Drop rows with NaN in File_Path
    original_count = len(df)
    df = df.dropna(subset=['File_Path'])
    if len(df) < original_count:
        print(f"Dropped {original_count - len(df)} rows with missing File_Path values")
    
    # Handle potential missing columns
    if 'frame_type' not in df.columns:
        print("Warning: 'frame_type' column not found in CSV. Adding default value 'unit'")
        df['frame_type'] = 'unit'
    
    # Separate avatar frames (these all go to training set)
    avatar_df = df[df['frame_type'] == 'unit_avatar'].copy()
    regular_df = df[df['frame_type'] != 'unit_avatar'].copy()
    
    print(f"Found {len(avatar_df)} avatar frames - all will go to training set")
    print(f"Found {len(regular_df)} regular unit frames - these will be split")
    
    # Get unique units for splitting
    unique_units = regular_df['Unit_name'].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(unique_units)
    
    # Calculate split counts
    total_units = len(unique_units)
    val_count = int(total_units * VAL_RATIO)
    test_count = int(total_units * TEST_RATIO)
    train_count = total_units - val_count - test_count
    
    # Split units by name
    train_units = unique_units[:train_count]
    val_units = unique_units[train_count:train_count+val_count]
    test_units = unique_units[train_count+val_count:]
    
    print(f"Split units: {train_count} training, {val_count} validation, {test_count} testing")
    
    # Create split dataframes based on unit name
    train_df = pd.concat([
        avatar_df,
        regular_df[regular_df['Unit_name'].isin(train_units)]
    ])
    val_df = regular_df[regular_df['Unit_name'].isin(val_units)]
    test_df = regular_df[regular_df['Unit_name'].isin(test_units)]
    
    print(f"Final split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} testing")
    
    # Function to copy files and update paths
    def process_split(split_df, split_name):
        split_dir = dataset_path / split_name
        processed_rows = []
        
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name} split"):
            # Get source file path
            src_path = row['File_Path']
            unit_name = row['Unit_name']
            
            # Skip if src_path is not valid
            if not isinstance(src_path, str):
                print(f"Warning: Invalid path ({src_path}) for unit {unit_name}. Skipping.")
                continue
            
            # Create unit directory in split folder if it doesn't exist
            unit_dir = split_dir / unit_name
            unit_dir.mkdir(exist_ok=True)
            
            # Get filename and destination path
            filename = os.path.basename(src_path)
            dst_path = unit_dir / filename
            
            # Copy the file
            try:
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                else:
                    print(f"Warning: Source file not found: {src_path}")
                    continue
            except Exception as e:
                print(f"Error copying {src_path} to {dst_path}: {e}")
                continue
                
            # Update path in dataframe to relative path from dataset root
            row_dict = row.to_dict()
            row_dict['File_Path'] = f"{split_name}/{unit_name}/{filename}"
            processed_rows.append(row_dict)
            
        # Create new dataframe with updated paths
        return pd.DataFrame(processed_rows)
    
    # Process each split
    train_df_processed = process_split(train_df, "train")
    val_df_processed = process_split(val_df, "val")
    test_df_processed = process_split(test_df, "test")
    
    # Save CSV annotations
    train_df_processed.to_csv(annotations_path / "train.csv", index=False)
    val_df_processed.to_csv(annotations_path / "val.csv", index=False)
    test_df_processed.to_csv(annotations_path / "test.csv", index=False)
    
    print(f"Dataset creation complete. Files organized in {OUTPUT_DATASET_DIR}")
    print(f"CSV annotations saved in {annotations_path}")

if __name__ == "__main__":
    create_dataset()