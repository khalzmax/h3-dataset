import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- Configuration ---
DATASET_DIR = './dataset'
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, 'annotations')
TRAIN_CSV = os.path.join(ANNOTATIONS_DIR, 'train.csv')
VAL_CSV = os.path.join(ANNOTATIONS_DIR, 'val.csv')
TEST_CSV = os.path.join(ANNOTATIONS_DIR, 'test.csv')

# Load CSVs
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

# Combine all splits for global stats
all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print('Train samples:', len(train_df))
print('Val samples:  ', len(val_df))
print('Test samples: ', len(test_df))
print('Total samples:', len(all_df))

# Units distribution across different sets
unit_distribution = pd.DataFrame({
    'Train': train_df['Unit_name'].value_counts(),
    'Val': val_df['Unit_name'].value_counts(),
    'Test': test_df['Unit_name'].value_counts(),
    'Total': all_df['Unit_name'].value_counts()
}).fillna(0).astype(int)

print('Units distribution across sets:')
print(unit_distribution)

# Plot units distribution
unit_distribution.plot(kind='bar', figsize=(12, 6))
plt.title('Units Distribution Across Sets')
plt.xlabel('Unit Name')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 1. Class distribution
plt.figure(figsize=(12, 5))
all_df['Unit_name'].value_counts().plot(kind='bar')
plt.title('Class Distribution (All Splits)')
plt.xlabel('Unit Name')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2. Frame size distribution
sizes = []
for path in all_df['File_Path']:
    img_path = os.path.join(DATASET_DIR, path)
    try:
        with Image.open(img_path) as img:
            sizes.append(img.size)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        continue
sizes = np.array(sizes)
if len(sizes) > 0:
    plt.figure(figsize=(8, 5))
    plt.scatter(sizes[:,0], sizes[:,1], alpha=0.2)
    plt.xlabel('Width (px)')
    plt.ylabel('Height (px)')
    plt.title('Frame Size Distribution')
    plt.grid(True)
    plt.show()
    print('Mean size:', np.mean(sizes, axis=0))
    print('Median size:', np.median(sizes, axis=0))
else:
    print('No images found for size analysis.')

# 3. Visualize random samples per class
def show_random_samples(df, class_name, n=10):
    samples = df[df['Unit_name'] == class_name].sample(min(n, df[df['Unit_name'] == class_name].shape[0]))
    plt.figure(figsize=(15, 2))
    for i, (_, row) in enumerate(samples.iterrows()):
        img_path = os.path.join(DATASET_DIR, row['File_Path'])
        try:
            img = Image.open(img_path)
            plt.subplot(1, n, i+1)
            plt.imshow(img)
            plt.axis('off')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    plt.suptitle(f'Random samples for class: {class_name}')
    plt.show()
    # plt.savefig(f"./random_samples-{class_name}.png")

# Show samples for a few classes
# for class_name in all_df['Unit_name'].value_counts().index[:5]:
#     show_random_samples(all_df, class_name, n=8)
