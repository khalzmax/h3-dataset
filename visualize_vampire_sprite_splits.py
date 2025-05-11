import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# --- Configuration ---
DATASET_DIR = './dataset'
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, 'annotations')
TRAIN_CSV = os.path.join(ANNOTATIONS_DIR, 'train.csv')
VAL_CSV = os.path.join(ANNOTATIONS_DIR, 'val.csv')
TEST_CSV = os.path.join(ANNOTATIONS_DIR, 'test.csv')
SPRITES_DIR = '../datasets/heroes3_sprites/units'
EXTENDED_ANN_PATH = os.path.join('output_patches', 'annotations_extended.csv')

# Load CSVs
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

# Load extended annotations for bounding boxes
if os.path.exists(EXTENDED_ANN_PATH):
    extended_df = pd.read_csv(EXTENDED_ANN_PATH)
else:
    extended_df = pd.DataFrame()

# Helper to get bounding boxes for a given unit_name and frame size
# Only matches boxes for Vampire frames with the same (Width, Height) as in the split df

def get_boxes_for_unit_and_size(df, unit_name):
    if extended_df.empty:
        return []
    # Get all Vampire frames in the split
    split_vampire = df[df['Unit_name'] == unit_name]
    boxes = []
    for _, row in split_vampire.iterrows():
        width = int(row['Width']) if 'Width' in row and not pd.isna(row['Width']) else None
        height = int(row['Height']) if 'Height' in row and not pd.isna(row['Height']) else None
        if width is None or height is None:
            continue
        # Find a matching box in extended_df with the same unit_name and frame size
        match = extended_df[(extended_df['Unit_name'] == unit_name) &
                            (extended_df['Box'].notna())]
        found = False
        for _, ext_row in match.iterrows():
            try:
                box = eval(ext_row['Box']) if isinstance(ext_row['Box'], str) else ext_row['Box']
                if int(box[2]) == width and int(box[3]) == height:
                    boxes.append(box)
                    found = True
                    break
            except:
                continue
        if not found:
            # Optionally, print a warning if no match is found
            pass
    return boxes

# Highlight Vampire frames on the original sprite image
sprite_path = os.path.join(SPRITES_DIR, 'Vampire.png')
if os.path.exists(sprite_path):
    sprite_img = Image.open(sprite_path).convert('RGBA')
    plt.figure(figsize=(12, 8))
    plt.imshow(sprite_img)
    
    def plot_boxes(df, color, label):
        ax = plt.gca()
        label_drawn = False
        boxes = get_boxes_for_unit_and_size(df, 'Vampire')
        for box in boxes:
            if box is not None:
                x, y, w, h = map(int, box)
                rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none',
                                     label=label if not label_drawn else None, zorder=10)
                ax.add_patch(rect)
                label_drawn = True
    plot_boxes(train_df, 'lime', 'Train')
    plot_boxes(val_df, 'orange', 'Validation')
    plot_boxes(test_df, 'red', 'Test')
    handles = [plt.Line2D([0], [0], color='lime', lw=4, label='Train'),
               plt.Line2D([0], [0], color='orange', lw=4, label='Validation'),
               plt.Line2D([0], [0], color='red', lw=4, label='Test')]
    plt.legend(handles=handles, loc='upper right')
    plt.title('Vampire.png: Train (lime), Validation (orange), Test (red) frame locations')
    plt.axis('off')
    plt.tight_layout()
    VISUALIZATION_OUTPUT_DIR = './visualization_output'
    os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, 'Vampire_sprite_split_highlight.png'))
    plt.show()
else:
    print('Vampire.png sprite not found in dataset directory.')
