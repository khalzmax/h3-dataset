import pandas as pd
import cv2
import os
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
import ast  # For safely evaluating string representations of tuples
import argparse
import tqdm as tqdm

# Configuration
ANNOTATIONS_CSV = './output_patches/annotations_extended.csv'
SPRITES_DIR = '../datasets/heroes3_sprites/units'
OUTPUT_DIR = './visualization_output/units'

# Utility: Normalize name (same as in your existing code)
def camel_to_hyphen(name):
    import re
    return re.sub(r'(?<!^)(?=[A-Z])', '-', name).title()

def get_sprite_path(unit_name):
    """Get path to sprite based on unit name"""
    return f"{SPRITES_DIR}/{camel_to_hyphen(unit_name)}.png"

def draw_bounding_boxes(image, boxes, colors=None):
    """Draw bounding boxes on the image with optional frame type labels"""
    img_with_boxes = image.copy()
    
    if not colors:
        # Default colors for different frame types
        colors = {
            'unit_avatar': (0, 255, 0),    # Green
            'unit': (255, 0, 0),           # Blue
            'noice': (0, 0, 255),          # Red
            'default': (255, 255, 0)       # Yellow
        }
    
    for i, box_data in enumerate(boxes):
        x, y, w, h = box_data['box']
        frame_type = box_data.get('frame_type', 'default')
        
        # Get color based on frame type
        color = colors.get(frame_type, colors['default'])
        
        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), color, 2)
        
        # Add text with frame number and type
        label = f"#{i} {frame_type}"
        cv2.putText(img_with_boxes, label, (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_with_boxes

def visualize_all_sprites(save_dir=None, show_plots=False):
    """
    Visualize all sprites with their bounding boxes.
    
    Args:
        save_dir: Directory to save visualizations (default: OUTPUT_DIR)
        show_plots: Whether to display plots interactively (default: False)
    """
    # Create output directory if it doesn't exist
    if save_dir is None:
        save_dir = OUTPUT_DIR
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"Loading annotations from {ANNOTATIONS_CSV}")
    df = pd.read_csv(ANNOTATIONS_CSV)
    
    # Get unique unit names
    unique_units = df['Unit_name'].unique()
    
    if len(unique_units) == 0:
        print("No units found in CSV!")
        return
    
    print(f"Found {len(unique_units)} unique units. Processing...")
    
    # Process each unit
    for i, unit_name in enumerate(tqdm.tqdm(unique_units, desc="Processing units")):
        print(f"Processing {i+1}/{len(unique_units)}: {unit_name}")
        
        # Get sprite path for the unit
        sprite_path = get_sprite_path(unit_name)
        
        if not os.path.exists(sprite_path):
            print(f"  - Error: Sprite not found at {sprite_path}")
            continue
        
        # Load the sprite image
        image = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
        
        if image is None:
            print(f"  - Error: Could not load image from {sprite_path}")
            continue
        
        # Convert RGBA to RGB if needed
        if image.shape[2] == 4:
            # Create a white background
            background = np.ones_like(image[:,:,:3]) * 255
            # Alpha blending
            alpha = image[:,:,3:] / 255.0
            image_rgb = (image[:,:,:3] * alpha + background * (1 - alpha)).astype(np.uint8)
        else:
            image_rgb = image
        
        # Get the boxes for this unit from the CSV
        unit_df = df[df['Unit_name'] == unit_name].copy()
        
        boxes = []
        for _, row in unit_df.iterrows():
            # Extract bounding box - handle both string representation and tuple
            if isinstance(row['Box'], str):
                try:
                    # Convert string representation of tuple to actual tuple
                    box = ast.literal_eval(row['Box'])
                except:
                    print(f"  - Error parsing box: {row['Box']}")
                    continue
            else:
                box = row['Box']
            
            # Extract frame type if available
            frame_type = row.get('frame_type', 'default')
            
            boxes.append({
                'box': box,
                'frame_type': frame_type,
                'id': row.get('Frame_ID', len(boxes))
            })
        
        if not boxes:
            print(f"  - No valid bounding boxes found for {unit_name}")
            continue
            
        # Draw the boxes on the image
        image_with_boxes = draw_bounding_boxes(image_rgb, boxes)
        
        # Create a figure for this sprite
        plt.figure(figsize=(12, 10))
        plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        plt.title(f"Unit: {unit_name} - {len(boxes)} bounding boxes")
        plt.axis('off')
        
        # Save the visualization
        output_path = os.path.join(save_dir, f"{unit_name}_boxes.png")
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  - Visualization saved to {output_path}")
        
        # Show the plot if requested
        if show_plots:
            plt.show()
        else:
            plt.close()  # Close the figure to free memory

def visualize_random_sprite():
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"Loading annotations from {ANNOTATIONS_CSV}")
    df = pd.read_csv(ANNOTATIONS_CSV)
    
    # Get unique unit names
    unique_units = df['Unit_name'].unique()
    
    if len(unique_units) == 0:
        print("No units found in CSV!")
        return
    
    # Select a random unit
    random_unit = random.choice(unique_units)
    print(f"Selected random unit: {random_unit}")
    
    # Get sprite path for the random unit
    sprite_path = get_sprite_path(random_unit)
    
    if not os.path.exists(sprite_path):
        print(f"Error: Sprite not found at {sprite_path}")
        return
    
    # Load the sprite image
    image = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        print(f"Error: Could not load image from {sprite_path}")
        return
    
    # Convert RGBA to RGB if needed
    if image.shape[2] == 4:
        # Create a white background
        background = np.ones_like(image[:,:,:3]) * 255
        # Alpha blending
        alpha = image[:,:,3:] / 255.0
        image_rgb = (image[:,:,:3] * alpha + background * (1 - alpha)).astype(np.uint8)
    else:
        image_rgb = image
    
    # Get the boxes for this unit from the CSV
    unit_df = df[df['Unit_name'] == random_unit].copy()
    
    boxes = []
    for _, row in unit_df.iterrows():
        # Extract bounding box - handle both string representation and tuple
        if isinstance(row['Box'], str):
            try:
                # Convert string representation of tuple to actual tuple
                box = ast.literal_eval(row['Box'])
            except:
                print(f"Error parsing box: {row['Box']}")
                continue
        else:
            box = row['Box']
        
        # Extract frame type if available
        frame_type = row.get('frame_type', 'default')
        
        boxes.append({
            'box': box,
            'frame_type': frame_type,
            'id': row.get('Frame_ID', len(boxes))
        })
    
    # Draw the boxes on the image
    image_with_boxes = draw_bounding_boxes(image_rgb, boxes)
    
    # Create a larger figure for better visibility
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title(f"Unit: {random_unit} - {len(boxes)} bounding boxes")
    plt.axis('off')
    
    # Save the visualization
    output_path = f"{OUTPUT_DIR}/{random_unit}_boxes.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Also show the result
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize sprite bounding boxes')
    parser.add_argument('--all', action='store_true', help='Visualize all sprites instead of a random one')
    parser.add_argument('--show', action='store_true', help='Show plots (only applicable with --all)')
    args = parser.parse_args()
    
    if args.all:
        visualize_all_sprites(show_plots=args.show)
    else:
        visualize_random_sprite()