import os
import cv2
import pandas as pd
from pathlib import Path
import re
import tqdm as tqdm
import numpy as np

# Constants
SPRITES_DIR = '../datasets/heroes3_sprites/units'
CSV_FILE = '../datasets/H3Units.csv'
OUTPUT_DIR = './output_patches'
OUTPUT_CSV_FILE = f"{OUTPUT_DIR}/annotations_extended.csv"

# Define sizes Frames to skip skip (e.g., unit avatars)
# Added both small and large avatar sizes with tolerance
unit_avatar_sizes = [(58, 64), (30, 30)]  # Large and small avatar sizes
avatar_size_tolerance = 5  # Allow ±3px deviation in size

# Utility: Normalize name
def camel_to_hyphen(name):
    return re.sub(r'(?<!^)(?=[A-Z])', '-', name).title()

# Get path to sprite
def get_sprite_path(unit_name):
    return f"{SPRITES_DIR}/{camel_to_hyphen(unit_name)}.png"

# Check if a size matches an avatar size with tolerance
def is_avatar_size(w, h):
    for avatar_w, avatar_h in unit_avatar_sizes:
        if (abs(w - avatar_w) <= avatar_size_tolerance and 
            abs(h - avatar_h) <= avatar_size_tolerance):
            return True
    return False

def filter_by_histogram_similarity(image, boxes, threshold=0.5):
    if not boxes:
        return []
        
    # Compute histograms for all patches
    histograms = []
    patches = []
    for (x, y, w, h) in boxes:
        patch = image[y:y+h, x:x+w]
        # Ensure we use RGB channels for histogram comparison
        if patch.shape[2] == 4:  # RGBA
            patch_rgb = patch[:,:,:3]
        else:
            patch_rgb = patch
            
        patches.append(patch)
        hist = cv2.calcHist([patch_rgb], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
    
    if not histograms:
        return []
        
    histograms = np.array(histograms)

    # Compute the average histogram
    avg_hist = np.mean(histograms, axis=0)

    # Filter out patches whose histogram is too different from the average
    filtered_boxes = []
    for i, hist in enumerate(histograms):
        w, h = boxes[i][2], boxes[i][3]
        
        # Skip frames matching the defined avatar sizes with tolerance
        if is_avatar_size(w, h):
            filtered_boxes.append(boxes[i])
            continue
        
        # Use correlation or Bhattacharyya distance
        dist = cv2.compareHist(hist.astype('float32'), avg_hist.astype('float32'), cv2.HISTCMP_BHATTACHARYYA)
        if dist < threshold:
            filtered_boxes.append(boxes[i])
        # else:
        #     print(f"Filtered out patch {i} with histogram distance: {dist:.4f}")
    return filtered_boxes

# Detect valid bounding boxes
def extract_valid_bounding_boxes(image, signature_template=None):
    if image.shape[2] == 4:  # RGBA
        alpha = image[:, :, 3]
        mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)[1]
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out small boxes
        if w > 25 and h > 25:
            all_boxes.append((x, y, w, h))

    # If signature_template is provided, filter out signature by proportion and template matching
    if signature_template is not None:
        temp_h, temp_w = signature_template.shape[:2]
        temp_ratio = temp_w / temp_h
        threshold = 0.85
        filtered_boxes = []
        for (x, y, w, h) in all_boxes:
            box_ratio = w / h
            # Check if aspect ratios are close (allow some tolerance)
            if abs(box_ratio - temp_ratio) < 0.2:
                # Resize the candidate region to template size
                candidate = image[y:y+h, x:x+w]
                candidate_resized = cv2.resize(candidate, (temp_w, temp_h), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(candidate_resized, signature_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val >= threshold:
                    # This is the signature, skip it
                    continue
            filtered_boxes.append((x, y, w, h))
        return filtered_boxes
    return all_boxes

def matches_small_avatar(w, h):
    avatar_w, avatar_h = unit_avatar_sizes[1]  # Small avatar size (30x30)
    return (abs(w - avatar_w) <= avatar_size_tolerance and 
            abs(h - avatar_h) <= avatar_size_tolerance)

def matches_large_avatar(w, h):
    avatar_w, avatar_h = unit_avatar_sizes[0]  # Large avatar size (58x64)
    return (w == avatar_w and h == avatar_h)

def handle_avatar_matching(image, boxes):
    """
    Match small avatars with large avatars in the image.
    
    Args:
        image: The source image containing avatars
        boxes: List of bounding boxes (x, y, w, h)
        
    Returns:
        small_matched: Set of indices of small avatars that match large ones
        large_matched: Set of indices of large avatars that have matching small ones
    """
    # Separate small avatars and large avatars for matching
    small_avatars = []
    large_avatars = []
    
    for i, (x, y, w, h) in enumerate(boxes):
        if matches_small_avatar(w, h):
            small_avatars.append((i, x, y, w, h))
        elif matches_large_avatar(w, h):
            large_avatars.append((i, x, y, w, h))
    
    # Try to match small avatars with large ones
    small_matched = set()  # Track which small avatars have a match
    large_matched = set()  # Track which large avatars have a match
    
    # Match threshold
    match_threshold = 0.9
    
    for small_idx, small_x, small_y, small_w, small_h in small_avatars:
        small_patch = image[small_y:small_y+small_h, small_x:small_x+small_w]
        # Use RGB channels for matching
        if small_patch.shape[2] == 4:
            small_patch_rgb = small_patch[:,:,:3]
        else:
            small_patch_rgb = small_patch
        
        best_match = None
        best_match_val = 0
        
        for large_idx, large_x, large_y, large_w, large_h in large_avatars:
            large_patch = image[large_y:large_y+large_h, large_x:large_x+large_w]
            if large_patch.shape[2] == 4:
                large_patch_rgb = large_patch[:,:,:3]
            else:
                large_patch_rgb = large_patch
                
            # Resize small avatar to same size as large for comparison
            small_resized = cv2.resize(small_patch_rgb, (large_patch_rgb.shape[1], large_patch_rgb.shape[0]))
            
            try:
                # Match small avatar in large avatar
                result = cv2.matchTemplate(large_patch_rgb, small_resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_match_val:
                    best_match_val = max_val
                    best_match = large_idx
            except:
                continue
        
        # If good match found
        if best_match_val > match_threshold:
            small_matched.add(small_idx)
            large_matched.add(best_match)
    
    return small_matched, large_matched

# Crop and save image patches
def crop_and_save(unit_row, signature_template=None):
    unit_name = unit_row['Unit_name']
    sprite_path = get_sprite_path(unit_name)
    if not os.path.exists(sprite_path):
        return []

    image = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Unable to load image - {sprite_path}")
        return []
    
    # Extract bounding boxes and optionally remove the author sign
    boxes = extract_valid_bounding_boxes(image, signature_template=signature_template)
    
    # Filter out patches by color histogram similarity
    boxes = filter_by_histogram_similarity(image, boxes, threshold=0.5)
    
    # Match small and large avatars
    small_matched, large_matched = handle_avatar_matching(image, boxes)
    
    saved_paths = []
    for i, (x, y, w, h) in enumerate(boxes):
        patch = image[y:y+h, x:x+w]
        out_dir = Path(OUTPUT_DIR) / unit_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{Path(sprite_path).stem}_frame_{i}.png"
        cv2.imwrite(str(out_path), patch)
        
        # Set frame_type based on size and matching results
        if matches_large_avatar(w, h):
            if i in large_matched:
                frame_type = 'unit_avatar'  # Confirmed large avatar
            else:
                frame_type = 'unit_avatar'  # Still assign as avatar based on size
        elif matches_small_avatar(w, h):
            if i in small_matched:
                frame_type = 'unit_avatar'  # Confirmed small avatar
            else:
                frame_type = 'unit_avatar'  # Small avatars also get the unit_avatar type
        else:
            # Other frames are considered unit frames
            frame_type = 'unit'
            
        saved_paths.append({
            'Unit_name': unit_name,
            'Frame_ID': int(i),
            'File_Path': str(out_path),
            'Box': (x, y, w, h),
            'frame_type': frame_type
        })
    return saved_paths

# Main function
def main():
    # Load the CSV file
    df = pd.read_csv(CSV_FILE, delimiter=',')
    
    # Load the signature template
    signature_template = cv2.imread('signature_avatar.png', cv2.IMREAD_UNCHANGED)
    if signature_template is None:
        raise FileNotFoundError("The signature template 'signature_avatar.png' could not be loaded.")

    all_saved_paths = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing units"):
        saved_paths = crop_and_save(row, signature_template=signature_template)
        all_saved_paths.extend(saved_paths)
    
    # Create a DataFrame for the extended data
    extended_df = pd.DataFrame(all_saved_paths)

    # Merge the original data with the extended data
    combined_df = pd.merge(df, extended_df, on='Unit_name', how='left')

    # Save the combined data to a new CSV file
    combined_df.to_csv(f"{OUTPUT_CSV_FILE}", index=False)

if __name__ == "__main__":
    main()