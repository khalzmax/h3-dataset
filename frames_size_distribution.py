import pandas as pd
import matplotlib.pyplot as plt

# Constants
ANNOTATIONS_FILE = './output_patches/annotations_extended.csv'
# ANNOTATIONS_FILE = './output_patches_noice/annotations_noice_extended.csv'

def analyze_frame_sizes():
    # Load the annotations CSV file
    df = pd.read_csv(ANNOTATIONS_FILE)

    # Create a new DataFrame for bounding box analysis
    box_df = df.dropna(subset=['Box']).copy()  # Drop rows where 'Box' is NaN and create a copy
    box_df['Box'] = box_df['Box'].apply(lambda x: tuple(map(int, x.strip("()").split(','))))
    box_df['Width'] = box_df['Box'].apply(lambda box: box[2])
    box_df['Height'] = box_df['Box'].apply(lambda box: box[3])

    # Analyze the distribution of frame sizes
    print("Frame Size Distribution:")
    print(box_df[['Width', 'Height']].describe())

    # Identify and print values with high frequency
    width_counts = box_df['Width'].value_counts()
    height_counts = box_df['Height'].value_counts()
    high_frequency_widths = width_counts[width_counts > 1000]
    high_frequency_heights = height_counts[height_counts > 1000]

    print("\nValues with high frequency (Width):")
    print(high_frequency_widths)
    print("\nValues with high frequency (Height):")
    print(high_frequency_heights)

    # Filter out values with frequency over 1000
    valid_widths = width_counts[width_counts <= 1000].index
    valid_heights = height_counts[height_counts <= 1000].index
    filtered_box_df = box_df[box_df['Width'].isin(valid_widths) & box_df['Height'].isin(valid_heights)]

    # Plot the distribution of widths and heights
    plt.figure(figsize=(12, 6))
    bins = range(0, 201, 5)  # Thinner bars with bin width of 5, max x-axis value of 200
    plt.hist(filtered_box_df['Width'], bins=bins, alpha=0.5, label='Width', color='blue')
    plt.hist(filtered_box_df['Height'], bins=bins, alpha=0.5, label='Height', color='orange')
    plt.title('Distribution of Frame Sizes (Filtered)')
    plt.xlabel('Size (pixels)')
    plt.ylabel('Frequency')
    plt.xlim(0, 200)  # Set x-axis limit to 200
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    analyze_frame_sizes()