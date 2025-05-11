"""
Script for unit classification inference using the trained model.
This script can be used for:
1. Inferring units from individual frame images
2. Processing a video file frame by frame for unit detection
"""
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

# --- Configuration ---
MODEL_PATH = './unit_classifier_model.keras'
DATASET_DIR = './dataset'
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, 'annotations')
TEST_CSV = os.path.join(ANNOTATIONS_DIR, 'test.csv')
VISUALIZATION_OUTPUT_DIR = 'visualization_output'

# Image dimensions - should match the training dimensions
IMG_WIDTH, IMG_HEIGHT = 80, 96

def load_trained_model():
    """Load the trained model from disk"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None
    
    try:
        model = load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def get_class_mapping():
    """Create mapping from indices to class names using test.csv"""
    if not os.path.exists(TEST_CSV):
        print(f"Error: Test CSV file not found at {TEST_CSV}")
        return None
    
    try:
        test_df = pd.read_csv(TEST_CSV)
        # Get unique class names
        class_names = sorted(test_df['Unit_name'].unique())
        # Create mapping from index to class name
        label_map = {i: name for i, name in enumerate(class_names)}
        return label_map
    except Exception as e:
        print(f"Error creating class mapping: {str(e)}")
        return None

def evaluate_test_set(model, label_map):
    """
    Evaluate the model on the test set
    
    Args:
        model: Trained model
        label_map: Dictionary mapping from index to class name
        
    Returns:
        dict: Results with accuracy and per-class metrics
    """
    try:
        test_df = pd.read_csv(TEST_CSV)
        correct = 0
        total = len(test_df)
        class_correct = {}
        class_total = {}
        
        for _, row in test_df.iterrows():
            image_path = os.path.join(DATASET_DIR, row['File_Path'])
            true_class = row['Unit_name']
            
            # Initialize class counters if needed
            if true_class not in class_correct:
                class_correct[true_class] = 0
                class_total[true_class] = 0
            
            # Increment total count for this class
            class_total[true_class] += 1
            
            # Get prediction
            pred_class, _ = predict_image(image_path, model, label_map)
            
            # Check if correct
            if pred_class == true_class:
                correct += 1
                class_correct[true_class] += 1
        
        # Calculate overall accuracy
        accuracy = correct / total if total > 0 else 0
        
        # Calculate per-class accuracy
        class_accuracy = {}
        for cls in class_total:
            class_accuracy[cls] = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        
        return {
            'overall_accuracy': accuracy,
            'per_class_accuracy': class_accuracy
        }
    
    except Exception as e:
        print(f"Error evaluating test set: {str(e)}")
        return None

def main():
    """Main function for inference"""
    # Load the model
    model = load_trained_model()
    if model is None:
        return
    
    # Get class mapping
    label_map = get_class_mapping()
    if label_map is None:
        return
    
    
    results = evaluate_test_set(model, label_map)
    
    if results:
        print(f"\nTest set evaluation results:")
        print(f"Overall accuracy: {results['overall_accuracy']:.4f}")
        
        print("\nPer-class accuracy:")
        sorted_classes = sorted(results['per_class_accuracy'].items(), 
                                key=lambda x: x[1], reverse=True)
        
        for cls, acc in tqdm(sorted_classes, desc="Processing classes"):
            print(f"  {cls}: {acc:.4f}")
        # --- Visualization ---
        class_names = [cls for cls, _ in sorted_classes]
        accuracies = [acc for _, acc in sorted_classes]
        plt.figure(figsize=(12, 5))
        plt.bar(class_names, accuracies)
        plt.ylabel('Accuracy')
        plt.xlabel('Class')
        plt.title('Per-Class Accuracy on Test Set')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(VISUALIZATION_OUTPUT_DIR, 'evaluation_per_class_accuracy.png'))
        plt.show()

if __name__ == "__main__":
    main()
