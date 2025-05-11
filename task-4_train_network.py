import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
DATASET_DIR = './dataset'
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, 'annotations')
TRAIN_CSV = os.path.join(ANNOTATIONS_DIR, 'train.csv')
VAL_CSV = os.path.join(ANNOTATIONS_DIR, 'val.csv')
TEST_CSV = os.path.join(ANNOTATIONS_DIR, 'test.csv')
MODEL_PATH = './unit_classifier_model.keras'

# Image dimensions - updated to better match the mean frame dimensions
IMG_WIDTH, IMG_HEIGHT = 80, 96  # Rounded up from mean of 74x86
# IMG_WIDTH, IMG_HEIGHT = 74, 86  # Rounded up from mean of 74x86
BATCH_SIZE = 24
EPOCHS = 50  # Increased for more training time with early stopping

# Set seeds for reproducibility
from numpy.random import seed
from tensorflow.random import set_seed
seed_value = 1234567890
seed(seed_value)
set_seed(seed_value)

def plot_history(history):
    """Plot training history"""
    h = history.history
    epochs = range(len(h['loss']))

    plt.figure(figsize=[15, 6])
    plt.subplot(121)
    plt.plot(epochs, h['loss'], '.-', epochs, h['val_loss'], '.-')
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['Train', 'Validation'])
    
    plt.subplot(122)
    plt.plot(epochs, h['accuracy'], '.-', epochs, h['val_accuracy'], '.-')
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.savefig('./training_history.png')
    
    print('Train Acc     ', h['accuracy'][-1])
    print('Validation Acc', h['val_accuracy'][-1])

def build_model(input_shape, num_classes):
    """Build CNN model with improved architecture"""
    model = Sequential([
        # First conv block
        Conv2D(32, kernel_size=(3, 3), padding='same', activation="relu", input_shape=input_shape),
        # Conv2D(32, kernel_size=(3, 3), padding='same', activation="relu"),
        # BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second conv block
        Conv2D(64, kernel_size=(3, 3), padding='same', activation="relu"),
        # Conv2D(64, kernel_size=(3, 3), padding='same', activation="relu"),
        # BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third conv block
        Conv2D(128, kernel_size=(3, 3), padding='same', activation="relu"),
        # Conv2D(128, kernel_size=(3, 3), padding='same', activation="relu"),
        # BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Fully connected layers
        Flatten(),
        Dense(256, activation="relu"),
        # Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        loss="categorical_crossentropy", 
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    
    return model

def main():
    print("TensorFlow version:", tf.__version__)
    
    # Load data
    print("Loading data...")
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV) or not os.path.exists(TEST_CSV):
        print("Error: CSV files not found. Make sure you've generated the dataset.")
        return
    
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    # Prepare label encoding
    all_labels = pd.concat([train_df['Unit_name'], val_df['Unit_name'], test_df['Unit_name']]).unique()
    all_labels_sorted = sorted(all_labels)
    num_classes = len(all_labels_sorted)
    print(f"Found {num_classes} unique classes")

    # Create a consistent class mapping
    class_mapping = {name: idx for idx, name in enumerate(all_labels_sorted)}

    # Add class indices as strings for categorical mode
    train_df['class_idx'] = train_df['Unit_name'].map(class_mapping).astype(str)
    val_df['class_idx'] = val_df['Unit_name'].map(class_mapping).astype(str)
    test_df['class_idx'] = test_df['Unit_name'].map(class_mapping).astype(str)
    
    # Data generators with enhanced augmentation for robustness
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=DATASET_DIR,
        x_col='File_Path',
        y_col='class_idx',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=[str(i) for i in range(num_classes)],
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=DATASET_DIR,
        x_col='File_Path',
        y_col='class_idx',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=[str(i) for i in range(num_classes)],
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=DATASET_DIR,
        x_col='File_Path',
        y_col='class_idx',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=[str(i) for i in range(num_classes)],
        shuffle=False
    )
    
    # Build model
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
    model = build_model(input_shape, num_classes)
    model.summary()
    
    # Improved callbacks with learning rate reduction
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, 
            min_lr=1e-6, verbose=1
        )
    ]
    
    # Train model
    print("Training model...")
    start_time = tf.timestamp()
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_steps=val_generator.samples // BATCH_SIZE
    )
    
    training_time = tf.timestamp() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    # Plot history
    plot_history(history)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(
        test_generator,
        steps=test_generator.samples // BATCH_SIZE + 1
    )
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    
    # Save model
    print(f"\nModel saved to {MODEL_PATH}")

def predict_unit(image_path, model, label_map):
    """
    Predict unit class for a single image
    
    Args:
        image_path (str): Path to the image file
        model (keras.Model): Trained model
        label_map (dict): Dictionary mapping from index to class name
        
    Returns:
        tuple: (predicted_class, confidence)
    """
    from tensorflow.keras.preprocessing import image
    import numpy as np
    
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class and confidence
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    predicted_class = label_map[predicted_idx]
    
    return predicted_class, confidence

if __name__ == "__main__":
    main()