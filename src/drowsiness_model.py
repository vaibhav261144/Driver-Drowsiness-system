import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_eye_model(input_shape=(24, 24, 1)):
    """
    Creates a CNN model for eye state classification (open vs closed)
    """
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification (open=1, closed=0)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_eye_model(train_data_dir, validation_data_dir, model_save_path, batch_size=32, epochs=20):
    """
    Train the eye state classification model
    
    Parameters:
    - train_data_dir: Directory containing 'open' and 'closed' subdirectories with eye images
    - validation_data_dir: Directory containing validation data with same structure
    - model_save_path: Path to save the trained model
    - batch_size: Batch size for training
    - epochs: Number of epochs to train
    """
    # Create model
    model = create_eye_model()
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    # Only rescale for validation
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(24, 24),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(24, 24),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary'
    )
    
    # Train model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    
    # Save model
    model.save(model_save_path)
    
    return model

def eye_aspect_ratio(eye_landmarks):
    """
    Calculate the eye aspect ratio (EAR) from eye landmarks
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    # Compute the euclidean distances between the vertical eye landmarks
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear

# Default EAR threshold values (can be calibrated for individual users)
EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES_THRESHOLD = 15  # Number of consecutive frames with closed eyes to trigger alert

if __name__ == "__main__":
    # This could be used to train the model given the right data
    pass 