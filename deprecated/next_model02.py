
# imports
import os
import numpy as np
import rasterio as rio
from pathlib import Path
from typing import Tuple, Dict, Any
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Input, Cropping2D, Dropout, Conv2DTranspose, Concatenate, Flatten, Dense, Reshape, Activation, Add
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
import tensorflow as tf


def load_single_dataset(fpath: Path) -> Tuple:
    """
    Load a single dataset from the given file path.
    """

    def raster_to_chunks(raster: str, chunk_size: Tuple[int, int]) -> np.ndarray:
        """
        Load a raster file and cut it into chunks.
        """
        with rio.open(raster) as src:
            data = src.read()
            
            # Get the dimensions of the raster
            height, width = data.shape[1], data.shape[2]
            chunks = []
            # Loop through the raster in steps of chunk_size
            for i in range(0, height, chunk_size[0]):
                for j in range(0, width, chunk_size[1]):
                    # Extract the chunk
                    chunk = data[:, i:i + chunk_size[0], j:j + chunk_size[1]]
                    # Append the chunk to the list
                    chunks.append(chunk)
            return chunks
        

    x_train_strings = ["wall_aspect", "wall_height", "SkyViewFactor", "landcover", "DO_DTM", "DO_DSM"]
    y_train_strings = ["Tmrt"]
    arrays = [np.array([]), np.array([]), np.array([]), np.array([])]

    for filename in fpath.iterdir():
        # skip if not a .tif file
        if not filename.name.endswith(".tif"): continue

        with rio.open(filename) as src:
            data = src.read()
            # chunk the data
            chunks = raster_to_chunks(filename, (250, 250))
            data = np.stack(chunks)  # Convert list of chunks to a single array

            # split the data into train and test sets
            train, test = data[:int(len(data) * 0.8)], data[int(len(data) * 0.8):]

            # check if the filename contains any of the x_train_strings
            if any(s in filename.name for s in x_train_strings):
                arrays[0] = np.concatenate((arrays[0], train), axis=1) if arrays[0].size else train
                arrays[1] = np.concatenate((arrays[1], test), axis=1) if arrays[1].size else test
                
            elif any(s in filename.name for s in y_train_strings):
                arrays[2] = np.concatenate((arrays[2], train), axis=1) if arrays[2].size else train
                arrays[3] = np.concatenate((arrays[3], test), axis=1) if arrays[3].size else test

    # Rearrange arrays to have shape (chunks, 250, 250, channels)
    arrays[0] = np.transpose(arrays[0], (0, 2, 3, 1))
    arrays[1] = np.transpose(arrays[1], (0, 2, 3, 1))
    arrays[2] = np.transpose(arrays[2], (0, 2, 3, 1))
    arrays[3] = np.transpose(arrays[3], (0, 2, 3, 1))

    print(f"Loaded {len(arrays[0])} training samples and {len(arrays[1])} testing samples.")
    print(f"Loaded {len(arrays[2])} training labels and {len(arrays[3])} testing labels.")
    return arrays

def preprocess_input(img):
    return img.astype('float32') / 255.0  # Normalize to [0,1]

def postprocess_output(img):
    return img.astype('float32') * 255.0  # Scale back to [0,255]

def build_model(input_shape):
    inputs = Input(shape=input_shape, dtype=tf.float32)  # Ensure float32 input

    # Convolutional feature extraction
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    skip_x = x  # Store intermediate feature maps

    # Bottleneck - Reduced spatial dims but still conv-based
    x = Conv2D(128, (3, 3), padding='same', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Upsampling with Conv2DTranspose
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    
    # Align shapes of x and skip_x using Cropping2D
    skip_x_cropped = Cropping2D(cropping=((1, 0), (1, 0)))(skip_x)  # Adjust cropping as needed
    x = Add()([x, skip_x_cropped])  # Skip connection
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Final output layer with sigmoid (scaled correctly)
    outputs = Conv2D(1, (3, 3), padding='same', activation='sigmoid', dtype=tf.float32)(x)

    model = Model(inputs, outputs)
    # Using 'mae' (Mean Absolute Error) as the loss function for simplicity and interpretability
    
def build_model_with_mse_loss(input_shape, learning_rate=0.0005):
    """
    Build a convolutional neural network model with linear output.

    This model is similar to `build_model` but differs in the final output layer,
    which does not use an activation function (linear output). It is designed for
    tasks where the output is not constrained to a specific range, such as regression.

    Args:
        input_shape (tuple): The shape of the input data (height, width, channels).

    Returns:
        tensorflow.keras.Model: The compiled Keras model.
    """

    inputs = Input(shape=input_shape, dtype=tf.float32)  # Ensure float32 input

    # Convolutional feature extraction
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    skip_x = x  # Store intermediate feature maps

    # Bottleneck - Reduced spatial dims but still conv-based
    x = Conv2D(128, (3, 3), padding='same', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Upsampling with Conv2DTranspose
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = Add()([x, skip_x])  # Skip connection
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Final output layer without activation function (linear output)
    outputs = Conv2D(1, (3, 3), padding='same', dtype=tf.float32)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae', 'mse'])

    return model

def build_model_with_mse_loss2(input_shape, learning_rate=0.0005):
    """
    Build a convolutional neural network model with linear output.

    This model is similar to `build_model` but differs in the final output layer,
    which does not use an activation function (linear output). It is designed for
    tasks where the output is not constrained to a specific range, such as regression.

    Args:
        input_shape (tuple): The shape of the input data (height, width, channels).

    Returns:
        tensorflow.keras.Model: The compiled Keras model.
    """

    inputs = Input(shape=input_shape, dtype=tf.float32)  # Ensure float32 input

    skip_x2 = Conv2D(32, (1, 1), padding='same')(inputs)  # Adjust input channels to match

    # Convolutional feature extraction
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    skip_x = x  # Store intermediate feature maps

    # Bottleneck - Reduced spatial dims but still conv-based
    x = Conv2D(128, (3, 3), padding='same', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Upsampling with Conv2DTranspose
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = Add()([x, skip_x])  # Skip connection
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32, (3, 3), padding='same')(x)
    x = Add()([x, skip_x2])  # Skip connection
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Final output layer without activation function (linear output)
    outputs = Conv2D(1, (3, 3), padding='same', dtype=tf.float32)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae', 'mse'])

    return model

def pixel_based_flattened_model(input_shape):
    """
    Build a pixel-based regression model using Conv2D layers with the same input and output dimensions.
    """

    inputs = Input(shape=input_shape, dtype=tf.float32)  # Ensure float32 input

    # Convolutional layers
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Output layer with a single channel
    outputs = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae', 'mse'])

    return model

def custom_model(input_shape):
    x = layers.Input(shape=input_shape)
    
    # First convolutional layer
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    pool1 = layers.MaxPooling2D((2, 2), padding='same')(conv1)
    
    # Second convolutional layer
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D((2, 2), padding='same')(conv2)
    
    # Upsample to match the input dimensions
    upsampled = layers.UpSampling2D(size=(2, 2))(pool2)
    upsampled = layers.UpSampling2D(size=(2, 2))(upsampled)
    upsampled = layers.Cropping2D(cropping=((1, 1), (1, 1)))(upsampled)  # Crop to match input dimensions

    combined = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(upsampled)
    
    model = models.Model(inputs=x, outputs=combined)

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def pixel_perfect_model(input_shape):
    """
    Build a pixel-perfect model with reduced noise by adding regularization and batch normalization.
    """
    inputs = Input(shape=input_shape, dtype=tf.float32)  # Ensure float32 input

    # Flatten the input
    x = Flatten()(inputs)

    # Fully connected layers with regularization and batch normalization
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Output layer with a single channel
    outputs = Dense(input_shape[0] * input_shape[1], activation='linear')(x)
    outputs = Reshape((input_shape[0], input_shape[1], 1))(outputs)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae', 'mse'])

    return model
    
if __name__ == "__main__":
    wdir = Path("data/static/")

    action = "both"  # "train" or "test" or "both"

    # Define model
    input_shape = (250, 250, 6)  # Multi-channel input
    model = build_model_with_mse_loss(input_shape)  # or build_model(input_shape)

    # Describe the model
    model.summary()

    # Training process
    for folder_datacube in wdir.iterdir():
        if folder_datacube.is_dir():
            x_train, x_test, y_train, y_test = load_single_dataset(folder_datacube)

            # Normalize inputs and targets
            x_train = preprocess_input(x_train)
            x_test = preprocess_input(x_test)
            y_train = preprocess_input(y_train)
            y_test = preprocess_input(y_test)

            print(f"Loaded data from {folder_datacube}")
            print(f"Shapes:\n x_train: {x_train.shape}\n x_test: {x_test.shape}\n y_train: {y_train.shape}\n y_test: {y_test.shape}")

            if action == "train" or action == "both":
                print("Training model...")
                # early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

                # Train using all available chunks, accumulating mini-batches
                history = model.fit(
                    x_train, y_train, 
                    batch_size=320,  # Adjust based on memory limits
                    epochs=20,
                    validation_data=(x_test, y_test),
                    # callbacks=[early_stopping],
                    verbose=1
                )
                print("Training complete.")
                # Save the model
                model.save("models/next_model02_model.keras", overwrite=True)

            if action == "test" or action == "both":
                print("Testing model...")

                # Load the model
                model = tf.keras.models.load_model("models/next_model02_model.keras", custom_objects={'Huber': Huber()})

                # Evaluate the model
                print("Evaluating model...")
                values = model.evaluate(x_test, y_test)
                print(f"Evaluation results: {values}")

                # Make predictions
                y_pred = model.predict(x_test)
                y_pred = postprocess_output(y_pred)
                y_test = postprocess_output(y_test)
                final_output = y_pred


                # Show the first 4 predictions as images
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(4, 3, figsize=(15, 15))
                for i in range(min(5, x_test.shape[0])):
                    # Subplot 1: x_test first channel
                    im1 = ax[i, 0].imshow(x_test[i, :, :, 0], cmap='gray')  # Display the first channel
                    ax[i, 0].set_title("Input (Channel 1)")
                    ax[i, 0].axis('off')
                    fig.colorbar(im1, ax=ax[i, 0], fraction=0.046, pad=0.04)

                    # Subplot 2: True output (y_test)
                    im2 = ax[i, 1].imshow(y_test[i, :, :, 0], cmap='gray', vmin=y_test.min(), vmax=y_test.max())  # Assuming single-channel output
                    ax[i, 1].set_title("True Output")
                    ax[i, 1].axis('off')
                    fig.colorbar(im2, ax=ax[i, 1], fraction=0.046, pad=0.04)

                    # Subplot 3: Predicted output (y_pred)
                    im3 = ax[i, 2].imshow(final_output[i, :, :, 0], cmap='gray', vmin=y_test.min(), vmax=y_test.max())
                    ax[i, 2].set_title("Predicted Output")
                    ax[i, 2].axis('off')
                    fig.colorbar(im3, ax=ax[i, 2], fraction=0.046, pad=0.04)


                plt.tight_layout()
                plt.show()



    # Save the model
    model.save("models/next_model02_model.keras", overwrite=True)
    print(f"Model saved to models/next_model02_model.keras")



            
        
