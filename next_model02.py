
# imports
import os
import numpy as np
import rasterio as rio
from pathlib import Path
from typing import Tuple, Dict, Any
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Input, Cropping2D, Dropout, Conv2DTranspose, Concatenate, Flatten, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def load_single_dataset(fpath: Path) -> Tuple[np.ndarray, np.ndarray]:
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

def normalize(img):
    return img / 255.0  # Scale to [0,1]

def denormalize(img):
    return img * 255.0  # Scale back to [0,255]

def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # Shallow Convolutional Layers (Spatial Feature Extraction)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    # Flatten and Pass to Dense Layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)

    # Reshape Back to Image
    x = Dense(input_shape[0] * input_shape[1], activation='sigmoid')(x)
    outputs = Reshape((input_shape[0], input_shape[1], 1))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
    
    return model

if __name__ == "__main__":
    wdir = Path("data/static/")

    action = "train"  # "train" or "test"
    
    # Define model
    input_shape = (250, 250, 6)  # Multi-channel input
    model = build_model(input_shape)

    # Training process
    for folder_datacube in wdir.iterdir():
        if folder_datacube.is_dir():
            x_train, x_test, y_train, y_test = load_single_dataset(folder_datacube)

            # normalize the data
            x_train = normalize(x_train)
            x_test = normalize(x_test)
            y_train = normalize(y_train)
            y_test = normalize(y_test)

            print(f"Loaded data from {folder_datacube}")
            print(f"Shapes:\n x_train: {x_train.shape}\n x_test: {x_test.shape}\n y_train: {y_train.shape}\n y_test: {y_test.shape}")

            if action == "train" or action == "both":
                print("Training model...")
                # Define early stopping
                # early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

                # Train using all available chunks, accumulating mini-batches
                model.fit(x_train, y_train, 
                        batch_size=512,  # Adjust based on memory limits
                        epochs=20,
                        validation_data=(x_test, y_test),
                        # callbacks=[early_stopping],
                        verbose=1)
            
            if action == "test" or action == "both":
                print("Testing model...")
                # Load the model
                model.load_weights("models/next_model02_model.keras")

                # Evaluate the model
                loss, mae = model.evaluate(x_test, y_test)
                print(f"Test Loss: {loss}, Test MAE: {mae}")
                # Show the first 5 predictions as images
                import matplotlib.pyplot as plt

                # renormalize the data
                y_pred = model.predict(x_test)
                y_pred = denormalize(y_pred)


                fig, ax = plt.subplots(5, 3, figsize=(15, 15))
                for i in range(min(5, x_test.shape[0])):
                    # Subplot 1: x_test first channel
                    ax[i, 0].imshow(x_test[i, :, :, 0], cmap='gray')  # Display the first channel
                    ax[i, 0].set_title("Input (Channel 1)")
                    ax[i, 0].axis('off')

                    # Subplot 2: True output (y_test)
                    ax[i, 1].imshow(y_test[i, :, :, 0], cmap='gray')  # Assuming single-channel output
                    ax[i, 1].set_title("True Output")
                    ax[i, 1].axis('off')

                    # Subplot 3: Predicted output (y_pred)
                    ax[i, 2].imshow(y_pred[i, :, :, 0], cmap='gray')
                    ax[i, 2].set_title("Predicted Output")
                    ax[i, 2].axis('off')

                plt.tight_layout()
                plt.show()



    # Save the model
    model.save("models/next_model02_model.keras", overwrite=True)
    print(f"Model saved to models/next_model02_model.keras")



            
        
