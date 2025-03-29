import tensorflow as tf
import numpy as np
from pathlib import Path
import rasterio as rio

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# # Example: Create dummy data. Replace these with your actual data.
# num_samples = 10  # for example
# x_train = np.random.rand(num_samples, 250, 250, 6).astype(np.float32)
# y_train = np.random.rand(num_samples, 250, 250, 1).astype(np.float32)

def cut_raster_into_chunks(raster, chunk_size):
    """
    Cut a raster into smaller chunks.
    Args:
        raster: The input raster to be cut.
        chunk_size: The size of each chunk.
    Returns:
        A list of chunks.
    """
    chunks = []
    height, width = raster.shape[1], raster.shape[2]  # Assuming raster is in (bands, height, width) format
    for i in range(0, height, chunk_size):
        for j in range(0, width, chunk_size):
            chunk = raster[:, i:i + chunk_size, j:j + chunk_size]  # Include all bands (if 3D raster)
            if chunk.shape[1] == chunk_size and chunk.shape[2] == chunk_size:
                chunks.append(chunk)
            else:
                print(f"Skipping chunk with size {chunk.shape} as it does not match expected size ({chunk_size}, {chunk_size}).")
    return chunks

def chunk_datacube(cut_raster_into_chunks):
    wdir = Path("data/static")
    for datacube in wdir.iterdir():
    # load all tifs that contain the following strings:
        x_train_strs = ["wall_aspect", "wall_height", "SkyViewFactor", "landcover", "DO_DTM", "DO_DSM"]
        y_train_strs = ["Tmrt"]
    
        rasterchunks = {"x_train": {}, "y_train": {}}
    
        for raster in datacube.iterdir():
            if not raster.name.endswith(".tif"): continue
            if any(s in raster.name for s in x_train_strs):
            # Load the raster and cut it into chunks
                with rio.open(raster) as src:
                    data = src.read()
                    chunks = cut_raster_into_chunks(data, 250)  # Assuming chunk size of 250x250
                    rasterchunks["x_train"][raster.name] = chunks
            elif any(s in raster.name for s in y_train_strs):
            # Load the raster and cut it into chunks
                with rio.open(raster) as src:
                    data = src.read()
                    chunks = cut_raster_into_chunks(data, 250)
                    rasterchunks["y_train"][raster.name] = chunks
    return rasterchunks


# Example model creation function
def create_compiled_conv2d_model(param_dict: dict):
    """
    Creates and compiles a convolutional model for channels-last data.
    Expected input shape: (250, 250, 6) and output shape: (250, 250, 1).
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, ZeroPadding2D

    # Unpack parameters
    model_compiler_metadata = param_dict["model_compiler_metadata"]
    filters = model_compiler_metadata["filters"]
    kernel_size = model_compiler_metadata["kernel_size"]
    activation = model_compiler_metadata["activation"]
    final_activation = model_compiler_metadata["final_activation"]
    pool_size = model_compiler_metadata["pool_size"]
    use_batchnorm = model_compiler_metadata["use_batchnorm"]
    
    # Input shape for channels-last ordering
    inputs = Input(shape=model_compiler_metadata["input_shape"])  # (250, 250, 6)
    
    # Encoder: Down-sampling layers
    x = inputs
    for f in filters[:-1]:
        x = Conv2D(f, kernel_size, padding="same")(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = MaxPooling2D(pool_size)(x)
    
    # Bottleneck layer
    x = Conv2D(filters[-1], kernel_size, padding="same")(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    # Decoder: Up-sampling layers
    for f in reversed(filters[:-1]):
        x = UpSampling2D(pool_size)(x)
        x = Conv2D(f, kernel_size, padding="same")(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
    
    # Final output layer: produces output shape (250, 250, 1)
    x = Conv2D(1, (1, 1), activation=final_activation, padding='same')(x)  # Ensure the final layer matches the size
    
    # To make sure the output is 250x250, use ZeroPadding2D if needed
    output = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)  # Adjust padding to match dimensions
    
    model = Model(inputs=inputs, outputs=output)
    
    model.compile(
        optimizer=model_compiler_metadata["optimizer"],
        loss=model_compiler_metadata["loss"],
        metrics=model_compiler_metadata["metrics"]
    )
    return model

# Aggregate all x_train and y_train chunks into single datasets
# Combine the 6 keys under x_train into channels for each chunk
def create_dataset(rasterchunks, batch_size):
    x_chunks = []
    for i in range(len(next(iter(rasterchunks["x_train"].values())))):  # Iterate over the number of chunks
        combined_chunk = []
        for key in rasterchunks["x_train"]:
            combined_chunk.append(rasterchunks["x_train"][key][i])  # Collect corresponding chunks across keys
        combined_chunk = np.concatenate(combined_chunk, axis=0)  # Concatenate along the channel dimension
        x_chunks.append(combined_chunk)

    # y chunks are already in the correct format and shape (single channel)
    y_chunks = []
    for key in rasterchunks["y_train"]:
        y_chunks.extend(rasterchunks["y_train"][key])  # Extend the list with all chunks from y_train

    # Ensure the number of x_chunks matches y_chunks
    min_chunks = min(len(x_chunks), len(y_chunks))
    x_chunks = x_chunks[:min_chunks]
    y_chunks = y_chunks[:min_chunks]

    print(f"Training on {min_chunks} paired chunks.")

    # Convert to numpy arrays
    x_chunks = np.array(x_chunks)
    y_chunks = np.array(y_chunks)

    # scale the data to [0, 1] range
    x_chunks = x_chunks.astype(np.float32) / 255.0
    y_chunks = y_chunks.astype(np.float32) / 255.0

    # Restructure to channel-last format by moxing the second axis to the last
    x_chunks = np.moveaxis(x_chunks, 1, -1)  # Move the second axis (height) to the last position
    y_chunks = np.moveaxis(y_chunks, 1, -1)  # Move the second axis (height) to the last position

    print(f"x_train dataset shape: {x_chunks.shape}")
    print(f"y_train dataset shape: {y_chunks.shape}")

    x_train, x_test = x_chunks[:int(0.8 * len(x_chunks))], x_chunks[int(0.8 * len(x_chunks)):]
    y_train, y_test = y_chunks[:int(0.8 * len(y_chunks))], y_chunks[int(0.8 * len(y_chunks)):]

    # Ensure x_train and y_train are numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Check if shapes are compatible
    assert x_train.shape[0] == y_train.shape[0], "x_train and y_train must have the same number of samples."

    # Create a tf.data dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    # Create a validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    print(f"Validation dataset shape: {val_dataset.element_spec}")
    # Print the dataset shape
    print(f"Train dataset shape: {train_dataset.element_spec}")
    print(f"Validation dataset shape: {val_dataset.element_spec}")


    return train_dataset, val_dataset


# Define your model parameters in a dictionary
param_dict = {
    "model_compiler_metadata": {
         "input_shape": (250, 250, 6),  # channels-last ordering
         "filters": [32, 64, 128],
         "kernel_size": (3, 3),
         "activation": "relu",
         "final_activation": "sigmoid",  # or another appropriate activation
         "pool_size": (2, 2),
         "use_batchnorm": True,
         "optimizer": "adam",
         "loss": "binary_crossentropy",  # adjust loss according to your problem
         "metrics": ["accuracy"]
    },
    "model_run_metadata": {
        "batch_size": 128,
        "epochs": 20,
        "steps_per_epoch": 2,          # adjust according to your dataset
        "validation_steps": 2,          # adjust according to your dataset
    }
}

if __name__ == "__main__":

    dataset_dir = Path("data/static")

    # Create and compile the model
    model = create_compiled_conv2d_model(param_dict)
    model.summary()


    for folder in dataset_dir.iterdir():
        if not folder.is_dir(): continue
        print(f"Processing folder: {folder.name}")

        # Load the raster data and cut it into chunks
        rasterchunks = 

        # Create the dataset
        train_dataset, val_dataset = create_dataset(rasterchunks, param_dict["model_run_metadata"]["batch_size"])

        # Print the dataset shape
        print(f"Train dataset shape: {train_dataset.element_spec}")

        # Train the model on the aggregated dataset
        with tf.device('/GPU:0'):
            model.fit(train_dataset, epochs=param_dict["model_run_metadata"]["epochs"],
                      steps_per_epoch=param_dict["model_run_metadata"]["steps_per_epoch"],
                      validation_steps=param_dict["model_run_metadata"]["validation_steps"])
    
        # Save the model
        model.save(f"model_stupid.keras", save_format='keras')
        print(f"Model saved to {folder}/model_stupid.keras")
        # Clear the model from memory
        del model
        tf.keras.backend.clear_session()
        print("Model cleared from memory.")
        # Clear the dataset from memory
        del train_dataset
        print("Dataset cleared from memory.")
        # Clear the raster chunks from memory

        # Clear the GPU memory
        tf.keras.backend.clear_session()
        print("GPU memory cleared.")

        # Load the model
        model = tf.keras.models.load_model(f"model_stupid.keras")
        print(f"Model loaded from {folder}/model_stupid.keras")

        # Evaluate the model on the dataset
        # Note: You might want to create a separate validation dataset for this
        model.evaluate(val_dataset, steps=param_dict["model_run_metadata"]["validation_steps"])

        # Predict a raster using x_test
        model_predictions = model.predict(val_dataset, steps=param_dict["model_run_metadata"]["validation_steps"])
        print(f"Model predictions shape: {model_predictions.shape}")
        
        # Plot the predictions´with two subplots; one being the y_test and the other being the model predictions
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        y_test = val_dataset.take(1)
        y_test = np.array(list(y_test.as_numpy_iterator()))
        y_test = y_test[0][1]
        y_test = np.squeeze(y_test)
        model_predictions = np.squeeze(model_predictions)
        print(f"y_test shape: {y_test.shape}")
        print(f"Model predictions shape: {model_predictions.shape}")

        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(y_test[0], cmap=cm.jet)
        ax[0].set_title("y_test")
        ax[1].imshow(model_predictions[0], cmap=cm.jet)
        ax[1].set_title("Model Predictions")
        plt.show()
