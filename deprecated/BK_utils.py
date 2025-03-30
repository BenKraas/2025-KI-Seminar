
def create_compiled_conv2d_model(param_dict: dict):
    """
    Function designed with ChatGPT guidance
    Creates a configurable convolutional model.

    Parameters:
        param_dict (dict): Dictionary containing model parameters.
            - model_compiler_metadata (dict): Metadata for model compilation.
                - filters (list): List of filter sizes for each layer.
                - kernel_size (tuple): Size of the convolutional kernel.
                - activation (str): Activation function to use.
                - final_activation (str): Activation function for the output layer.
                - pool_size (tuple): Size of the pooling window.
                - use_batchnorm (bool): Whether to use batch normalization.
                - optimizer (str): Optimizer to use for compilation.
                - loss (str): Loss function to use for compilation.
                - metrics (list): List of metrics to use for compilation.
        
    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D
    import tensorflow as tf

    # Unpack parameters
    model_compiler_metadata = param_dict["model_compiler_metadata"]
    print(f"Model Compiler Metadata: {model_compiler_metadata}")
    filters, kernel_size, activation, final_activation, pool_size, use_batchnorm = (
        model_compiler_metadata["filters"],
        model_compiler_metadata["kernel_size"],
        model_compiler_metadata["activation"],
        model_compiler_metadata["final_activation"],
        model_compiler_metadata["pool_size"],
        model_compiler_metadata["use_batchnorm"]
    )

    # Create the model
    inputs = Input(shape=model_compiler_metadata["input_shape"])
    
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
    
    # Final output layer
    output = Conv2D(1, (1, 1), activation=final_activation)(x)
    
    model = Model(inputs=inputs, outputs=output)


    # compile the model
    model.compile(
        optimizer=model_compiler_metadata["optimizer"],
        loss=model_compiler_metadata["loss"],
        metrics=model_compiler_metadata["metrics"]
    )
    return model


def train_test_raster_split(data_dict, test_size=0.2, random_state=42):
    """
    Split the raster data into train and test sets and update the data_dict.
    
    Parameters:
        data_dict (dict): Dictionary containing raster data.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        data_dict (dict): Updated dictionary with train and test sets.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    for parent_key, sub_dict in data_dict.items():
        for file_key, file_data in sub_dict.items():
            # get the data and meta
            data = file_data["data"]
            # get the shape of the data
            shape = data.shape
            # flatten the data
            flat_data = data.reshape(shape[0], -1).T
            # create a DataFrame
            df = pd.DataFrame(flat_data)

            # split the data into train and test sets
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

            # update the data_dict with train and test sets
            file_data["train"] = train_df
            file_data["test"] = test_df

    return data_dict


def to_x_y_train_test(data_dict):
    """
    Convert the data_dict to X and y for training and testing.
    y_train is the raster with the name "Tmrt"
    Everything else is X_train.
    
    Preserves exact dimensions without padding and reshapes data
    to be compatible with Conv2D while maintaining original structure.
    
    Args:
        data_dict: Nested dictionary containing raster data
        
    Returns:
        x_train, y_train, x_test, y_test: Arrays prepared for Conv2D model training
    """
    import numpy as np
    x_train_list, y_train_list, x_test_list, y_test_list = [], [], [], []
    
    # Determine original grid size from data
    original_height = None
    original_width = None
    
    # Try to extract original dimensions from data dictionary
    for parent_key, sub_dict in data_dict.items():
        for file_key, file_data in sub_dict.items():
            if "data" in file_data and hasattr(file_data["data"], "shape"):
                data_shape = file_data["data"].shape
                print(f"Found data shape in {parent_key} - {file_key}: {data_shape}")
                if len(data_shape) >= 2:
                    original_height, original_width = data_shape[:2]
                    print(f"Using original dimensions: {original_height}x{original_width}")
                    break
        if original_height is not None:
            break
    
    if original_height is None or original_width is None:
        print("Warning: Could not determine original grid dimensions from data.")
        print("Will use the square root of sample count as fallback.")
    
    for parent_key, sub_dict in data_dict.items():
        for file_key, file_data in sub_dict.items():
            print(f"Processing {parent_key} - {file_key}")
            
            # If the file_key contains "Tmrt", it's the target variable
            if "Tmrt" in file_key:
                y_train_list.append(file_data["train"].values)
                y_test_list.append(file_data["test"].values)
            else:
                x_train_list.append(file_data["train"].values)
                x_test_list.append(file_data["test"].values)
    
    # Verify we have target data
    if not y_train_list:
        raise ValueError("y_train is empty. Check the data_dict for the 'Tmrt' key.")
    
    # Stack the data along the features axis
    x_train = np.concatenate(x_train_list, axis=-1) if len(x_train_list) > 1 else x_train_list[0]
    y_train = np.concatenate(y_train_list, axis=-1) if len(y_train_list) > 1 else y_train_list[0]
    x_test = np.concatenate(x_test_list, axis=-1) if len(x_test_list) > 1 else x_test_list[0]
    y_test = np.concatenate(y_test_list, axis=-1) if len(y_test_list) > 1 else y_test_list[0]
    
    print(f"Initial shapes:")
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # If we know the original dimensions, reshape without padding
    if original_height is not None and original_width is not None:
        try:
            # Check if the sample count matches the expected grid size
            train_samples = x_train.shape[0]
            expected_samples = original_height * original_width
            
            if train_samples != expected_samples:
                # Adjust for train/test split
                test_ratio = x_test.shape[0] / (x_train.shape[0] + x_test.shape[0])
                print(f"Sample count ({train_samples}) doesn't match expected grid size ({expected_samples})")
                print(f"This may be due to train/test split (test ratio: {test_ratio:.2f})")
                
                # Use the exact sample count to derive grid dimensions
                total_samples = train_samples
                # Find factors close to original aspect ratio
                factor = np.sqrt(total_samples / (original_height * original_width))
                new_height = int(original_height * factor)
                new_width = int(total_samples / new_height)
                
                # Adjust to ensure exactly the right number of samples
                while new_height * new_width != total_samples:
                    if new_height * new_width < total_samples:
                        new_width += 1
                    else:
                        new_width -= 1
                    if new_width <= 0:
                        new_height -= 1
                        new_width = int(total_samples / new_height)
                
                print(f"Adjusted dimensions to {new_height}x{new_width} = {new_height * new_width} samples")
                original_height, original_width = new_height, new_width
            
            # Reshape to exact grid size with no padding
            num_features = x_train.shape[-1]
            x_train = x_train.reshape(original_height, original_width, num_features)
            y_train = y_train.reshape(original_height, original_width, y_train.shape[-1])
            
            # For test data, we need to determine its grid size separately
            test_samples = x_test.shape[0]
            test_height = int(np.sqrt(test_samples * original_height / original_width))
            test_width = int(test_samples / test_height)
            
            # Adjust to ensure exactly the right number of samples
            while test_height * test_width != test_samples:
                if test_height * test_width < test_samples:
                    test_width += 1
                else:
                    test_width -= 1
                if test_width <= 0:
                    test_height -= 1
                    test_width = int(test_samples / test_height)
            
            x_test = x_test.reshape(test_height, test_width, num_features)
            y_test = y_test.reshape(test_height, test_width, y_test.shape[-1])
            
            print(f"Final shapes:")
            print(f"x_train: {x_train.shape} - Grid: {original_height}x{original_width}")
            print(f"y_train: {y_train.shape}")
            print(f"x_test: {x_test.shape} - Grid: {test_height}x{test_width}")
            print(f"y_test: {y_test.shape}")
            
        except Exception as e:
            print(f"Error during reshaping: {e}")
            print("Falling back to original data format")
    else:
        # Use square root as a fallback if original dimensions unknown
        train_samples = x_train.shape[0]
        grid_size = int(np.sqrt(train_samples))
        # Adjust to get a perfect square
        while grid_size * grid_size > train_samples:
            grid_size -= 1
        
        print(f"Using grid size: {grid_size}x{grid_size} = {grid_size * grid_size} samples")
        print(f"Note: This will use only {grid_size * grid_size}/{train_samples} samples")
        
        # Use exact subset without padding
        num_features = x_train.shape[-1]
        x_train = x_train[:grid_size * grid_size].reshape(grid_size, grid_size, num_features)
        y_train = y_train[:grid_size * grid_size].reshape(grid_size, grid_size, y_train.shape[-1])
        
        # Do the same for test data
        test_samples = x_test.shape[0]
        test_grid = int(np.sqrt(test_samples))
        while test_grid * test_grid > test_samples:
            test_grid -= 1
        
        x_test = x_test[:test_grid * test_grid].reshape(test_grid, test_grid, num_features)
        y_test = y_test[:test_grid * test_grid].reshape(test_grid, test_grid, y_test.shape[-1])
        
        print(f"Final shapes:")
        print(f"x_train: {x_train.shape} - Grid: {grid_size}x{grid_size}")
        print(f"y_train: {y_train.shape}")
        print(f"x_test: {x_test.shape} - Grid: {test_grid}x{test_grid}")
        print(f"y_test: {y_test.shape}")
    
    return x_train, y_train, x_test, y_test


def display_rasters(rasters, titles=None, cmap='viridis', cols=4):
    """Display multiple rasters as subplots in a grid layout."""
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    
    num_images = len(rasters)
    rows = math.ceil(num_images / cols)  # Determine number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))  # Adjust subplot size
    axes = np.array(axes).flatten()  # Flatten axes for easy iteration

    for i, (filename, raster) in enumerate(rasters.items()):
        ax = axes[i]
        
        # Remove channel dimension if only 1 band
        if raster.shape[0] == 1:
            raster = raster[0]
        else:
            raster = np.mean(raster, axis=0)

        im = ax.imshow(raster, cmap=cmap)
        ax.set_title(titles[i] if titles else filename, fontsize=8)
        ax.axis('off')
        
        # Add colorbar for each subplot
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def process_and_reconstruct(x_bands, y_bands, tile_metadata, tile_size=250):
    """Process the datacube, tile rasters, and reconstruct the original rasters."""
    import numpy as np

    def reconstruct_raster(tiles, metadata, original_shape):
        c, h, w = original_shape
        reconstructed = np.zeros((c, h, w), dtype=tiles[0].dtype)
        for tile, meta in zip(tiles, metadata):
            if "tile_row" in meta and "tile_col" in meta:
                i, j = meta["tile_row"], meta["tile_col"]
                reconstructed[:, i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size] = tile
            else:
                raise KeyError("Metadata is missing 'tile_row' or 'tile_col' keys.")
        return reconstructed

    reconstructed_rasters = {}
    for filename, metadata in tile_metadata.items():
        tiles = x_bands.get(filename) or y_bands.get(filename)
        if tiles:
            reconstructed_rasters[filename] = reconstruct_raster(tiles, metadata, metadata[0]["original_shape"])

    return reconstructed_rasters


def split_train_test(X_train, y_train, test_size=0.2):
    """Split the training data into train and test chunks. Dimensions are (chunks, bands, height, width)"""
    from sklearn.model_selection import train_test_split

    # Flatten the data to 2D for splitting
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)

    # Split the data into train and test sets
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train_flat, y_train_flat, test_size=test_size, random_state=42
    )

    # Reshape back to original dimensions
    X_train_split = X_train_split.reshape(-1, *X_train.shape[1:])
    X_test_split = X_test_split.reshape(-1, *X_train.shape[1:])
    y_train_split = y_train_split.reshape(-1, *y_train.shape[1:])
    y_test_split = y_test_split.reshape(-1, *y_train.shape[1:])

    return X_train_split, X_test_split, y_train_split, y_test_split


def display_dict(data, indent=0):
    """
    Generated using Claude AI 3.7 Sonnet
    Display a dictionary that may contain non-JSON-serializable data.
    
    This function recursively traverses dictionary structures and displays their contents.
    When it encounters non-serializable objects, it describes them instead of attempting
    to serialize them.
    
    Parameters:
        data (dict): The dictionary to display.
        indent (int): The current indentation level for pretty printing.
        
    Returns:
        None
    """
    indent_str = "  " * indent
    
    # Handle different data types
    if isinstance(data, dict):
        if not data:
            print(f"{indent_str}{{}}")
            return
            
        print(f"{indent_str}{{")
        for key, value in data.items():
            key_repr = repr(key)
            print(f"{indent_str}  {key_repr}: ", end="")
            display_dict(value, indent + 1)
        print(f"{indent_str}}}")
            
    elif isinstance(data, (list, tuple)):
        container_type = "list" if isinstance(data, list) else "tuple"
        if not data:
            print(f"empty {container_type}")
            return
            
        print(f"{container_type} with {len(data)} items [")
        for item in data:
            print(f"{indent_str}  ", end="")
            display_dict(item, indent + 1)
        print(f"{indent_str}]")
            
    elif isinstance(data, (int, float, str, bool, type(None))):
        # These types are directly JSON serializable
        print(repr(data))
        
    elif hasattr(data, "__dict__"):
        # For objects with a __dict__, display their class and attributes
        cls_name = data.__class__.__name__
        print(f"<{cls_name} object with attributes>")
        display_dict(data.__dict__, indent + 1)
        
    elif callable(data):
        # Handle functions and other callables
        if hasattr(data, "__name__"):
            print(f"<function {data.__name__}>")
        else:
            print(f"<callable {type(data).__name__}>")
            
    elif hasattr(data, "__iter__"):
        # Handle other iterables
        print(f"<iterable {type(data).__name__}>")
        
    else:
        # Handle any other types by showing their type and repr
        type_name = type(data).__name__
        try:
            data_repr = repr(data)
            if len(data_repr) > 50:
                data_repr = data_repr[:47] + "..."
            print(f"<{type_name}: {data_repr}>")
        except Exception:
            print(f"<{type_name}: [repr failed]>")
    
    return None