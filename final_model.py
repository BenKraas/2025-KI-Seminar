# final layout for the model - this time I have a dataset with many tiles, thus I need to do partial fitting

# Imports
import copy
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rasterio as rio
import re
import tensorflow as tf
import time

from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.config.experimental import list_physical_devices, set_memory_growth


# Set the logging level to WARNING to suppress TensorFlow logs
tf.get_logger().setLevel('WARNING')

# setup globals
script_dir = Path(__file__).resolve().parent

# Set the level of the bprint function.
# 0 - debug
# 1 - info
# 2 - success
# 3 - warning
# 4 - error
# 5 - mute everything
bprint_console_level = 0
bprint_logging_level = 0



# Utility functions
def utility_init_logging():
    """
    Written by BK
    Function that initializes the logging.
    """
    global script_dir
    logging.basicConfig(
        filename=script_dir / "logs" / "log.txt",
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG
    )
    # Add the start time to the log file
    start_time = time.time()
    logging.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    return

def utility_show_model_summary(model: Sequential):
    """
    Show the model summary
    """
    bprint("Showing the model summary", level="starting")
    model.summary()

def bprint(text, indent=0, level="info", show="", **kwargs):
    """
    Written by BK with very little help from ChatGPT
    Function that prints the text with indentation and level.
    """
    # if bprint_level is not set, set it to 1 (debug)
    global bprint_console_level
    global bprint_logging_level
    global levels

    if 'bprint_console_level' not in globals():
        bprint_console_level = 1

    if 'bprint_logging_level' not in globals():
        bprint_logging_level = 1
    
    if 'levels' not in globals():
        levels = {
            "info": { # white color
                "color": "\033[0m",
                "prefix": "         ",
                "level": 1
            },
            "warning": { # yellow color
                "color": "\033[93m",
                "prefix": "WARNING: ",
                "level": 3
            },
            "error": { # red color
                "color": "\033[91m",
                "prefix": "ERROR:   ",
                "level": 4
            },
            "success": { # green color
                "color": "\033[92m",
                "prefix": "SUCCESS: ",
                "level": 4
            },
            "debug": { # blue color
                "color": "\033[94m",
                "prefix": "         ",
                "level": 0
            },
            "critical": { # magenta color
                "color": "\033[95m",
                "prefix": "CRITICAL ",
                "level": 4
            },
            "starting": { # cyan color
                "color": "\033[96m",
                "prefix": "START:   ",
                "level": 2
            },
        }
    if level not in levels:
        level = "info"

    # Check if the level is set to mute everything
    if levels[level]["level"] >= bprint_console_level:

        prefix = levels[level]["prefix"]
        indentation = "│ " * indent
        
        # Determine the color and prefix for console output
        if show:
            # Override the displayed level with the show level
            color = levels[show]["color"]
        else:
            color = levels[level]["color"]
        white = "\033[0m"

        # Print to console
        print(f"{color}{prefix}{indentation}{text}{white}", **kwargs)

    # Log to file if logging level allows
    if levels[level]["level"] >= bprint_logging_level:
        
        prefix = levels[level]["prefix"]
        indentation = "│ " * indent

        # Ensure the log directory exists
        log_dir = script_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "log.txt"

        # Write to the log file
        with open(log_file, "a") as log:
            log.write(f"{prefix}{indentation}{text}\n")

    return

# Preprocessing functions
def preprocessing_get_filelist(data_fdir_path: Path) -> dict:
    """
    Written by BK
    Function that prepares a dictionary with the file names of the images.
    """
    bprint(f"Preprocessing the file list in {data_fdir_path}", level="starting")
    # Get a list of all files in the directpathory
    fdict = {}

    template_subdict = {
        "X": {
            "CDSM": "",
            "DTM": "",
            "DTM+DSM": "",
            "landcover": "",
            "skyviewfactor": "",
            "wall_height": "",
        },
        "y": {
            "tmrt": "",
        },
        "train_or_test": ""
    }

    dirlist = os.listdir(data_fdir_path)
    # keep ony the directories that are in regex format XX_XX where X is a digit
    dirlist = [d for d in dirlist if re.match(r'^\d{2}_\d{2}$', d)]
    initial_dir_length = len(dirlist)
    bprint(f"Initial directory length: {initial_dir_length}", indent=1, level="info")

    # keep only dirs that have a .tif file with string *"Tmrt_3m"* in the name with wildards around
    dirlist = [d for d in dirlist if any(re.search(r'Tmrt_3m', f) for f in os.listdir(data_fdir_path / d))]
    bprint(f"Directory length after filtering: {len(dirlist)}", indent=1, level="info")

    # Loop through the directories and get the file names into a template subdict copy and fill it
    for grid_dir in dirlist:
        bprint(f"Processing directory: {grid_dir}", indent=2, level="debug")
        subdict = copy.deepcopy(template_subdict)
        n_files_found = os.listdir(data_fdir_path / grid_dir)
        for f in os.listdir(data_fdir_path / grid_dir):
            # get full path to the file
            fullpath = data_fdir_path / grid_dir / f

            if re.search(r'Tmrt_3m', f):
                subdict["y"]["tmrt"] = fullpath
            elif re.search(r'DO_DSM', f):
                subdict["X"]["CDSM"] = fullpath
            elif re.search(r'DO_DTM_mosaic', f):
                subdict["X"]["DTM"] = fullpath
            elif re.search(r'DTM\+DSM', f):
                subdict["X"]["DTM+DSM"] = fullpath
            elif re.search(r'landcover', f):
                subdict["X"]["landcover"] = fullpath
            elif re.search(r'SkyViewFactor', f):
                subdict["X"]["skyviewfactor"] = fullpath
            elif re.search(r'wall_height', f):
                subdict["X"]["wall_height"] = fullpath

        # verify that all files are present
        if not all(subdict["X"].values()) or not subdict["y"]["tmrt"]:
            # if not all files are present, print a warning and skip the directory
            bprint(f"Not all files are present in directory {data_fdir_path / grid_dir}", indent=2, level="warning")
            continue
        else:
            bprint(f"All files found in directory {data_fdir_path / grid_dir}", indent=2, level="debug", show="success")
            fdict[grid_dir] = subdict

    # Check if the dictionary is empty
    if not fdict:
        bprint(f"No files found in the directory {data_fdir_path / grid_dir}", level="error")
        raise ValueError("No files found in the directory")
        return None
    else:
        bprint(f"Found {len(fdict)} directories with files", level="success")	
        return fdict

def preprocessing_train_test_split(fdict: dict, test_size: float = 0.2) -> tuple:
    """
    Written by BK with a bit of help by ChatGPT
    Function that splits the file list into train and test sets.
    """
    bprint("Splitting the file list into train and test sets", level="starting")
    
    # Get the keys of the dictionary
    keys = list(fdict.keys())
    
    # Shuffle the keys
    np.random.shuffle(keys)
    
    # Calculate the split index
    split_index = int(len(keys) * (1 - test_size))
    
    # Split the keys into train and test sets
    train_keys = keys[:split_index]
    test_keys = keys[split_index:]
    
    # Create the train and test dictionaries
    train_dict = {k: fdict[k] for k in train_keys}
    test_dict = {k: fdict[k] for k in test_keys}

    # Add the train_or_test key to the dictionaries
    for k in train_dict.keys():
        train_dict[k]["train_or_test"] = "train"
    for k in test_dict.keys():
        test_dict[k]["train_or_test"] = "test"

    # check if the split was successful
    if len(train_dict) + len(test_dict) != len(fdict):
        bprint("The split was not successful - total number of files does not match", level="error")
        raise ValueError()
    
    # confirm the dicts are not empty
    if not train_dict:
        bprint("The train dictionary is empty", level="error")
        raise ValueError()
    if not test_dict:
        bprint("The test dictionary is empty", level="error")
        raise ValueError()
    
    bprint(f"Train dictionary length: {len(train_dict)}", indent=1, level="info")
    bprint(f"Test dictionary length: {len(test_dict)}", indent=1, level="info")

    bprint(f"Successfully split the file list into train and test sets with ratio {1-test_size}:{test_size}", level="success")
    return train_dict, test_dict

def preprocessing_get_dict_tensors(fdict: dict, grid_identifier: str) -> tuple:
    """
    Written by BK with help from ChatGPT
    Function that gets X and y tensors from the input dictionary.
    """

    def load_dictionary_fpaths(single_dict: dict, expected_shape: tuple) -> np.ndarray:
        """
        Written by BK
        Function that loads the file paths from the dictionary.
        """
        global script_dir

        bprint("Loading the file paths from the dictionary", indent=1, level="debug")
        # Make a copy of the dictionary
        single_dict = copy.deepcopy(single_dict)

        final_tensor = np.zeros(expected_shape)
        bprint(f"Final tensor initialized with shape: {final_tensor.shape}", indent=2, level="debug")

        channel_idx = 0

        # Get the file paths from the dictionary
        for _, value in single_dict.items():
            bprint(f"File path found: {value}", indent=2, level="debug")

            # Try to open the file with rasterio
            try:
                with rio.open(script_dir / value) as src:
                    # Check if the file is opened successfully
                    if src is None:
                        bprint(f"File could not be opened: {value}", indent=3, level="error")
                        return None
                    bprint(f"File opened successfully... Shape: {src.shape}", indent=3, level="debug")

                    # Read the data from the file
                    data = np.array(src.read())
                    bprint(f"Data read   successfully ...", indent=3, level="debug")

                    # Check if the data is empty
                    if data.size == 0:
                        bprint(f"Data is empty: {data}", indent=3, level="error")
                        return None
                    
                    # Check if the data is a 3D array
                    if len(data.shape) != 3:
                        bprint(f"Data is not a 3D array: {data.shape}", indent=3, level="error")
                        return None
                    
                    # Insert the data into the final tensor appropriately
                    if channel_idx < expected_shape[2]:
                        final_tensor[:, :, channel_idx] = data[0, :, :]
                        bprint(f"Data inserted into final tensor at channel {channel_idx}", indent=3, level="debug")
                        channel_idx += 1
                    else:
                        bprint(f"Channel index out of range: {channel_idx} ... This should not happen - it means that more images were found than expected", indent=5, level="error")

            except Exception as e:
                bprint(f"Error opening file {value}: {e}", indent=3, level="error")
                return None

        bprint(f"Final tensor shape: {final_tensor.shape}", indent=4, level="debug")
        # Check if the final tensor contains any channels (axis 2) that are empty
        if np.any(np.all(final_tensor == 0, axis=(0, 1))):
            bprint(f"Final tensor contains empty channels", indent=3, level="error")
            raise ValueError("Final tensor contains empty channels")

        # Assert the shape of the data
        if final_tensor.shape != expected_shape:
            bprint(f"Final Tensor shape does not match expected shape: {final_tensor.shape} != {expected_shape}", indent=3, level="error")
            return None

        bprint(f"Final tensor shape matches expected shape: {final_tensor.shape} == {expected_shape}", indent=1, level="debug", show="success")
        return final_tensor

    global bprint_console_level

    # Get the tensors from the dictionary
    bprint(f"Getting the tensors from the dictionary {grid_identifier}", level="starting")

    X_tensor = load_dictionary_fpaths(fdict["X"], expected_shape=(1000, 1000, 6))
    y_tensor = load_dictionary_fpaths(fdict["y"], expected_shape=(1000, 1000, 1))

    # Check if the tensors are empty
    if X_tensor is None or y_tensor is None:
        bprint(f"X_tensor or y_tensor is None", level="error")
    else:
        bprint(f"Successfully loaded the tensors", level="success")

    return X_tensor, y_tensor

def preprocessing_prepare_training_data(path: Path) -> None:
    """
    Written by BK
    Build the training data from raw dataset by loading file lists, 
    splitting into train/test sets, and creating the training tensors.

    Parameters:
        path (Path): Path to the directory containing the data files.

    Returns:
        None
    """
    # Call the function to get the file names
    fdict = preprocessing_get_filelist(path)

    # Split the file list into train and test sets
    train_fdict, test_fdict = preprocessing_train_test_split(fdict)

    # Get the entire training data tensor
    bprint("Loading training data", level="starting")

    final_tensors = {
        "X_train": [],
        "y_train": [],
        "X_test": [],
        "y_test": []
    }

    for tt_dict in [train_fdict, test_fdict]:

        for grid_identifier, grid_dict in tt_dict.items():
            bprint(f"Loading data for grid {grid_identifier}", level="debug")
            X_data, y_data = preprocessing_get_dict_tensors(grid_dict, grid_identifier)
            bprint(f"Data loaded successfully for grid {grid_identifier}", level="debug", show="success")

            # Check if the data is empty
            if X_data is None or y_data is None:
                bprint(f"X_train or y_train is None", level="error")
                continue

            # Insert the data into the final tensor
            if grid_dict["train_or_test"] == "train":
                bprint(f"Data is for training set", level="debug")
                final_tensors["X_train"].append(X_data)
                final_tensors["y_train"].append(y_data)
            elif grid_dict["train_or_test"] == "test":
                bprint(f"Data is for test set", level="debug")
                final_tensors["X_test"].append(X_data)
                final_tensors["y_test"].append(y_data)

            bprint(f"Data inserted into final tensor for grid {grid_identifier}", level="debug")


    # Convert the final tensor to a numpy array
    bprint("Converting the final tensor to a numpy array", level="starting")
    for key, value in final_tensors.items():
        final_tensors[key] = np.array(value)
        bprint(f"Final tensor {key} shape: {final_tensors[key].shape}", level="info")

    # Save the final tensor to a file
    bprint("Saving the final tensors to a file", level="starting")
    for key, value in final_tensors.items():

        np.save(f"data/final_tensor_{key}.npy", value)
        bprint(f"Final tensor {key} saved to file", level="info")
    
    bprint("Final tensors saved successfully", level="success")

    return 

def preprocess_force_rebuild():
    """
    Function that forces the rebuild of the npy tensors.
    This function is called when the script is run with the --force_rebuild flag.
    """
    bprint("Session mode: {session_mode} - Rebuilding the npy tensors", level="info")
    bprint("This will delete all existing npy tensors", level="warning")
    input("Press Enter to confirm this action...")
    # Rebuild the npy tensors
    preprocessing_prepare_training_data(Path("data/static"))
    bprint("Npy tensors rebuilt successfully", level="success")
    return

# Model compilation functions
def compilation_model_01(input_shape: tuple) -> Sequential:
    """
    This model uses a reduction and upscaling approach with intermediate data feeding to preserve pixel values.
    It takes as input a 4D tensor of shape (item_id, height, width, channels) and outputs a tensor of shape (height, width, 1).
    """
    bprint("Creating the model with reduction and upscaling approach", level="starting")
    
    model = Sequential([
        # Reduction phase
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape[1:]),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        
        # Bottleneck
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        
        # Upscaling phase
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        
        # Final output layer
        Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    bprint("Model created successfully with reduction and upscaling", level="success")
    return model

def compilation_model_02(input_shape: tuple) -> Sequential:
    """
    This model uses a reduction and upscaling approach with intermediate data feeding to preserve pixel values.
    It takes as input a 4D tensor of shape (item_id, height, width, channels) and outputs a tensor of shape (height, width, 1).
    """
    bprint("Creating the model with reduction and upscaling approach", level="starting")
    
    model = Sequential([
        # Reduction phase
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape[1:]),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),

        # Add dropout layer to reduce overfitting
        Dropout(0.2),
        
        # Bottleneck
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        
        # Upscaling phase
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        
        # Final output layer
        Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    bprint("Model created successfully with reduction and upscaling", level="success")
    return model

# Add the decorator to register the custom loss function
@tf.keras.utils.register_keras_serializable()
def weighted_mse(y_true, y_pred):
    """
    Custom weighted mean squared error loss function.
    Gives higher weight to non-black pixels (values > 0.1).
    """
    weights = tf.where(tf.greater(y_true, 0.1), 5.0, 1.0)
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))

def compilation_model_03(input_shape: tuple, learning_rate: float = 0.001) -> Sequential:
    """
    Mainly written by Claude Sonnet 3.7
    Enhanced preprocessing model with better initialization, skip connections,
    and improved activation functions to prevent the "black field" problem.
    It takes as input a 4D tensor of shape (item_id, height, width, channels) and outputs a tensor of shape (height, width, 1).
    """
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Input, Concatenate, BatchNormalization, LeakyReLU
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    
    bprint("Creating improved model with skip connections and better activations", level="starting")
    
    # Use functional API instead of Sequential for skip connections
    inputs = Input(shape=input_shape[1:])
    
    # Encoder path with LeakyReLU and BatchNormalization
    # Use he_normal initialization for better gradient flow
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    skip1 = x  # Store for skip connection
    
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    skip2 = x  # Store for skip connection
    
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    
    # Bottleneck
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Decoder path with skip connections
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    x = UpSampling2D((2, 2))(x)
    # Add skip connection from encoder
    x = Concatenate()([x, skip2])
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    x = UpSampling2D((2, 2))(x)
    # Add skip connection from encoder
    x = Concatenate()([x, skip1])
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Final output layer with linear activation for preserving pixel values
    outputs = Conv2D(1, (3, 3), padding='same', activation='linear')(x)
    
    # Create and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=weighted_mse,
        metrics=['mean_squared_error']
    )
    
    bprint("Improved model created successfully with skip connections and better activations", level="success")
    return model


# Main process functions
def main_process_train_model(script_dir, args, session_mode):
    bprint(f"Session mode: {session_mode} - Training model", level="info")

    # Check if data dir contains tensors
    final_tensor_X_train = script_dir / "data" / "final_tensor_X_train.npy"
    if not final_tensor_X_train.is_file():
        bprint("Tensors not found, requiring rebuild, please run the script with --mode force_rebuild first", level="critical")
        bprint("Exiting the script", level="error")
        input("Press Enter to exit...")
        exit(1)

    # Load the final tensors
    bprint("Found final tensors, loading training data", level="starting")
    final_tensor_X_train = np.load("data/final_tensor_X_train.npy")
    final_tensor_y_train = np.load("data/final_tensor_y_train.npy")
    bprint("Final tensors training data loaded successfully", level="success")

    # get the number of tiles in the training set
    train_tiles_number = final_tensor_X_train.shape[0]
    bprint(f"Number of tiles in the training set: {train_tiles_number}", level="info")

    # Initialize the model
    bprint("Initializing the model", level="starting")
    model = compilation_model_03(input_shape=(train_tiles_number, 1000, 1000, 6), learning_rate=args.learning_rate)
    utility_show_model_summary(model)
    bprint("Model initialized successfully", level="success")

    # Train the model
    bprint("Training the model", level="starting")
    # Limit GPU memory usage
    gpus = list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)]
                    )
            bprint("GPU memory limit set to 5GB", level="info")
        except RuntimeError as e:
            bprint(f"Error setting GPU memory limit: {e}", level="error")

    # Set memory growth for GPUs
    for gpu in gpus:
        try:
            set_memory_growth(gpu, True)
        except RuntimeError as e:
            bprint(f"Error setting memory growth: {e}", level="error")

    # Train the model with GPU
    bprint("Training the model with GPU", level="starting")
        # Check if GPU is available
    if not tf.config.list_physical_devices('GPU'):
        bprint("GPU not available, training on CPU", level="warning")
    else:
        bprint("GPU available, training on GPU", level="success")
        
    # Train the model
    bprint("MAIN: Training the model", level="starting")
    with tf.device('/GPU:0'):
        history = model.fit(final_tensor_X_train, final_tensor_y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.2)
    bprint("Model trained successfully", level="success")

    # Save the model
    bprint("Saving the model", level="starting")
    model.save("models/final_model.keras")
    bprint("Model saved successfully", level="success")

    # Save the history
    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history.history)

    # save to csv: 
    hist_csv_file = 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    bprint("History saved successfully", level="success")

    # Clear the session
    bprint("Clearing the session", level="starting")
    keras.backend.clear_session()
    bprint("Session cleared successfully", level="success")

def main_process_test_model(session_mode):
    bprint(f"Session mode: {session_mode} - Testing model", level="starting")

    # Load the model
    bprint("Loading the model", level="starting")
    model = keras.models.load_model("models/final_model.keras")
    bprint("Model loaded successfully", level="success")


    # Evaluate on test data
    try:
        bprint("Loading test data", level="starting")
        final_tensor_X_test = np.load("data/final_tensor_X_test.npy")
        final_tensor_y_test = np.load("data/final_tensor_y_test.npy")
        bprint("Test data loaded successfully", level="success")
    except FileNotFoundError:
        # critical error - test data not found
        bprint("Test data not found - please run the script with --mode force_rebuild first", level="critical")
        bprint("Exiting the script", level="error")
        input("Press Enter to exit...")
        exit(1)
        
    # Prediction
    bprint("Predicting on test data", level="starting")
    predictions = model.predict(final_tensor_X_test[0:4], batch_size=1)
    bprint("Predictions made successfully", level="success")

    # Show the first three predictions next to the original data
    bprint("Showing predictions", level="starting")
    plot, ax = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(3):
        # Ground truth
        # Calculate the min and max of the ground truth tensor
        gt_min = np.min(final_tensor_y_test)
        gt_max = np.max(final_tensor_y_test)

        im1 = ax[i, 0].imshow(final_tensor_y_test[i, :, :, 0], cmap='viridis', vmin=gt_min, vmax=gt_max)
        ax[i, 0].set_title(f"Ground Truth {i+1}")
        plot.colorbar(im1, ax=ax[i, 0], orientation='vertical')

        # Prediction clipped to ground truth min/max
        im2 = ax[i, 1].imshow(np.clip(predictions[i, :, :, 0], gt_min, gt_max), cmap='viridis', vmin=gt_min, vmax=gt_max)
        ax[i, 1].set_title(f"Prediction {i+1} (Clipped to GT)")
        plot.colorbar(im2, ax=ax[i, 1], orientation='vertical')

        # Prediction with its own min/max
        im3 = ax[i, 2].imshow(predictions[i, :, :, 0], cmap='viridis')
        ax[i, 2].set_title(f"Prediction {i+1} (Own Min/Max)")
        plot.colorbar(im3, ax=ax[i, 2], orientation='vertical')

    plot.tight_layout()
    plot.show()

    input("Press Enter to continue...")

    # Save to file with timestamp into data/pngs
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = script_dir / "data" / "pngs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predictions_{timestamp}.png"
    plot.savefig(output_path)
    bprint(f"Predictions saved to {output_path}", level="success")

    # Show the model summary
    utility_show_model_summary(model)
    
    # Show the model architecture
    bprint("Showing the model architecture", level="starting")
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    bprint("Model architecture shown successfully", level="success")

# Command line parser
def cli_parser():
    """
    Command line parser for the script
    """
    import argparse
    parser = argparse.ArgumentParser(description="Train and test the model")
    parser.add_argument("-m", "--mode", type=str, choices=["train", "test", "both"], default="train", help="Mode of operation")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="Number of epochs to train the model")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size to use for training")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Learning rate to use for training")
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild the npy tensors")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Written by BK with only little help from ChatGPT
    # clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    tf.get_logger().setLevel('WARNING')
    utility_init_logging()
    bprint("Starting the script", level="starting")

    # Parse the command line arguments
    args = cli_parser()

    session_mode = args.mode
    if args.force_rebuild:
        session_mode = "force_rebuild"
        preprocess_force_rebuild()

    if session_mode == "train" or session_mode == "both":
        main_process_train_model(script_dir, args, session_mode)

    if session_mode == "test" or session_mode == "both":
        main_process_test_model(session_mode)
        