### Declaration of code authenticity ###
# This code was written by Ben Kraas with the help of AI-Tools such as Copilot, Claude and ChatGPT.
# Especially boilerplate was more often written by the AI than by me.

# Import standard libraries
from pathlib import Path

# Import third-party libraries
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow.keras import layers, models, Input

# python script imports
from deprecated.binaries import *



def main_single_tile(data_dir: Path, tilenr: str, epsg: str = "EPSG:25832"):
    """
    Main function for processing a single tile."
    """

    # Load data
    
    data_container = ModelDataContainer(data_dir, tilenr, epsg)

    # Load single tile data into data_container
    data_container, flag = load_single_tile(data_container)

    # Preprocess data
    data_container = preprocess_raster_data(data_container)

    # Setup model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(data_container.data["raster"]["svf"]["data"].shape[0],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))





if __name__ == "__main__":
    main_single_tile(data_dir=Path("data"), 
                     tilenr="02_04", 
                     epsg="EPSG:25832")