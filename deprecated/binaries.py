import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import json

@dataclass
class ModelDataContainer:
    """Data container with all raster data and labels for machine learning."""
    fdir: Path
    tilenr: str
    projection: str

    def __post_init__(self):
        printb("Initializing data...", "info")
        self.data = {
            "metadata": {
                "tilenr": self.tilenr,
                "projection": self.projection,
            },
            "raster": {
                "svf": {
                    "file": self.fdir / "static" / self.tilenr / f"SkyViewFactor_{self.tilenr}.tif",
                    "data": None, # Will be loaded on demand
                    "metadata": {
                        "dynstat": "static",
                    },
                },
                "wall_height": {
                    "file": self.fdir / "static" / self.tilenr / f"wall_height_{self.tilenr}.tif",
                    "data": None, # Will be loaded on demand
                    "metadata": {
                        "dynstat": "static",
                    },
                },
                "wall_aspect": {
                    "file": self.fdir / "static" / self.tilenr / f"wall_aspect_{self.tilenr}.tif",
                    "data": None, # Will be loaded on demand
                    "metadata": {
                        "dynstat": "static",
                    },
                },
                "landcover": {
                    "file": self.fdir / "static" / self.tilenr / f"DO_landcover_3m_reclassified_final_{self.tilenr}.tif",
                    "data": None, # Will be loaded on demand
                    "metadata": {
                        "dynstat": "static",
                    },
                },
                "temperature_dir": {
                    "file": self.fdir / "dynamic" / "temperature",
                    "data": None, # Will be loaded on demand
                    "metadata": {
                        "dynstat": "dynamic",
                        "skipload": True,
                    },
                },
                "precipitation_dir": {
                    "file": self.fdir / "dynamic" / self.tilenr / f"precipitation_{self.tilenr}.tif",
                    "data": None, # Will be loaded on demand
                    "metadata": {
                        "dynstat": "dynamic",
                        "skipload": True,
                    },
                },
            },
        }


@dataclass
class ModelConfiguration:
    """
    This class contains the relevant hyperparameters for the model.
    """
    hyperparameters: dict

    def import_from_json(self, path: Path):
        """Import hyperparameters from a JSON file."""
        with open(path, "r") as f:
            self.hyperparameters = json.load(f)

    def export_to_json(self, path: Path):
        """Export hyperparameters to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.hyperparameters, f)
    
    def init_default(self):
        self.hyperparameters = {
            "model": {
                "n_estimators": [100, 200, 300],
                "max_depth": 3,
                "min_samples_split": 2,
                "learning_rate": 0.1,
                "loss": "ls",
            },
            "training": {
                "test_size": 0.2,
                "random_state": 42,
            }
        }


class ModelTrainer:
    """
    This class accepts a ModelConfiguration
    """




def printb(text: str, status: str = "empty", indent: int = 0):
    """Print text with status color."""
    # Define indentation
    indentation_spaces = ""
    for _ in range(indent):
        indentation_spaces += "  "

    # Print text with color
    match status:
        case "empty"|"": print(f"\033[0m         {indentation_spaces}{text}")
        case "ok": print(f"\033[92m[  OK  ] \033[0m{indentation_spaces}{text}")
        case "info": print(f"\033[94m[ INFO ] \033[0m{indentation_spaces}{text}")
        case "warn": print(f"\033[93m[ WARN ] \033[0m{indentation_spaces}{text}")
        case "error": print(f"\033[91m[ FAIL ] \033[0m{indentation_spaces}{text}")
        case "critical" : print(f"\033[41m[CRITIC] {indentation_spaces}{text}\033[0m")
        case _: 
            print(f"\033[41m[CRITIC] Unsupported status: {status}\033[0m")
            raise ValueError(f"Unsupported status: {status}")
        
def load_single_tile(l_data_container: ModelDataContainer) -> ModelDataContainer:
    """
    Load raster tiles for a given tile number.
    """
    def load_data(file: Path):
        """
        Robust raster loading function."
        """
        match file.suffix:
            case ".tif":
                # Load GeoTIFF to np.array
                with rasterio.open(file) as src:
                    data = src.read(1)
                    data = np.ma.masked_equal(data, src.nodata)
                    return data
            case _:
                raise ValueError(f"Unsupported file format: {file.suffix}")
    
    # Keep track of error levels throughout loading
    errorflag = 0

    # --- Removing skipped data ---
    printb("Removing skipped data...","",1)
    layers_to_remove = []
    for layer in l_data_container.data["raster"]:
        if l_data_container.data["raster"][layer]["metadata"].get("skipload", False):
            layers_to_remove.append(layer)
    for layer in layers_to_remove:
        l_data_container.data["raster"].pop(layer)
        printb(f"Removed {layer} due to skip flag","",2)

    # --- Load data ---
    printb(f"Loading data ...","",1)
    for layer in l_data_container.data["raster"]:
        try:
            with rasterio.open(l_data_container.data["raster"][layer]["file"]) as src:
                l_data_container.data["raster"][layer]["metadata"]["file_metadata"] = src.meta
                l_data_container.data["raster"][layer]["metadata"]["projection"] = src.crs.to_string()
                l_data_container.data["raster"][layer]["metadata"]["transform"] = src.transform
                l_data_container.data["raster"][layer]["metadata"]["shape"] = (src.meta["height"], src.meta["width"])

                if src.crs != l_data_container.data["metadata"]["projection"]:
                    printb(f"The projection of the raster file {layer} does not match the projection defined by the model container.", "warn", 2)
                    if errorflag < 1: errorflag = 1

            l_data_container.data["raster"][layer]["data"] = load_data(l_data_container.data["raster"][layer]["file"])
            printb(f"Loaded {layer} with shape {l_data_container.data['raster'][layer]['data'].shape}","",2)

        except Exception as e: 
            printb(f"Failed to load constituent {layer}: {e}", "critical",2)
            errorflag = 2

    # Load meteo csv data
    printb(f"Skipping meteo data for now [PENDING IMPLEMENTATION]...","warn",2)

    # --- Quality assurance ---
    printb("Performing quality assurance checks...","",1)
    qualflag = 0

    sets = {
        "shape": set(),
        "crs": set(),
    }
    
    for layer in l_data_container.data["raster"]:
        match l_data_container.data["raster"][layer]["metadata"]["dynstat"]:
            case "static":
                # Add shape to set
                sets["shape"].add(l_data_container.data["raster"][layer]["metadata"]["shape"])
                # Add crs to set
                sets["crs"].add(l_data_container.data["raster"][layer]["metadata"]["projection"])
            case "dynamic":
                printb(f"Skipping dynamic data check for now [PENDING IMPLEMENTATION]...","warn",2)
    
    # Assert that all data has the same shape
    for key, value in sets.items():
        if len(value) > 1:
            printb(f"Data has different {key}: {value}", "error",2)
            qualflag = 2
        else:
            printb(f"Data has consistent {key}: {value}. Writing to metadata...", "",2)
            l_data_container["metadata"][key] = value.pop()

    match qualflag:
        case 0: printb("Quality assurance checks passed successfully.", "ok",1)
        case 1: printb("Some quality assurance checks failed.", "warn",1)
        case 2:
            printb("Critial quality assurance checks failed.", "error",1)
            errorflag = 2

    # --- Print load status ---
    match errorflag:
        case 0: printb("All data loaded successfully.", "ok",1)
        case 1: printb("Some data was not loaded successfully.", "warn",1)
        case 2: 
            printb("Critial errors occurred during data loading.", "error",1)
            exit(1)

    return l_data_container, errorflag

def preprocess_raster_data(data_container):
    """
    Preprocesses raster data for the model.
    
    Args:
        data_container: ModelDataContainer instance with loaded raster data.
    
    Returns:
        Preprocessed raster data ready for model input.
    """
    printb("Preprocessing raster data...", "info")
    for layer in data_container.data["raster"]:
        try:
            # Normalize data to [0, 1]
            layer_raster_values = data_container.data["raster"][layer]["data"]
            layer_raster_values = (layer_raster_values - layer_raster_values.min()) / (layer_raster_values.max() - layer_raster_values.min())
            
            # Free up memory
            data_container.data["raster"][layer]["data"] = None
            data_container.data["raster"][layer]["data_preprocessed"] = layer_raster_values

        except Exception as e:
            printb(f"Failed to preprocess {layer}: {e}", "error", 2)
            exit(1)
    
    printb("Finished preprocessing raster data.", "ok")

    
    # Preprocess meteo data
    # (PENDING IMPLEMENTATION)

    return data_container









