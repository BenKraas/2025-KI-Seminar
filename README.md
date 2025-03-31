# 2025 AI Seminar
Main repositoriy for the final project of the AI-Seminar at RUB.

# Project Structure
The project is structured as follows:
```
.
├── data
│   ├── output
│   │   ├── pngs
│   │   │   └── ... png files of the training results
│   │   └── ... files
│   ├── static
│   │   ├── 01_04
│   │   │   ├── 01_04_Tmrt_3m_v0.6.0_2024_093_1000D.tif
│   │   │   ├── DO_DSM_mosaic_3m_bilinear_masked_trees_01_04.tif
│   │   │   ├── DO_DTM_mosaic_3m_bilinear_01_04.tif
│   │   │   ├── DO_DTM+DSM_mosaic_3m_01_04.tif
│   │   │   ├── DO_landcover_3m_reclassified_final_01_04.tif
│   │   │   ├── SkyViewFactor_01_04.tifl
│   │   │   ├── wall_height_01_04.tif
│   │   │   └── ... other files that are not needed
│   │   ├── 01_05
│   │   └── ...
│   ├── final_tensor_X_test.npy
│   ├── final_tensor_X_train.npy
│   ├── final_tensor_y_test.npy
│   └── final_tensor_y_train.npy
├── deprecated
│   └── ... old code that is not used anymore
├── logs
│   └── ... logs from the training process
├── models (not on github)
│   ├── best_unet_model.h5
│   ├── final_model.keras
│   └── ... other models
├── pngs
│   └── ... png files of the training results
├── environment.yaml
├── environment_short.yml
├── final_model.py				- Main file to run and test the model
├── model.png
├── README.md
├── test_gpu_availability.ipynb - Testing notebook to check if GPU is available
└── ... further less important files

```

# Requirements
The project requires tensorflow; the progress of setting this up can be painful. Ensure you build the correct version of tensorflow for your GPU and python version.

# Running the Project
The main script is `final_model.py`. It contains the code to train and test the model. The script is designed to be run from the command line. You can run it with the following command:

```bash
python final_model.py -m MODE_OF_OPERATION 
```

Flags:
- `-m` or `--mode`: The mode of operation. It can be one of the following:
  - `train`: Train the model.
  - `test`: Test the model.
  - `both`: Train and test the model.
- `-e` or `--epochs`: The number of epochs to train the model. Default is 10.
- `-b` or `--batch_size`: The batch size to use for training. Default is 32.
- `-l` or `--learning_rate`: The learning rate to use for training. Default is 0.001.
- `--force_rebuild`: Force the model to be rebuilt. This is useful only if you have changed the raw data eg. changed the area of interest. 


# Example Commands
These are some example commands to run the script - the third command is the one that was used to get the result on the poster.

```bash
python final_model.py -m "test" -b 2 --learning_rate 0.001 -e 7
python final_model.py -m "both" -b 3 --learning_rate 0.0005 -e 30
python final_model.py -m "both" -b 2 --learning_rate 0.0001 -e 200
```