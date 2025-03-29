import tensorflow as tf
import numpy as np

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# Example: Create dummy data. Replace these with your actual data.
num_samples = 10  # for example
x_train = np.random.rand(num_samples, 250, 250, 6).astype(np.float32)
y_train = np.random.rand(num_samples, 250, 250, 1).astype(np.float32)

# Define training parameters
batch_size = 32
epochs = 10

# Create a tf.data.Dataset pipeline
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=num_samples)  # Shuffle entire dataset
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

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
    }
}

# Create and compile the model
model = create_compiled_conv2d_model(param_dict)
model.summary()

# Train the model using the tf.data pipeline
with tf.device('/GPU:0'):
    model.fit(train_dataset, epochs=epochs)
