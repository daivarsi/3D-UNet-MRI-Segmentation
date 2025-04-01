# -*- coding: utf-8 -*-
"""MRISegmentationProject.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16lDEQ1Zxqc3CrlC0fc_EW5leoXd9KNzy
"""

!nvidia-smi

!pip install tensorflow torch numpy matplotlib nibabel

from google.colab import drive
drive.mount('/content/drive')

base_path = "/content/drive/MyDrive/archive/BraTS2020_training_data/content/data/"

import h5py
import os

sample_file_path = os.path.join(base_path, "volume_351_slice_64.h5")  # Or any other .h5 file

image_data = None
mask_data = None

try:
    with h5py.File(sample_file_path, 'r') as hf:
        print("Keys in h5 file:", list(hf.keys()))  # Print the keys

        image_data = hf['image'][:]  # Read image data
        mask_data = hf['mask'][:]    # Read mask data

        print("Image data shape:", image_data.shape)
        print("Mask data shape:", mask_data.shape)

except FileNotFoundError:
    print(f"Error: File not found: {sample_file_path}")

# You can now use image_data and mask_data outside the try block (if they were read)
if image_data is not None:
    # Do something with image_data
    pass

print("Image data shape:", image_data.shape)

import matplotlib.pyplot as plt

# Assuming image_data has shape (height, width, channels)
channel_titles = ["T1-weighted", "T2-weighted", "FLAIR", "T1-weighted with contrast"]

for channel in range(image_data.shape[2]):
    plt.imshow(image_data[:, :, channel], cmap="gray")
    plt.title(channel_titles[channel])  # Use the channel title from the list
    plt.show()

plt.imshow(mask_data, cmap="jet", alpha=0.5)
plt.title("Segmentation Mask")
plt.show()

import os
import random
import h5py
import numpy as np

base_path = "/content/drive/MyDrive/archive/BraTS2020_training_data/content/data/"

tumor_files = []  # List to store files with tumor presence
sample_size = 1000 #Adjust sample size as needed

all_files = [f for f in os.listdir(base_path) if f.endswith(".h5")]

sampled_files = random.sample(all_files, min(sample_size, len(all_files)))

for file_name in sampled_files:
    file_path = os.path.join(base_path, file_name)

    try:
        with h5py.File(file_path, 'r') as hf:
            mask_data = hf['mask'][:]

        # Check for non-zero mask values (tumor presence)
        unique_mask_values = np.unique(mask_data)
        if len(unique_mask_values) > 1:  # More than just background
            tumor_files.append(file_name)
            print(f"Tumor found in: {file_name}, unique mask values: {unique_mask_values}")
        else:
            print(f"No tumor in: {file_name}")

    except Exception as e:
        print(f"An error occurred: {e}")

print("\nFiles with potential tumor regions:")
print(tumor_files)
print(f"\n{len(tumor_files)} files out of {len(sampled_files)} files contain tumors.")

tumor_volumes = []
for file_name in tumor_files:
    if 'volume_' in file_name:
        volume_id = file_name.split('volume_')[1].split('_slice_')[0]
        if volume_id not in tumor_volumes:
            tumor_volumes.append(volume_id)

print(tumor_volumes)

import os

file_path = "/content/drive/MyDrive/archive/BraTS2020_training_data/content/data/volume_128_slice_155.h5"
directory = os.path.dirname(file_path)

if os.path.exists(directory):
    print(f"Files in directory '{directory}':")
    for filename in os.listdir(directory):
        print(filename)
else:
    print(f"Directory '{directory}' does not exist.")

import h5py
file_path = "/content/drive/MyDrive/archive/BraTS2020_training_data/content/data/volume_1_slice_0.h5"

with h5py.File(file_path, 'r') as hf:
    print("Keys in the .h5 file:", list(hf.keys()))

import h5py
import os
import numpy as np

data_dir = '/content/drive/MyDrive/archive/BraTS2020_training_data/content/data/'  # Correct data directory
selected_volume_id = '1'  # Choose a volume ID

#Data Loading
loaded_data = {}

for file_name in os.listdir(data_dir):
    if file_name.endswith('.h5') and f'volume_{selected_volume_id}_' in file_name:
        file_path = os.path.join(data_dir, file_name)
        with h5py.File(file_path, 'r') as hf:
            mri_data = np.array(hf['image'])  # Use 'image' key
            seg_mask = np.array(hf['mask'])   # Use 'mask' key
            loaded_data.setdefault(selected_volume_id, {'image': [], 'mask': []})
            loaded_data[selected_volume_id]['image'].append(mri_data)
            loaded_data[selected_volume_id]['mask'].append(seg_mask)

#Convert the lists to numpy arrays.
for volume_no in loaded_data:
    loaded_data[volume_no]['image'] = np.array(loaded_data[volume_no]['image'])
    loaded_data[volume_no]['mask'] = np.array(loaded_data[volume_no]['mask'])

def reconstruct_3d_volumes(loaded_data):
    reconstructed_volumes = {}
    for volume_id, data in loaded_data.items():
        z_dim = data['image'].shape[0]
        y_dim = data['image'].shape[1]
        x_dim = data['image'].shape[2]

        reconstructed_volumes[volume_id] = {
            'image': np.zeros((z_dim, y_dim, x_dim, data['image'].shape[3])),
            'mask': np.zeros((z_dim, y_dim, x_dim))
        }

        for i in range(z_dim):
            reconstructed_volumes[volume_id]['image'][i] = data['image'][i]
            reconstructed_volumes[volume_id]['mask'][i] = data['mask'][i][:,:,0] # Take the first channel of the mask.

    return reconstructed_volumes

reconstructed_data = reconstruct_3d_volumes(loaded_data)

print(reconstructed_data[selected_volume_id]['image'].shape)
print(reconstructed_data[selected_volume_id]['mask'].shape)

import numpy as np

#Preprocessing
def preprocess_volume(reconstructed_data):
    """
    Normalizes the MRI data using z-score normalization.

    Args:
        reconstructed_data: A dictionary containing the reconstructed MRI data and mask.

    Returns:
        A dictionary containing the normalized MRI data and the original mask.
    """

    mri_data = reconstructed_data['1']['image']
    mask_data = reconstructed_data['1']['mask']

    # Z-score normalization for each channel of the MRI data.
    normalized_mri_data = np.zeros_like(mri_data, dtype=np.float32)
    for channel in range(mri_data.shape[-1]):  # Iterate over channels
        channel_data = mri_data[..., channel]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std != 0:  # Avoid division by zero
            normalized_mri_data[..., channel] = (channel_data - mean) / std
        else:
            normalized_mri_data[..., channel] = channel_data  # Keep original if std is 0

    return {'image': normalized_mri_data, 'mask': mask_data}

#Preprocess the data.
preprocessed_data = preprocess_volume(reconstructed_data)

print(preprocessed_data['image'].shape)
print(preprocessed_data['mask'].shape)
print(f'MRI data mean: {np.mean(preprocessed_data["image"])}')
print(f'MRI data std: {np.std(preprocessed_data["image"])}')

import tensorflow as tf
from tensorflow.keras import layers

def build_3d_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder (Downsampling)
    conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    # Bottleneck
    conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(conv3)

    # Decoder (Upsampling)
    up4 = layers.Conv3DTranspose(64, 2, strides=(2, 2, 2), padding='same')(conv3)
    concat4 = layers.concatenate([up4, conv2], axis=-1)
    conv4 = layers.Conv3D(64, 3, activation='relu', padding='same')(concat4)
    conv4 = layers.Conv3D(64, 3, activation='relu', padding='same')(conv4)

    up5 = layers.Conv3DTranspose(32, 2, strides=(2, 2, 2), padding='same')(conv4)
    concat5 = layers.concatenate([up5, conv1], axis=-1)
    conv5 = layers.Conv3D(32, 3, activation='relu', padding='same')(concat5)
    conv5 = layers.Conv3D(32, 3, activation='relu', padding='same')(conv5)

    # Output layer
    outputs = layers.Conv3D(1, 1, activation='sigmoid')(conv5)  # 1 channel for segmentation mask

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Build the model
input_shape = (152, 240, 240, 4)  # Adjusted input shape
model = build_3d_unet(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy') #Example loss.

#Print model summary.
model.summary()

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

# Mock data (replace with your actual data loading)
def generate_mock_data(num_volumes, z_dim, y_dim, x_dim, channels):
    mri_data = np.random.rand(num_volumes, z_dim, y_dim, x_dim, channels).astype(np.float32)
    mask_data = np.random.randint(0, 2, (num_volumes, z_dim, y_dim, x_dim, 1)).astype(np.float32)
    return mri_data, mask_data

num_volumes = 10  # Example number of volumes
z_dim = 152
y_dim = 240
x_dim = 240
channels = 4

mri_data, mask_data = generate_mock_data(num_volumes, z_dim, y_dim, x_dim, channels)

# Split into training and validation sets
train_ratio = 0.8
split_index = int(num_volumes * train_ratio)

train_mri = mri_data[:split_index]
train_mask = mask_data[:split_index]
val_mri = mri_data[split_index:]
val_mask = mask_data[split_index:]

# 3D U-Net Model Definition
def unet_3d(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder
    conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv3D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv3D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv3D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    # Bottleneck
    conv4 = layers.Conv3D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv3D(256, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up5 = layers.Conv3DTranspose(128, 2, strides=(2, 2, 2), padding='same')(conv4)
    merge5 = layers.concatenate([up5, conv3], axis=4)
    conv5 = layers.Conv3D(128, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv3D(128, 3, activation='relu', padding='same')(conv5)

    up6 = layers.Conv3DTranspose(64, 2, strides=(2, 2, 2), padding='same')(conv5)
    merge6 = layers.concatenate([up6, conv2], axis=4)
    conv6 = layers.Conv3D(64, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv3D(64, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv3DTranspose(32, 2, strides=(2, 2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([up7, conv1], axis=4)
    conv7 = layers.Conv3D(32, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv3D(32, 3, activation='relu', padding='same')(conv7)

    # Output
    outputs = layers.Conv3D(1, 1, activation='sigmoid')(conv7)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Dice Loss and Dice Coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Model Compilation
input_shape = (z_dim, y_dim, x_dim, channels)
model = unet_3d(input_shape)
model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coefficient])

# Model Training
history = model.fit(train_mri, train_mask, validation_data=(val_mri, val_mask), epochs=10, batch_size=2)

# Model Evaluation
evaluation = model.evaluate(val_mri, val_mask)
print(f"Validation loss: {evaluation[0]}, Validation Dice: {evaluation[1]}")