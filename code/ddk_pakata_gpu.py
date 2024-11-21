# Dependencies and modules
import pathlib
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import splitfolders
import soundfile as sf
import random
import time
# import torch

tf.compat.v1.enable_eager_execution()

from tensorflow.keras import layers
from tensorflow.keras import models
from keras.callbacks import LearningRateScheduler
from IPython import display
from numba import cuda
from audiomentations import Compose, AddGaussianSNR, RoomSimulator, TimeStretch, PitchShift

# torch.cuda.empty_cache()

device = cuda.get_current_device()
device.reset()

# Using the GPU
gpu_device = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu_device, True)

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Empty the folders
# if os.path.isdir('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split/train'):
#     shutil.rmtree('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split/train')
    
# if os.path.isdir('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split/val'):
#     shutil.rmtree('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split/val')

# if os.path.isdir('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split/test'):
#     shutil.rmtree('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split/test')
    
# if os.path.isdir('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-15 split/train'):
#     shutil.rmtree('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-15 split/train')
    
# if os.path.isdir('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-50 split/val'):
#     shutil.rmtree('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-15 split/val')

# if os.path.isdir('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-15 split/test'):
#     shutil.rmtree('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-15 split/test')
    
if os.path.isdir('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split/train'):
    shutil.rmtree('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split/train')
    
if os.path.isdir('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split/val'):
    shutil.rmtree('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split/val')

if os.path.isdir('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split/test'):
    shutil.rmtree('F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split/test')

time.sleep(3)

# Importing the dataset
input_folder = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/dataset_output'

# output_folder = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split'
# output_folder = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-15 split'
output_folder = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split'

seed = random.randint(1, 1000)
print(seed)

# Splitting the files
splitfolders.ratio(input_folder, output=output_folder, seed=seed, ratio=(.80, .10, .10), group_prefix=None, move=False)

# DATASET_PATH_TRAIN = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split/train'
# DATASET_PATH_VAL = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split/val'
# DATASET_PATH_TEST = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split/test'

# DATASET_PATH_TRAIN = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-15 split/train'
# DATASET_PATH_VAL = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-15 split/val'
# DATASET_PATH_TEST = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-15 split/test'

DATASET_PATH_TRAIN = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split/train'
DATASET_PATH_VAL = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split/val'
DATASET_PATH_TEST = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split/test'

data_dir_train = pathlib.Path(DATASET_PATH_TRAIN)
data_dir_val = pathlib.Path(DATASET_PATH_VAL)
data_dir_test = pathlib.Path(DATASET_PATH_TEST)

commands = np.array(tf.io.gfile.listdir(str(data_dir_train)))
print('Commands:', commands)

augment = Compose([
    AddGaussianSNR(min_snr_in_db=30, max_snr_in_db=40, p=0.5),
    RoomSimulator(min_size_x=3.6, max_size_x=5.6,
                  min_size_y=3.6, max_size_y=3.9,
                  min_size_z=2.4, max_size_z=3.0,
                  min_absorption_value=0.3, max_absorption_value=0.4,
                  min_source_x=0.1, max_source_x=3.5,
                  min_source_y=0.1, max_source_y=2.7,
                  min_source_z=1.0, max_source_z=2.1,
                  min_mic_distance=0.05, max_mic_distance=0.1,
                  calculation_mode="absorption",
                  use_ray_tracing=True,
                  max_order=1,
                  leave_length_unchanged=True,
                  p=0.5),
    TimeStretch(min_rate=0.95, max_rate=1.05, leave_length_unchanged=True, p=0.5),
    PitchShift(min_semitones=-1, max_semitones=1, p=0.5),
])

# dir_hc_train = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split/train/hc_output'
# dir_pd_train = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split/train/pd_output'

# dir_hc_val = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split/val/hc_output'
# dir_pd_val = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split/val/pd_output'

# dir_hc_test = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split/test/hc_output'
# dir_pd_test = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/60-20-20 split/test/pd_output'

# dir_hc_train = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-15 split/train/hc_output'
# dir_pd_train = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-15 split/train/pd_output'

# dir_hc_val = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-15 split/val/hc_output'
# dir_pd_val = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-15 split/val/pd_output'

# dir_hc_test = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-15 split/test/hc_output'
# dir_pd_test = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/70-15-15 split/test/pd_output'

dir_hc_train = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split/train/hc_output'
dir_pd_train = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split/train/pd_output'

dir_hc_val = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split/val/hc_output'
dir_pd_val = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split/val/pd_output'

dir_hc_test = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split/test/hc_output'
dir_pd_test = 'F:/programare/project code/data/PC-GITA_per_task_44100Hz/DDK analysis/pakata/con normalizar/80-10-10 split/test/pd_output'

# List all audio files in the directories
audio_files_hc_train = [file for file in os.listdir(dir_hc_train) if file.endswith(".wav")]
audio_files_pd_train = [file for file in os.listdir(dir_pd_train) if file.endswith(".wav")]

audio_files_hc_val = [file for file in os.listdir(dir_hc_val) if file.endswith(".wav")]
audio_files_pd_val = [file for file in os.listdir(dir_pd_val) if file.endswith(".wav")]

audio_files_hc_test = [file for file in os.listdir(dir_hc_test) if file.endswith(".wav")]
audio_files_pd_test = [file for file in os.listdir(dir_pd_test) if file.endswith(".wav")]

for i in range(9):
    
    # TRAINING
    # Loop through each audio file and apply augmentations
    for audio_file in audio_files_hc_train:
        input_path_hc_train = os.path.join(dir_hc_train, audio_file)

        # Load the audio file using soundfile
        audio, sample_rate = sf.read(input_path_hc_train)

        # Apply augmentations using audiomentations
        augmented_audio = augment(samples=audio, sample_rate=sample_rate)

        # Modify the output file path to include "_augmented"
        output_file_name = os.path.splitext(audio_file)[0] + "_augmented_" + str(i + 1) + ".wav"
        output_path = os.path.join(dir_hc_train, output_file_name)

        # Save the augmented audio as a new file using soundfile
        sf.write(output_path, augmented_audio, sample_rate)

        print(f"Augmented {audio_file} and saved to {output_path}")

        # Loop through each audio file and apply augmentations
    for audio_file in audio_files_pd_train:
        input_path_pd_train = os.path.join(dir_pd_train, audio_file)

        # Load the audio file using soundfile
        audio, sample_rate = sf.read(input_path_pd_train)

        # Apply augmentations using audiomentations
        augmented_audio = augment(samples=audio, sample_rate=sample_rate)

        # Modify the output file path to include "_augmented"
        output_file_name = os.path.splitext(audio_file)[0] + "_augmented_" + str(i + 1) + ".wav"
        output_path = os.path.join(dir_pd_train, output_file_name)

        # Save the augmented audio as a new file using soundfile
        sf.write(output_path, augmented_audio, sample_rate)

        print(f"Augmented {audio_file} and saved to {output_path}")

# Get sequences of 1 second (16000 since the samples are at 16 kHz frequency)
train_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir_train,
    batch_size=32,
    seed=0,
    output_sequence_length=16000)

val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir_val,
    batch_size=32,
    seed=0,
    output_sequence_length=16000)

test_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir_test,
    batch_size=32,
    seed=0,
    output_sequence_length=16000)

label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)

train_ds.element_spec

for example_audio, example_labels in train_ds.take(1):
  print(example_audio.shape)
  print(example_labels.shape)

# Drop the extra axis
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)
test_ds = test_ds.map(squeeze, tf.data.AUTOTUNE)

for example_audio, example_labels in train_ds.take(1):
  print(example_audio.shape)
  print(example_labels.shape)

# Plotting audio waveforms
label_names[[0, 1]]

n = 4
fig, axes = plt.subplots(n, figsize=(16, 10))

for i in range(n):
  if i >= n:
    break
  ax = axes[i]
  ax.plot(example_audio[i].numpy())
  ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
  label = label_names[example_labels[i]]
  ax.set_title(label)
  ax.set_ylim([-1.1, 1.1])
  ax.set_xlim([0, len(example_audio[i])])

plt.show()

# Converting waveforms to spectograms
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  stfts = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(stfts)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

# Printing the shapes of the tensorized waveform and its corresponding spectrogram
for i in range(2):
  label = label_names[example_labels[i]]
  waveform = example_audio[i]
  spectrogram = get_spectrogram(waveform)

  print('Label:', label)
  print('Waveform shape:', waveform.shape)
  print('Spectrogram shape:', spectrogram.shape)
  print('Audio playback')
  display.display(display.Audio(waveform, rate=16000))

# Function for displaying a spectrogram
def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

# Plotting the example waveform over time and its corresponding spectrogram (frequencies over time)
fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.suptitle(label.title())
plt.show()

# Printing the shapes of the tensorized waveform and its corresponding spectrogram
for i in range(4):
  label = label_names[example_labels[i]]
  waveform = example_audio[i]
  spectrogram = get_spectrogram(waveform)

  print('Label:', label)
  print('Waveform shape:', waveform.shape)
  print('Spectrogram shape:', spectrogram.shape)
  print('Audio playback')
  display.display(display.Audio(waveform, rate=16000))

# Plotting the example waveform over time and its corresponding spectrogram (frequencies over time)
fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 16000])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.suptitle(label.title())
plt.show()

# Creating spectrogram datasets from audio datasets
def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio, label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

# Examining different spectrograms
train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break

n = 4
fig, axes = plt.subplots(n, figsize=(16, 9))

for i in range(n):

    ax = axes[i]
    plot_spectrogram(example_spectrograms[i].numpy(), ax)
    ax.set_title(label_names[example_spect_labels[i].numpy()])

plt.show()

# Building and training the model
train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
# train_spectrogram_ds = train_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

with tf.device('/GPU:0'):

    model = models.Sequential([
        
        layers.Input(shape=input_shape),

        norm_layer,
        
        layers.Conv2D(32, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        
        layers.Conv2D(64, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        
        layers.Conv2D(128, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        
        layers.Conv2D(256, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),

        layers.Flatten(),
        layers.Dropout(0.5),
        
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(0.5),
        
        layers.Dense(num_labels),

    ])

model.summary()

# def lr_scheduler(epoch, lr):
#     decay_rate = 0.1
#     decay_step = 20
#     if epoch % decay_step == 0 and epoch:
#         return lr * decay_rate
#     return lr
# callbacks = [LearningRateScheduler(lr_scheduler, verbose=1)]

# initial_learning_rate = 0.001
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=500,
#     decay_rate=0.1,
#     staircase=True)

# Configuring the Keras model with the Adam optimizer and the cross-entropy loss     
model.compile(  
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),   
    metrics=['accuracy'],
)

EPOCHS = 100

# Training the model over a number of epochs
with tf.device('/GPU:0'):

    history = model.fit(
        train_spectrogram_ds,   
        validation_data=val_spectrogram_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=100),
        # callbacks=callbacks

    )
    
model.save('pakata.h5')

# Plotting the training and validation loss curves
metrics = history.history
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')

plt.subplot(1, 2, 2)
plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')

# Run the model on the test set and check its performance
model.evaluate(test_spectrogram_ds, return_dict=True)

with tf.device('/GPU:0'):
    # Displaying a confusion matrix
    y_pred = model.predict(test_spectrogram_ds)
    
y_pred = tf.argmax(y_pred, axis=1)
y_true = tf.concat(list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0)

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=label_names,
            yticklabels=label_names, annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

true_positives = confusion_mtx[1, 1]
false_positives = confusion_mtx[0, 1]
true_negatives = confusion_mtx[0, 0]
false_negatives = confusion_mtx[1, 0]

print(f"True Positives: {true_positives}")
print(f"False Positives: {false_positives}")
print(f"True Negatives: {true_negatives}")
print(f"False Negatives: {false_negatives}")

accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * precision * recall / (precision + recall)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")

