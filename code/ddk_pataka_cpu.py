# Dependencies and modules
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from keras.callbacks import LearningRateScheduler
from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Importing the dataset
DATASET_PATH = 'F:\programare\project\data\PC-GITA_per_task_44100Hz\DDK analysis\pataka\con normalizar\dataset_augmented'
data_dir = pathlib.Path(DATASET_PATH)
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
print('Commands:', commands)

# Get sequences of 3 seconds (48000 since the samples are at 16000 Hz frequency)
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=256,
    validation_split=0.4,
    seed=0,
    output_sequence_length=48000,
    subset='both')

label_names = np.array(train_ds.class_names)
print()
print("label names:", label_names)

train_ds.element_spec

# Drop the extra axis
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels


train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# Split the validation into two halves
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

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
train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(
    10000).prefetch(tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=train_spectrogram_ds.map(
    map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    # layers.Resizing(32, 32),
    # Normalize.
    norm_layer,

    # layers.Conv2D(32, 3, activation='relu'),
    # layers.Conv2D(64, 3, activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Dropout(0.25),
    # layers.Flatten(),
    # layers.Dense(128, activation='relu'),
    # layers.Dropout(0.5),
    # layers.Dense(num_labels),

    # layers.Conv2D(32, 3, activation='relu', padding='same'),
    # layers.Conv2D(32, 3, activation='relu', padding='same'),
    # layers.MaxPooling2D(),
    # layers.Dropout(0.25),
    # layers.Conv2D(64, 3, activation='relu', padding='same'),
    # layers.Conv2D(64, 3, activation='relu', padding='same'),
    # layers.MaxPooling2D(),
    # layers.Dropout(0.25),
    # layers.Conv2D(128, 3, activation='relu', padding='same'),
    # layers.Conv2D(128, 3, activation='relu', padding='same'),
    # layers.MaxPooling2D(),
    # layers.Dropout(0.25),
    # layers.Flatten(),
    # layers.Dense(256, activation='relu'),
    # layers.Dropout(0.5),
    # layers.Dense(128, activation='relu'),
    # layers.Dropout(0.5),
    # layers.Dense(num_labels, activation='softmax'),

    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(num_labels),

])

model.summary()

def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 20
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

callbacks = [LearningRateScheduler(lr_scheduler, verbose=1)]

# Configuring the Keras model with the Adam optimizer and the cross-entropy loss
model.compile(  
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),   
    metrics=['accuracy'],
)

# Training the model over a number of epochs
EPOCHS = 80 
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

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
plt.plot(history.epoch, 100 *
         np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
plt.legend(['accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')

# Run the model on the test set and check its performance
model.evaluate(test_spectrogram_ds, return_dict=True)

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

# Run interference on an audio file
x = data_dir/'F:\programare\project\data\PC-GITA_per_task_44100Hz\DDK analysis\pataka\con normalizar\dataset_augmented\hc_augmented\AVPEPUDEAC0001_pataka.wav'
x = tf.io.read_file(str(x))
x, sample_rate = tf.audio.decode_wav(
    x, desired_channels=1, desired_samples=48000,)
x = tf.squeeze(x, axis=-1)
waveform = x
x = get_spectrogram(x)
x = x[tf.newaxis, ...]

prediction = model(x)
x_labels = ['HC', 'PD']
plt.bar(x_labels, tf.nn.softmax(prediction[0]))
plt.title('HC')
plt.show()

display.display(display.Audio(waveform, rate=16000))

# Exporting the model with preprocessing
class ExportModel(tf.Module):
  def __init__(self, model):
    self.model = model

    # Accept either a string-filename or a batch of waveforms.
    # YOu could add additional signatures for a single wave, or a ragged-batch.
    self.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=(), dtype=tf.string))
    self.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=[None, 48000], dtype=tf.float32))

  @tf.function
  def __call__(self, x):
    # If they pass a string, load the file and decode it.
    if x.dtype == tf.string:
      x = tf.io.read_file(x)
      x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=48000,)
      x = tf.squeeze(x, axis=-1)
      x = x[tf.newaxis, :]

    x = get_spectrogram(x)
    result = self.model(x, training=False)

    class_ids = tf.argmax(result, axis=-1)
    class_names = tf.gather(label_names, class_ids)
    return {'predictions': result,
            'class_ids': class_ids,
            'class_names': class_names}

# Testing the export model
export = ExportModel(model)
export(tf.constant(str(data_dir/'F:\programare\project\data\PC-GITA_per_task_44100Hz\DDK analysis\pataka\con normalizar\dataset_augmented\hc_augmented\AVPEPUDEAC0001_pataka.wav')))

# Save and reload the model, expecting identical output
tf.saved_model.save(export, "saved")
imported = tf.saved_model.load("saved")
imported(waveform[tf.newaxis, :])
