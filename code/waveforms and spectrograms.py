import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

input_file_path = "F:\programare\project code\data\PC-GITA_per_task_44100Hz\DDK analysis\pataka\con normalizar\dataset_output\hc_output\AVPEPUDEAC0001_pataka.wav"
# Read the audio file
audio_data, sample_rate = sf.read(input_file_path)

def get_spectrogram(waveform):
  stfts = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(stfts)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

# Create a plot
fig_1, ax_1 = plt.subplots(figsize=(16, 10))

# Plot the audio data
ax_1.plot(audio_data)
ax_1.set_yticks(np.arange(-1.2, 1.2, 0.2))
label = "PD (pataka)"
ax_1.set_title(label)
ax_1.set_ylim([-1.1, 1.1])
ax_1.set_xlim([0, 16000])

audio_data = audio_data[:16000]

spectrogram = get_spectrogram(audio_data)

# Create a plot
fig_2, ax_2 = plt.subplots(figsize=(16, 10))

plot_spectrogram(spectrogram.numpy(), ax_2)
ax_2.set_title("PD (pataka)")
plt.show()

# Display the plot
plt.show()
