import numpy as np
import soundfile as sf
from audiomentations import Compose, TimeStretch
import matplotlib.pyplot as plt

# Load an audio file
audio, sample_rate = sf.read("F:\programare\project code\data\PC-GITA_per_task_44100Hz\_vowels\_normalized\A\dataset_output\hc_output\AVPEPUDEAC0006a3.wav")

# Get the original length of the audio
original_length = len(audio)

# Create an augmenter with TimeStretch
augmenter = Compose([
    TimeStretch(min_rate=0.8, max_rate=0.8, leave_length_unchanged=True)
])

# Apply the augmentation
augmented_audio = augmenter(samples=audio, sample_rate=sample_rate)

# Check the lengths
print(f"Original Length: {original_length}")
print(f"Augmented Length: {len(augmented_audio)}")

# Compare the end of the original and augmented audio
original_end = audio[-500:]  # Last 500 samples of the original audio
augmented_end = augmented_audio[-500:]  # Last 500 samples of the augmented audio

# Plot the end portions to visually inspect the padding
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(original_end)
plt.title('Original Audio End')

plt.subplot(2, 1, 2)
plt.plot(augmented_end)
plt.title('Augmented Audio End')

plt.tight_layout()
plt.show()
