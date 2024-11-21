import soundfile as sf
from audiomentations import Compose, AddGaussianSNR, RoomSimulator, TimeStretch, PitchShift

# Load the audio file
input_file = "F:\programare\project code\data\PC-GITA_per_task_44100Hz\_vowels\_normalized\A\dataset_output\pd_output\AVPEPUDEAP0001a1.wav"
output_file = "F:\programare\project code\data\PC-GITA_per_task_44100Hz\_vowels\_normalized\A\dataset_output\pd_output\AVPEPUDEAP0001a1_augmented.wav"
audio_data, sample_rate = sf.read(input_file)

# Create an augmentation pipeline with AddGaussianSNR
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

# Apply the augmentation
augmented_audio = augment(samples=audio_data, sample_rate=sample_rate)

# Save the augmented audio to a new file
sf.write(output_file, augmented_audio, sample_rate)

print(f"Augmented audio saved to {output_file}")
