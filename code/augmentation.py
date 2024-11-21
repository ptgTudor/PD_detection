# Dependencies and modules
import os

import numpy as np
import tensorflow as tf
import soundfile as sf

from audiomentations import Compose, AddGaussianSNR, RoomSimulator, TimeStretch, PitchShift, Shift

augment = Compose([
    AddGaussianSNR(min_snr_in_db=10, max_snr_in_db=20, p=0.5),
    RoomSimulator(min_size_x=3.6, max_size_x=5.6,
                  min_size_y=3.6, max_size_y=3.9,
                  min_size_z=2.4, max_size_z=3.0,
                  min_absorption_value=0.075, max_absorption_value=0.4,
                  min_target_rt60=0.15, max_target_rt60=0.8,
                  min_source_x=0.1, max_source_x=3.5,
                  min_source_y=0.1, max_source_y=2.7,
                  min_source_z=1.0, max_source_z=2.1,
                  min_mic_distance=0.15, max_mic_distance=0.35,
                  calculation_mode="absorption",
                  use_ray_tracing=True,
                  max_order=1,
                  p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
])

augment = Compose([
    AddGaussianSNR(min_snr_in_db=10, max_snr_in_db=20, p=0.5),
    RoomSimulator(min_size_x=3.6, max_size_x=5.6,
                  min_size_y=3.6, max_size_y=3.9,
                  min_size_z=2.4, max_size_z=3.0,
                  min_absorption_value=0.075, max_absorption_value=0.4,
                  min_target_rt60=0.15, max_target_rt60=0.8,
                  min_source_x=0.1, max_source_x=3.5,
                  min_source_y=0.1, max_source_y=2.7,
                  min_source_z=1.0, max_source_z=2.1,
                  min_mic_distance=0.15, max_mic_distance=0.35,
                  calculation_mode="absorption",
                  use_ray_tracing=True,
                  max_order=1,
                  p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
])

dir_hc = 'D:\programare\project\data\PC-GITA_per_task_44100Hz\DDK analysis\pataka\con normalizar\hc_output'
dir_hc_augmented = 'D:\programare\project\data\PC-GITA_per_task_44100Hz\DDK analysis\pataka\con normalizar\dataset_augmented\hc_augmented'
dir_pd = 'D:\programare\project\data\PC-GITA_per_task_44100Hz\DDK analysis\pataka\con normalizar\pd_output'
dir_pd_augmented = 'D:\programare\project\data\PC-GITA_per_task_44100Hz\DDK analysis\pataka\con normalizar\dataset_augmented\pd_augmented'

# List all audio files in the directories
audio_files_hc = [file for file in os.listdir(dir_hc) if file.endswith(".wav")]
audio_files_pd = [file for file in os.listdir(dir_pd) if file.endswith(".wav")]

# Loop through each audio file and apply augmentations
for audio_file in audio_files_hc:
    input_path_hc = os.path.join(dir_hc, audio_file)

    # Load the audio file using soundfile
    audio, sample_rate = sf.read(input_path_hc)

    # Apply augmentations using audiomentations
    augmented_audio = augment(samples=audio, sample_rate=sample_rate)

    # Modify the output file path to include "_augmented"
    output_file_name = os.path.splitext(audio_file)[0] + "_augmented_10.wav"
    output_path = os.path.join(dir_hc_augmented, output_file_name)

    # Save the augmented audio as a new file using soundfile
    sf.write(output_path, augmented_audio, sample_rate)

    print(f"Augmented {audio_file} and saved to {output_path}")

# Loop through each audio file and apply augmentations
for audio_file in audio_files_pd:
    input_path_pd = os.path.join(dir_pd, audio_file)

    # Load the audio file using soundfile
    audio, sample_rate = sf.read(input_path_pd)

    # Apply augmentations using audiomentations
    augmented_audio = augment(samples=audio, sample_rate=sample_rate)

    # Modify the output file path to include "_augmented"
    output_file_name = os.path.splitext(audio_file)[0] + "_augmented_10.wav"
    output_path = os.path.join(dir_pd_augmented, output_file_name)

    # Save the augmented audio as a new file using soundfile
    sf.write(output_path, augmented_audio, sample_rate)

    print(f"Augmented {audio_file} and saved to {output_path}")
