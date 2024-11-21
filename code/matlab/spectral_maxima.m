clear all;
close all;
% clc;

folder_hc = "F:\programare\project code\data\PC-GITA_per_task_44100Hz\_vowels\_normalized\A\dataset_output\hc_output";
folder_pd = "F:\programare\project code\data\PC-GITA_per_task_44100Hz\_vowels\_normalized\A\dataset_output\pd_output";

audio_files_hc = dir(fullfile(folder_hc, '*.wav'));
audio_files_pd = dir(fullfile(folder_pd, '*.wav'));

max_freqs_hc = zeros(length(audio_files_hc), 1);
max_freqs_pd = zeros(length(audio_files_pd), 1);

for k = 1 : length(audio_files_hc)
    [x, fs] = audioread(strcat(audio_files_hc(k).folder, '\', audio_files_hc(k).name));
    max_freqs_hc(k) = spectral_maximum(x, fs);

    [x, fs] = audioread(strcat(audio_files_pd(k).folder, '\', audio_files_pd(k).name));
    max_freqs_pd(k) = spectral_maximum(x, fs);
end

figure
subplot(2, 1, 1)
histogram(max_freqs_hc)
title('Fundamental frequency locations for the vowel "A" - HC')
subplot(2, 1, 2)
histogram(max_freqs_pd)
title('Fundamental frequency locations for the vowel "A" - PD')