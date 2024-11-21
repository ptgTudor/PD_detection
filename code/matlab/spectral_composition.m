clear all;
close all;
clc;

% opening the file
file_1 = "F:\programare\project code\data\PC-GITA_per_task_44100Hz\DDK analysis\pataka\con normalizar\dataset_output\pd_output\AVPEPUDEA0005_pataka.wav"
file_2 = "F:\programare\project code\data\PC-GITA_per_task_44100Hz\DDK analysis\pataka\con normalizar\dataset_output\pd_output\AVPEPUDEA0005_pataka_augmented.wav"

[x, fs] = audioread(file_1);
[y, fs] = audioread(file_2);

figure
% signal as a function of time
subplot(2, 1, 1);
plot(x)
% signal as a function of frequency
subplot(2, 1, 2);
% spectrum(x, fs);
plot(y)