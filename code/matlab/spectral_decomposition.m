clear all;
close all;
clc;

% opening the file
file = "F:\programare\project code\data\PC-GITA_per_task_44100Hz\_vowels\_normalized\A\dataset_output\hc_output\AVPEPUDEAC0001a1.wav";

[x, fs] = audioread(file);

figure
% % signal as a function of time
% subplot(2, 1, 1);
% plot(x)
% % signal as a function of frequency
% subplot(2, 1, 2);
spectrum(x, fs);

first = bandpass(x, [1050, 1150], fs, Steepness=0.99);
audiowrite('hc/first.wav', first, fs);
% figure
% spectrum(first, fs);

% second = bandpass(x, [300, 500], fs, Steepness=0.99);
% audiowrite('hc/second.wav', second, fs);
% % figure
% % spectrum(second, fs);
% 
% third = bandpass(x, [500, 725], fs, Steepness=0.99);
% audiowrite('hc/third.wav', third, fs);
% % figure
% % spectrum(third, fs);
% 
% fourth = bandpass(x, [725, 950], fs, Steepness=0.99);
% audiowrite('hc/fourth.wav', fourth, fs);
% % figure
% % spectrum(fourth, fs);
% 
% fifth = bandpass(x, [950, 1175], fs, Steepness=0.99);
% audiowrite('hc/fifth.wav', fifth, fs);
% % figure
% % spectrum(fifth, fs);
% 
% sixth = bandpass(x, [1175, 1500], fs, Steepness=0.99);
% audiowrite('hc/sixth.wav', sixth, fs);
% % figure
% % spectrum(sixth, fs);