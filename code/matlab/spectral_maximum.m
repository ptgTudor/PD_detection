function maximum = spectral_maximum(signal, fs)

Y = fft(signal); % Fourier transform

L = length(Y); % number of samples

P2 = abs(Y / L); % the magnitude of the frequency components
P1 = P2(1 : fix(L/2) + 1); % keep only the first half
                           % for a single-sided spectrum

P1(2 : end - 1) = 2 * P1(2 : end - 1); % multiply non-DC components
                                       % by two to account for discarding
                                       % the double-sided spectrum

f = fs * (0 : (L/2)) / L; % compute the frequencies based on the sampling
                          % frequency and the number of samples

max_amplitude = max(P1);
I = find(P1 == max_amplitude);
maximum = f(I);

end