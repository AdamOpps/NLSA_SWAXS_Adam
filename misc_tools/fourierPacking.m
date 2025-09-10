function [normalizedFreq,fourierOrder]=fourierPacking(nfft)
% 
% [normalizedFreq,fourierOrder] = fourierPacking(nfft)
% 
% allows the results of FFTs to be organized and calibrated.
% 
% Input:
%   nfft: Number of Fourier coefficients.
% Output:
%   normalizedFreq: Normalized frequencies.
%   fourierOrder: Packing order of Fourier coefficients.
% Example call:
%   [normalizedFreq,fourierOrder] = fourierPacking(nFFT);
%   freq = normalizedFreq*1/(2*dt); % (unit of dt)^-1, f_max --> 1/(2*dt)
%   F = fft(y,nFFT);
%   F = F(fourierOrder);
% 
% copyright (c) Russell Fung 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  delta = [0:nfft-1];
  wrapAround = delta>nfft/2;
  delta(wrapAround) = delta(wrapAround)-nfft;
  [delta,fourierOrder] = sort(delta);
  normalizedFreq = delta/max(delta);
%end
