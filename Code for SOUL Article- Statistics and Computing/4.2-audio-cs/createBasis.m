%% This file generates a basis from piano note frequencies (plus a few
% more). We will use this as the basis for our songs during l1-minimization routines.
clear all; clc;

fs=44100;           % Sampling frequency
t = 1/fs:1/fs:0.25; % Time points of each sample in a quarter-second note
T = length(t);      % Number of samples in a note
M = 29;             % Number of notes in Mary Had a Little Lamb.

% The frequencies for piano notes can be derived from formula.
% I simply downloaded them from Professor Bryan Suits at Michigan Tech.
% http://www.phy.mtu.edu/~suits/notefreqs.html

allfreqs = [16.35 17.32 18.35 19.45 20.6 21.83 23.12 24.5 25.96 27.5 29.14 30.87 32.7 34.65 ...
36.71 38.89 41.2 43.65 46.25 49 51.91 55 58.27 61.74 65.41 69.3 73.42 77.78 82.41 87.31 92.5 ...
98 103.83 110 116.54 123.47 130.81 138.59 146.83 155.56 164.81 174.61 185 196 207.65 220 233.08 ...
246.94 261.63 277.18 293.66 311.13 329.63 349.23 369.99 392 415.3 440 466.16 493.88 523.25 ...
554.37 587.33 622.25 659.26 698.46 739.99 783.99 830.61 880 932.33 987.77 1046.5 1108.73 ...
1174.66 1244.51 1318.51 1396.91 1479.98 1567.98 1661.22 1760 1864.66 1975.53 2093 2217.46 ...
2349.32 2489.02 2637.02 2793.83 2959.96 3135.96 3322.44 3520 3729.31 3951.07 4186.01 4434.92 ...
4698.64 4978.03];

%% First let's generate the scale so we can hear what the notes will sound
% like.

xall=[]; % This vector will hold all our notes.

% I multiply the sine waves by exponentials to get a "ding" sound:
df = 15; % This is the rate of decay of the note.
sf = 15; % This is the rate of growth at the beginning of the note.

for frequency = allfreqs
    n1 = (1+exp(-df*t(end))-exp(-sf*t)).*exp(-df*t).*sin(2*pi*frequency*t);
    xall = [xall n1];
end
% Play the whole scale to hear what it sounds like:
sound(xall,fs);


%% Now let's create a basis that we will use to reconstruct songs.
% All notes will be a quarter-second long, and our basis will handle 
% 29-note songs. So we simply need to replicate each note in each position 
% in the song.

% Since there will be a lot of zeros, we will use the sparse matrix format 
% for efficient computation.

I=zeros(1,length(t)*length(allfreqs)*M); % Row indices
J=zeros(1,length(t)*length(allfreqs)*M); % Column indices
S=zeros(1,length(t)*length(allfreqs)*M); % The matrix value at that index.
inds=1:length(t);
biter=1;
for fiter=1:length(allfreqs)
    for k=1:M
        I(inds) = [ones(1,length(t))*biter];
        J(inds) = [((k-1)*T+1):k*T];
        S(inds) = [(1+exp(-df*t(end))-exp(-sf*t)).*exp(-df*t).*sin(2*pi*allfreqs(fiter)*t)];
        biter=biter+1;
        inds = inds+length(t);
    end
end
tic
pianoBasis = sparse(I,J,S,length(allfreqs)*M,M*T);
toc

save pianoBasis.mat pianoBasis fs t M allfreqs
