#!/usr/bin/python3
#
# Training Data Preprocessor
#
# Takes in the raw .WAV files, and turns it into
# something that can be classified
#
# Gavin Tersteeg, 2024

import sys
import librosa
import matplotlib.pyplot as plt
import numpy as np
import cv2

def preprocess(input, output, post_width, post_height):

    # Define some constants
    sample_rate = 44100
    segment_length = 10
    
    # We will assume that everything is 44100 sample rate
    signal, fs = librosa.load(input, sr=sample_rate)
    if fs != sample_rate:
        print("Sample rate does not match for " + input)
        return
    
    # Calculate samples in chunk
    signal_samples = len(signal)
    segment_samples = sample_rate * segment_length
    
    # Start generating plots
    i = 0;
    offset = 0;
    while True:
        
        # Check bounds on segment size
        if offset + segment_samples >= signal_samples:
            break;
        
        # Get segment
        segment = signal[offset:offset + segment_samples]
        
        # Constants
        n_fft = 4096
        hop_length = 2205
        n_mfcc = 128
        
        # # Extract features
        # mfcc = librosa.feature.mfcc(y=segment, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
        # spectrogram = librosa.feature.melspectrogram(y=segment, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mfcc)
        # spec_centroid = librosa.feature.spectral_centroid(y=segment, sr=sample_rate)
        # # spec_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sample_rate)
        
        # # Start fixing values to be placed into an image
        # mfcc *= 20.0
        # mfcc[mfcc<0] = 0
        # mfcc[mfcc>255] = 255
        
        # spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        # spectrogram *= -3.2
        # spectrogram[spectrogram<0] = 0
        # spectrogram[spectrogram>255] = 255
        
        
        # # Create output image
        # image = np.zeros((np.shape(mfcc)[0],np.shape(mfcc)[1],3), dtype=np.uint8)
        # image[:, :, 0] = mfcc
        # image[:, :, 1] = spectrogram
        
        # x = 0
        # for v in spec_centroid[0]:
            # v = int(min(v/64, 127))
            # sx = int((x / np.shape(spec_centroid)[1]) * np.shape(mfcc)[1])
            # image[v, sx, 2] = 255
            # x += 1

        # cv2.imwrite(output + "." + str(i) + ".png", image)
        
        # Set plot config
        plt.figure(figsize=(post_width / 100, post_height / 100), dpi=1000)
        plt.axis('off')
        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        
        # Create the spectrogram
        spectrogram = librosa.feature.melspectrogram(y=segment, sr=sample_rate, n_fft=4096, hop_length=2205, n_mels=512)
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max))

        # Save spectrogram
        plt.savefig(output + "." + str(i) + ".png", dpi=100, bbox_inches=None, pad_inches=0)
        plt.close()
        
        # Increment i and offset
        i += 1
        offset += segment_samples

def usage():

    print("\nusage: Preprocess.py [input] [output]")
    exit(1)

def main():

     # Parse arguments
    argc = len(sys.argv)
    
    # Are the args correct?
    if argc != 3:
        usage()
        
    # Do the preprocessing
    preprocess(sys.argv[1], sys.argv[2], 160, 120)

if __name__ == "__main__":
    main()