"""
K-Nearest Neighbors (KNN) Classification for Genre Recognition
Sri Harsha Mudumba
Description: This script uses the KNN algorithm to classify audio files into genres based on their features extracted using librosa.
It also integrates with Spotify to classify previews of tracks.
"""

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import classification_report, confusion_matrix
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials  # Correct import statement
import re  # For extracting Spotify track ID from URL

# Basic configuration for logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# Spotify API setup
client_id = 'YOUR_CLIENT_ID'  # Replace with your Spotify client ID
client_secret = 'YOUR_CLIENT_SECRET'  # Replace with your Spotify client secret
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Define path to the dataset
dataset_path = r"S:\EDU\MS Computer Engineering\CPRE 575 Computaional Perception\Course project\GTZAN_dataset\Data"

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        extended_features = np.hstack([np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(contrast, axis=1)])
    except Exception as e:
        logging.error(f"Error encountered while parsing file: {file_path} - {str(e)}")
        return None
    return extended_features

def load_data(dataset_path):
    features, labels = [], []
    label_classes = {}
    label_index = 0
    for root, dirs, files in os.walk(dataset_path):
        for name in files:
            if name.endswith(".wav"):
                genre = os.path.basename(root)
                if genre not in label_classes:
                    label_classes[genre] = label_index
                    label_index += 1
                file_path = os.path.join(root, name)
                data = extract_features(file_path)
                if data is not None:
                    features.append(data)
                    labels.append(label_classes[genre])
    return np.array(features), np.array(labels), label_classes

features, labels, label_classes = load_data(dataset_path)
mean = np.mean(features, axis=0)
std = np.std(features, axis=0)
features = (features - mean) / std
np.random.seed(42)
indices = np.random.permutation(len(features))
train_idx, test_idx = indices[:int(0.7*len(features))], indices[int(0.7*len(features)):]
x_train, x_test = features[train_idx], features[test_idx]
y_train, y_test = labels[train_idx], labels[test_idx]

def knn_classify(x_train, y_train, x_test, k=5):
    y_pred = []
    for test_point in x_test:
        distances = np.sqrt(((x_train - test_point) ** 2).sum(axis=1))
        nearest_idx = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_idx]
        votes = np.bincount(nearest_labels)
        y_pred.append(votes.argmax())
    return np.array(y_pred)

k = 5
y_pred = knn_classify(x_train, y_train, x_test, k)
accuracy = np.mean(y_pred == y_test)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=[k for k, v in sorted(label_classes.items(), key=lambda item: item[1])]))

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, linewidths=.5)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(class_names, rotation=0)
    plt.show()

cm = confusion_matrix(y_test, y_pred)
class_names = [k for k, v in sorted(label_classes.items(), key=lambda item: item[1])]
plot_confusion_matrix(cm, class_names)

def fetch_and_classify_spotify_track(track_url):
    match = re.search(r'track/(\w+)', track_url)
    if match:
        track_id = match.group(1)
        track_info = sp.track(track_id)
        preview_url = track_info['preview_url']
        if preview_url:
            response = requests.get(preview_url)
            preview_file_path = 'spotify_preview.mp3'
            with open(preview_file_path, 'wb') as f:
                f.write(response.content)
            genre = classify_audio(preview_file_path, mean, std, label_classes)
            return genre
        else:
            return "No preview available for this track."
    else:
        return "Invalid Spotify URL"

def classify_audio(file_path, mean, std, label_classes):
    features = extract_features(file_path)
    if features is None:
        return "Feature extraction failed, check logs."
    features = (features - mean) / std
    features = features.reshape(1, -1)
    predicted_label = knn_classify(x_train, y_train, features, k)
    predicted_genre = [genre for genre, idx in label_classes.items() if idx == predicted_label[0]][0]
    return predicted_genre

# User interaction for Spotify track classification
spotify_track_url = input("Enter the full Spotify track URL to classify: ")
predicted_genre = fetch_and_classify_spotify_track(spotify_track_url)
print(f"The predicted genre of the Spotify track is: {predicted_genre}")
