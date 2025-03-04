#!/usr/bin/python3
#
# CNN Training Script
#
# The purpose of this script is to perform the actual
# training of the model. To be interfaced with TrainMan.
#
# Gavin Tersteeg, 2024

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random

from tensorflow.keras import layers, models

def do_train_cnn(label_set, max_label, datapoint_width, datapoint_height):

    # Define input shape
    input_shape = (datapoint_height, datapoint_width, 1)
    
    # Start inputting datapoints from label directories
    dataset_images = []
    dataset_labels = []
    
    # Iterate through all labels
    for label in label_set:
    
        datapath = label[2]
        print("gathering data for label " + label[1])
        
        # Gather all files
        i = 0;
        for f in os.listdir(datapath):
            
            # Get datapoint path
            path = os.path.join(datapath, f)
            
            # Read the image
            datapoint_original = cv2.imread(path)
            
            # Convert it to grayscale
            datapoint_gray = cv2.cvtColor(datapoint_original, cv2.COLOR_BGR2GRAY) / 255.0
            
            # Check input shape
            datapoint_shape = np.shape(datapoint_gray)
            if input_shape[0] != datapoint_shape[0] or input_shape[1] != datapoint_shape[1]:
                print(path + " does not meet input size constraints")
                continue

            # Add it to the dataset
            dataset_images.append(datapoint_gray)
            dataset_labels.append([label[0]])
           
            i += 1
            
        print(str(i) + " datapoints gathered")
            
    # Done with gathering
    print(str(len(dataset_images)) + " datapoints gathered in total")
    print("datapoint gathering complete")
            
    # Shuffle labels and dataset
    seed = 1234
    random.Random(seed).shuffle(dataset_images)
    random.Random(seed).shuffle(dataset_labels)
    
    # Divide data between training set and testing set
    dataset_images = np.array(dataset_images)
    dataset_labels = np.array(dataset_labels)
    midpoint = int(len(dataset_images) * (2/3))
    train_images = dataset_images[:midpoint]
    train_labels = dataset_labels[:midpoint]
    test_images = dataset_images[midpoint:]
    test_labels = dataset_labels[midpoint:]
    print("train dataset size: " + str(len(train_images)))
    print("test dataset size: " + str(len(test_images)))
    
    # Create CNN model
    model = models.Sequential()
    
    # Define CNN shape here
    # This should really be defined elsewhere, but oh well
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    
    # Prepare to generate output
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation="relu"))
    
    # Create activation output
    model.add(layers.Dense(max_label+1, activation="softmax"))
    model.summary()
    
    # Compile model
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    # Start training
    epoch_cnt = 10
    history = model.fit(train_images, train_labels, epochs=epoch_cnt,  validation_data=(test_images, test_labels))
    
    # Training complete, display history
    # Adapted from Deep Learning with Python by Francois Chollet, 2018
    history_data = history.history
    
    # Get relevant data
    accuracy = history_data["accuracy"]
    loss = history_data["loss"]
    validation_accuracy = history_data["val_accuracy"]
    validation_loss = history_data["val_loss"]
    
    # Plot data
    epoch_range = range(1, epoch_cnt+1)
    figure, (axis1, axis2) = plt.subplots(1,2,figsize=(15,5))
    
    axis1.plot(epoch_range, loss, "bo", label="Training Loss")
    axis1.plot(epoch_range, validation_loss, "orange", label="Validation Loss")
    axis1.set_title("Training and Validation Loss")
    axis1.set_xlabel("Epochs")
    axis1.set_ylabel("Loss")
    axis1.legend()
    
    axis2.plot(epoch_range, accuracy, "bo", label="Training Accuracy")
    axis2.plot(epoch_range, validation_accuracy, "orange", label="Validation Accuracy")
    axis2.set_title("Training and Validation Accuracy")
    axis2.set_xlabel("Epochs")
    axis2.set_ylabel("Accuracy")
    axis2.legend()
    
    plt.show()