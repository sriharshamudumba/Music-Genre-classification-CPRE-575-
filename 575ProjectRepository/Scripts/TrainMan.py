#!/usr/bin/python3
#
# Training Manager Script
#
# The purpose of this script is to automatically
# manage labelled input sources, preprocessing
# said sources, and dispatching training scripts
# on the post-processed labelled inputs.
#
# Gavin Tersteeg, 2024

import json
import os
import sys
import shutil

from Preprocess import preprocess
from TrainerCNN import do_train_cnn

CFG_PATH = "../config.json"
CFG_TEMPLATE = "../Resources/config_template.json"

# Attempts to load up config.json
# Will populate with an blank JSON file if it is missing
def loadConfigurationFile():
    global config
    
    print("loading configuration...")
    
    # Make sure the file exists
    if not os.path.isfile(CFG_PATH):
        print("WARNING: configuration file not found!")
        
        # It is a directory maybe?
        if os.path.exists(CFG_PATH):
            print("directory exists at file location, bailing out!")
            exit(1)
            
        # Hopefully the template exists, at least
        if not os.path.isfile(CFG_TEMPLATE):
            print("can't find template file, bailing out!")
            exit(1)
        
        # Attempt to copy template over
        print("creating template configration...")
        try:
            shutil.copyfile(CFG_TEMPLATE, CFG_PATH)
        except:
            print("can't copy template file, bailing out!")
            exit(1)
        
    # Load config
    with open(CFG_PATH, 'r') as config_file:
    
        print("reading config file...")
        config = json.loads(config_file.read())
        
    # Done
    print("configuration load complete")
    
# Checks to see if the post process path exists
# If not, attempts to create it
def hasPostProcessPath():

    # Get post-processing path
    postPath = config['postProcessPath']
    print("checking path " + postPath)
    
    # Does it exist
    if not os.path.isdir(postPath):
        
        print("WARNING: " + postPath + " does not exist, attempting to create")
        
        # Make the path
        try:
            os.makedirs(postPath)
        except:
            print("exception during path create")
            
        # Did it work?
        if not os.path.isdir(postPath):
        
            print("failed to create post-processing path, bailing out!")
            exit(1)
    
def clean():
    
    print("cleaning training data...")
    
    # Check if the directory exists
    hasPostProcessPath()
    
    # Remove everything
    for f in os.listdir(config['postProcessPath']):
        path = os.path.abspath(os.path.join(config['postProcessPath'], f));
        
        if os.path.isdir(path):
            shutil.rmtree(path)
    
    
def preproc():
    
    # First, we do a clean
    clean()
    
    print("starting data preprocessing...")
    
    # Get width and height
    post_width = config['dataPointWidth']
    post_height = config['dataPointHeight']
    
    # Iterate over all labels
    for label in config['labels']:
    
        # Get label characteristics
        name = label['name']
        enabled = label['isEnabled']
        sources = label['sources']
        
        # Do not process if it is not enabled
        if not enabled:
            continue
            
        # Create directory for source
        dirpath = os.path.join(config['postProcessPath'], name)
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)
        
        # Iterate over all sources, and all files in said source
        i = 0
        for source in sources:
        
            for f in os.listdir(source):
                path = os.path.join(source, f)
                postpath = os.path.join(dirpath, str(i) + "-" + f)
                
                # Start preprocessing
                print("preprocessing " + path + " to " + postpath)
                
                try:
                    preprocess(path, postpath, post_width, post_height)
                except:
                    print("WARNING: could not preprocess " + path)
                
            i += 1
    
def train(model_name):
    
    print("gathering labels...")
    
    # Get width and height
    data_width = config['dataPointWidth']
    data_height = config['dataPointHeight']
    
    # Iterate over all labels
    label_set = []
    max_label = 0
    for label in config['labels']:
    
        # Make sure that the label is enabled
        enabled = label['isEnabled']
        if not enabled:
            continue
            
        # Get label characteristics
        name = label['name']
        label_id = label['labelNum']
        max_label = max(max_label, label_id)  
          
        # Get directory of post-processed training data
        dirpath = os.path.join(config['postProcessPath'], name)
        
        # Create entry
        entry = [label_id, name, dirpath]
        label_set.append(entry)
        
    if model_name == "cnn":
        do_train_cnn(label_set, max_label, data_width, data_height)
    else:
        print("unknown model type, bailing out!")
        exit(1)
    
def usage():

    print("\nusage: TrainMan.py [operation] [args]\n")
    print("valid operations:")
    print("\tclean: removes all preprocessed training data")
    print("\tpreproc: cleans out and preprocesses all training data")
    print("\ttrain [model type]: starts or resumes training on a model")
    exit(1)

def main():
    
    # Start by loading the configuration file
    loadConfigurationFile()
    
    # Parse arguments
    argc = len(sys.argv)
    
    # Do we have a command?
    if argc <= 1:
        usage()
        
    # Get the command and operate
    command = sys.argv[1].lower()
    if command == "clean":
    
        # Check arg count
        if argc != 2:
            usage()
        
        # Do a clean
        clean()
        
    elif command == "preproc":
    
        # Check arg count
        if argc != 2:
            usage()
    
        # Preprocess all training data
        preproc()
    
    elif command == "train":
    
        # Check arg count
        if argc != 3:
            usage()
    
        # Start or resume training
        train(sys.argv[2].lower())
        
    else:
    
        # Unknown command
        usage()
        
    print("operation complete")

if __name__ == "__main__":
    main()