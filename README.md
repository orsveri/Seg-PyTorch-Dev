## Seg-PyTorch-Dev

This repository contains simple baseline for segmentation projects.

Structure of the repository:
- _[additional]_\
    Some scripts and tools to prepare certain datasets
- _[models]_\
    Segmentation models and functions for them (checkpoint saving, exporting, ..)\
    There will also be custom loss functions
- _[data]_\
    All for handling the data: pytorch datasets, samplers and functions for data processing
- _train.py_\
    Simple script for training 
    
#### Notes and TODOs
- max 255 clsses\
    What to do if we have more?
- test UNet model and add other variations (test on some simple dataset with 1-5 classes)
- add another models from torchvision
- download weights for torchvision models and organize their loading to models
    

