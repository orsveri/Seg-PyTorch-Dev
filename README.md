## Seg-PyTorch-Dev

This repository contains simple baseline for segmentation projects.

Structure of the repository:
- `[additional]`\
    Some scripts and tools to prepare certain datasets
- `[models]`\
    Segmentation models and functions for them (checkpoint saving, exporting, ..)\
    There will also be custom loss functions
- `[data]`\
    All for handling the data: pytorch datasets, samplers and functions for data processing
- `train.py`\
    Simple script for training 

#### How to use
1. Download dataset (works only with ADE20K for now) and run additional/ade20k_1csv_list.py, then additional/ade20k_2csv_sets.py. Don't forget to change parameters (paths) in these scripts!
2. Train with train.py.

There are no scripts for inference yet :( I need some time...

#### Datasets
- _ADE20K_ [http://sceneparsing.csail.mit.edu/, scene Parsing train/val data]\
Scene-centric indoor and outdoor images.
- _The Oxford-IIIT Pet_ [https://www.kaggle.com/devdgohil/the-oxfordiiit-pet-dataset]\
a 37 category pet dataset with roughly 100 images for each class created by the Visual Geometry Group at Oxford.
- [TODO] _Makeup. Pixel Perfect Lips Segmentation_ [https://www.kaggle.com/olekslu/makeup-lips-segmentation-28k-samples]\
The data was gathered and annotated with the custom semisupervised image annotation algorithm.
- [TODO] _Segmentation Full Body MADS Dataset_ [https://www.kaggle.com/tapakah68/segmentation-full-body-mads-dataset]\
The MADS dataset with segmented people, 1192 images. 



#### Notes and TODOs
- download weights for torchvision models and organize their loading to models
- reorganize train.py script - all input data and parameters should be taken from config file
- test UNet model and add some other variations

