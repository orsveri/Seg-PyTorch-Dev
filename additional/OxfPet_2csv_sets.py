"""
The Oxford-IIIT Pet dataset, https://www.kaggle.com/devdgohil/the-oxfordiiit-pet-dataset

STEP 2 in a database preprocessing:
	splitting a full list of image and annotation files into two lists for train and validation/test
"""

import os
import numpy as np
import pandas as pd


inp_anno_csv = "/media/sveta/DATASTORE/AI_ML_DL/Datasets/5_TheOxford-IIITPet/anno_meta/anno_filelist.csv"
out_anno_sets_dir = "/media/sveta/DATASTORE/AI_ML_DL/Datasets/5_TheOxford-IIITPet/anno_meta/sets"
out_anno_sets_file = "anno_filelist_{}_{}.csv"
# columns = ["subdirs_img", "subdirs_seg", "filename_img", "filename_seg", "id", "species", "breed", "split_1",
# 			 "split_2", ..]

data = pd.read_csv(inp_anno_csv, index_col=0)

os.makedirs(out_anno_sets_dir, exist_ok=True)

split_cols = [col for col in list(data.columns) if col.startswith("split_")]
splits = [col.split("_")[-1] for col in split_cols if col.startswith("split_")]
for split in splits:
	data_train = data[data["split_{}".format(split)] == 'train'].reset_index(drop=True)
	data_test  = data[data["split_{}".format(split)] == 'test'].reset_index(drop=True)
	# drop unneeded columns
	data_train = data_train.drop(columns=split_cols)
	data_test  = data_test.drop(columns=split_cols)
	# save
	data_train.to_csv(os.path.join(out_anno_sets_dir, out_anno_sets_file.format('train', split)))
	data_test.to_csv(os.path.join(out_anno_sets_dir, out_anno_sets_file.format('test', split)))

print("Done!")