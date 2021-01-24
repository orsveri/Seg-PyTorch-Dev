"""
ADE20K Dataset, http://sceneparsing.csail.mit.edu/ (scene Parsing train/val data)

STEP 2 in a database preprocessing:
	splitting a full list of image and annotation files into two lists for train and validation/test
"""

import pandas as pd


inp_anno_csv = "/media/sveta/DATASTORE/AI_ML_DL/Datasets/4_ADE20K/ADEChallengeData2016/anno_meta/anno_filelist.csv"
out_anno_csv = "/media/sveta/DATASTORE/AI_ML_DL/Datasets/4_ADE20K/ADEChallengeData2016/anno_meta/anno_filelist_{}.csv"
# columns = ["idx", "set", "subdirs_img", "subdirs_seg", "filename_img", "filename_seg", "train"]

data = pd.read_csv(inp_anno_csv, index_col=0)

data_val = data[data["train"] == False].reset_index(drop=True)
data_val.index.name = 'idx'
data_val.to_csv(out_anno_csv.format("val"))

data = data[data["train"] == True].reset_index(drop=True)
data.index.name = 'idx'
data.to_csv(out_anno_csv.format("train"))



