"""
The Oxford-IIIT Pet dataset, https://www.kaggle.com/devdgohil/the-oxfordiiit-pet-dataset

STEP 1 in a database preprocessing:
	making a list of all input image and annotation files and saving it as .csv file

As a result a .csv annotation file will be saved, with following columns:
["subdirs_img", "subdirs_seg", "filename_img", "filename_seg", "id", "species", "breed", "split_1", "split_2", ..]
Where number of columns "split_X" is n_splits.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


# -------------------------------------------------------------------------------------------------------------------- #
# Parameters and input data
# -------------------------------------------------------------------------------------------------------------------- #

rootdir = "/media/sveta/DATASTORE/AI_ML_DL/Datasets/5_TheOxford-IIITPet"
out_anno_subdir = "anno_meta"
out_anno_csv = "anno_filelist.csv"
# required format: {"train": {"images": "<..>", "masks": "<..>"}, "val": {"images": "<..>", "masks": "<..>"}}

images_dir = "images/images"
masks_dir  = "annotations/annotations/trimaps"
anno_file  = "annotations/annotations/list.txt"

columns = ["subdirs_img", "subdirs_seg", "filename_img", "filename_seg", "id", "species", "breed"] # + ["split_X", ..]
img_ext = ".jpg"
seg_ext = ".png"
seg_pre = "._"
n_splits = 5
test_fraction = 0.1 # from 0. to 1.

do_show_stats = False

# -------------------------------------------------------------------------------------------------------------------- #
# Make a list with information for future .csv table
# -------------------------------------------------------------------------------------------------------------------- #

data_list = []

subdir_img = os.path.join(rootdir, images_dir)
subdir_seg = os.path.join(rootdir, masks_dir)
annotations = pd.read_csv(os.path.join(rootdir, anno_file), sep=" ", header=None, skiprows=6)
annotations.columns = ["imgname", "id", "species", "breed_id"]

img_list = [i for i in os.listdir(subdir_img) if i.endswith(img_ext)]
L_img = len(img_list)
for i_im, img_name in enumerate(img_list):
	print("{}/{} \t processing.. ".format(i_im+1, L_img), end='')
	# check if there is corresponding annotation file
	seg_name = seg_pre + os.path.splitext(img_name)[0] + seg_ext
	if not os.path.isfile(os.path.join(subdir_seg, seg_name)):
		print("segmentation mask hasn't been found! Skip")
		continue

	# mask is here, now let's define a species (cat/dog) and a breed
	ok = False
	anno = annotations[annotations["imgname"] == os.path.splitext(img_name)[0]]
	if len(anno) == 1:
		anno = anno.iloc[0]
		id = anno["id"]
		species = anno["species"]
		breed_id = anno["breed_id"]
		ok = True
	elif len(anno) == 0:
		print("Annotation file doesn't contain this record! Image {}".format(img_name))
		print("Trying to find identificators..")
		# try to restore id, species, breed
		breed_name = img_name.split("_")[0]
		breed_rows = annotations[annotations["imgname"].str.split("_").str.get(0) == breed_name]
		id = pd.unique(breed_rows["id"]).tolist()
		species = pd.unique(breed_rows["species"]).tolist()
		breed_id = pd.unique(breed_rows["breed_id"]).tolist()
		if len(id) == len(species) == len(breed_id) == 1:
			id = id[0]
			species = species[0]
			breed_id = breed_id[0]
			print("successful.. ")
			ok = True
	else:
		print("\n\t\tAnnotation file contains more than one record with this file! Image {}".format(img_name))
		_ = input("\n\t\tPress Enter to continue..")

	if ok:
		data_list.append([images_dir, masks_dir, img_name, seg_name, id, species, breed_id])
		print('done!')
	else:
		print('this record was skipped')

# -------------------------------------------------------------------------------------------------------------------- #
# Make a pandas.DataFrame from the list
# -------------------------------------------------------------------------------------------------------------------- #

data_list = pd.DataFrame(data_list, columns=columns)

# show some statistics
if do_show_stats:
	print(data_list['species'].value_counts())
	data_list['id'].value_counts().sort_index(ascending=True).plot(kind='bar')
	plt.show()

# -------------------------------------------------------------------------------------------------------------------- #
# Create several splits (train-test) for cross-validation and add corresponding columns
# -------------------------------------------------------------------------------------------------------------------- #

skf = StratifiedShuffleSplit(n_splits=n_splits, train_size=1.-test_fraction, test_size=test_fraction)

for n, (train_index, test_index) in enumerate(skf.split(data_list["filename_img"], data_list["id"]), start=1):
	data_list.loc[train_index, 'split_{}'.format(n)] = 'train'
	data_list.loc[test_index, 'split_{}'.format(n)] = 'test'

#'''
# checking that splits are really balanced
temp = data_list[data_list["split_2"] == 'test']
print("len: {}".format(len(temp)))
temp['id'].value_counts().sort_index(ascending=True).plot(kind='bar')
plt.show()
#'''

# -------------------------------------------------------------------------------------------------------------------- #
# Save resulting
# -------------------------------------------------------------------------------------------------------------------- #

# save resulting annotation file
os.makedirs(os.path.join(rootdir, out_anno_subdir), exist_ok=True)
data_list.to_csv(os.path.join(rootdir, out_anno_subdir, out_anno_csv))

print('Everything is done!')


