from utilities import *

print('Starting')
output_dir = 'datasets/coco_docks_stratify/yolo'
clear_directories(output_dir)
print('directory cleared')
location_folders = ['20210806_hegnstrup', '20210806_stengard', '20210807_lundholm', '20211006_stengard', '20210908_lundholm']
parse_cvat_annotation('datasets/RumexWeedsDataset', location_folders, output_dir)
print('starting to split')
train_test_split_stratify(output_dir, 0.7, 0.1, 0.2)
