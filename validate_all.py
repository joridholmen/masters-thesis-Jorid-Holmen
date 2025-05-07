import pandas as pd
import matplotlib.pyplot as plt
from utilities import *
from ultralytics import YOLO

## before transfer learning 
model = YOLO(f'weights/best.pt') # orion
output_dir = 'runs_eval'
clear_directories(output_dir)
data_dir = 'datasets'   # orion

# rumexweeds 
COCORumexWeeds = f'{data_dir}/rumex.yaml'
val_dir = 'valRumexWeeds'
val_results = model.val(data=COCORumexWeeds, split='test', project=output_dir, name=val_dir, exist_ok=True)
create_table(val_results, f'{output_dir}/{val_dir}/table.png', f'{output_dir}/{val_dir}/table.csv')

# crispus in box
COCOCrispusInBox = f'{data_dir}/CrispusInBox.yaml'
val_dir = 'valCrispusInBox'
val_results = model.val(data=COCOCrispusInBox, project=output_dir, name=val_dir, exist_ok=True)
create_table(val_results, f'{output_dir}/{val_dir}/table.png', f'{output_dir}/{val_dir}/table.csv')

# dronedata 
COCODronedata = f'{data_dir}/dronedata.yaml'
val_dir = 'valDronedata'
val_results = model.val(data=COCODronedata, project=output_dir, name=val_dir, exist_ok=True)
create_table(val_results, f'{output_dir}/{val_dir}/table.png', f'{output_dir}/{val_dir}/table.csv')

# lars olav 
LarsOlav = f'{data_dir}/larsolav'
predict_results = model.predict(source=LarsOlav, project=output_dir, name='predictLarsOlav', exist_ok=True, save=True)


## after transfer learning 
model = YOLO(f'runs/train/exp/weights/best.pt') # local
output_dir = 'runs_eval_all'
clear_directories(output_dir)

# rumexweeds
val_dir = 'valRumexWeeds'
val_results = model.val(data=COCORumexWeeds, project=output_dir, name=val_dir, exist_ok=True)
create_table(val_results, f'{output_dir}/{val_dir}/table.png', f'{output_dir}/{val_dir}/table.csv')

# crispus in box
val_dir = 'valCrispusInBox'
val_results = model.val(data=COCOCrispusInBox, project=output_dir, name=val_dir, exist_ok=True)
create_table(val_results, f'{output_dir}/{val_dir}/table.png', f'{output_dir}/{val_dir}/table.csv')

# dronedata
val_dir = 'valDronedata'
val_results = model.val(data=COCODronedata, project=output_dir, name=val_dir, exist_ok=True)
create_table(val_results, f'{output_dir}/{val_dir}/table.png', f'{output_dir}/{val_dir}/table.csv')

# lars olav 
predict_results = model.predict(source=LarsOlav, project=output_dir, name='predictLarsOlav', exist_ok=True, save=True)
