from ultralytics import YOLO
from utilities import *

BoxDroneRumex = 'datasets/BoxDroneRumex.yaml'
model = YOLO('weights/best.pt')
output_dir = 'runs_transfer_learning'
clear_directories(f'{output_dir}/train')
clear_directories(f'{output_dir}/val')
clear_directories(f'{output_dir}/final_res')
dir_save = f'{output_dir}/train'
result_save = 'exp'

results = model.train(data=BoxDroneRumex,
                      epochs=150,
                      imgsz=640,
                      batch=16, 
                      project=dir_save,
                      name=result_save,
                      val=True,
                      augment=True,
                      dfl=2,                        # focal loss 
                      exist_ok=True 
                      )

val_results = model.val(data=BoxDroneRumex, split='test', project=output_dir, name='val', exist_ok=True)

create_table(val_results, f'{output_dir}/final_res/detection_metrics.png', f'{output_dir}/final_res/detection_metrics.csv')
train_plots(f'{output_dir}/train/exp', f'{output_dir}/final_res/train_results.png')
gather_curve_plots(f'{output_dir}/val', f'{output_dir}/final_res')
