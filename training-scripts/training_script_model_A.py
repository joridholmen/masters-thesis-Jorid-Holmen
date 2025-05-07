from ultralytics import YOLO
from utilities import *
from utilities_tracking import * 

rumex_data = 'datasets/RumexWeeds.yaml'
dir_save = 'runs_A/train'
result_save = 'exp'
clear_directories('runs_A')

model = YOLO("yolo11s") # weights from model A

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data=rumex_data, 
                      epochs=150, 
                      imgsz=640, 
                      batch=8,                    # 60% GPU memory utilization
                      project=dir_save,
                      name=result_save,
                      val=True, 
                      augment=True, 
                      dfl=2,                        # focal loss 
                      exist_ok=True,
                      single_cls=True)


# Define variables for validating 
dir_save = 'runs_A/train'  # Path to the directory where the training results are saved
result_save = 'exp' 
area = '20210806_hegnstrup'     # Name of the area
model = YOLO(f"{dir_save}/{result_save}/weights/best.pt")   # Load the model

# Get results
val_results = model.val(data=rumex_data, split='test', project='runs_A/detect', name='val', exist_ok=True)
output_dir = 'final_res_A'
clear_directories(output_dir)
create_table(val_results, f'{output_dir}/detection_metrics.png', f'{output_dir}/detection_metrics.csv')
train_plots(f'{dir_save}/{result_save}', f'{output_dir}/train_results.png')
gather_curve_plots('runs_A/detect/val', output_dir)


# Tracking 
sequences = ['seq0', 'seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'seq6', 'seq7', 'seq8', 'seq9', 'seq10', 'seq11', 'seq12', 'seq13', 'seq14', 'seq15', 'seq16', 'seq17']
metric_res = []
latitudes_longitudes_obtus = []
latitudes_longitudes_crisp = []

for seq in sequences:
    video_path = f'videos/output_video_{area}_{seq}.mp4'
    track_results = model.track(source=video_path)

    tracking_output2(track_results, area, seq)

    gtSource = f'tracking_annotations/gt/{area}/gt_creator_{seq}.txt'
    tSource = f'tracking_annotations/t/{area}/t_{seq}.txt'

    mh, acc = motMetricsAccumulator(gtSource, tSource)

    summary = compute_metrics(mh, acc, ['mota', 'motp', 'idf1'], 'final_res_A/tracking_metrics')
    metric_res.append(summary)

    id_name_frame_dict = get_unique_tracking_ids_with_frames(track_results)
    latitudes, longitudes = get_gps_data_from_area('20210806_hegnstrup', 'datasets/RumexWeedsDataset', sequences=[seq])

    obtusifolius_latitudes, obtusifolius_longitudes, crispus_latitudes, crispus_longitudes =\
    get_gps_coordinates_for_detected_weeds(id_name_frame_dict, latitudes, longitudes)
    
    for olab, olob in zip(obtusifolius_latitudes, obtusifolius_longitudes):
        latitudes_longitudes_obtus.append((olab, olob))
    for clab, clob in zip(crispus_latitudes, crispus_longitudes):
        latitudes_longitudes_crisp.append((clab, clob))


write_tracking_results(metric_res, f'{output_dir}/tracking_metrics')

class_counts_based_on_area(area, latitudes_longitudes_obtus, latitudes_longitudes_crisp, f'{output_dir}/class_counts')

plot_gps_coordinates(latitudes_longitudes_obtus, latitudes_longitudes_crisp, f'{output_dir}/plot_map.png')

plot_gps_coordinates_to_map(latitudes_longitudes_obtus, latitudes_longitudes_crisp, f'{output_dir}/sat_map.html')



