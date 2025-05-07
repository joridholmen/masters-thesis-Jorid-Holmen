import os
import cv2
import xml.etree.ElementTree as ET
from annotation_converter.AnnotationConverter import AnnotationConverter
import glob
import os
import cv2
import matplotlib.pyplot as plt
import motmetrics as mm
import numpy as np
import pandas as pd


def MOT16_annotator(main_dir, output_dir, location, seq):

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"gt_creator_{seq}.txt")
    open(output_file, 'w').close()

    main_dir = os.path.join(main_dir, location, seq)

    #if os.path.isdir(full_path):  # Check if it's a folder
    print(f"Sequence folder: {main_dir}")
    current_annotation_file = f"{main_dir}/annotations.xml"
    current_image_dir = f"{main_dir}/imgs"

    tree = ET.parse(current_annotation_file)
    root = tree.getroot()

    # Get a sorted list of image filenames
    image_files = sorted(
        [img.attrib["name"] for img in root.findall("image")]
    )

    index_dict = {file_name: i+1 for i, file_name in enumerate(image_files)}

    xml_mapping = {img.attrib["name"]: img for img in root.findall("image")}

    for image_tag in root.findall("image"):
        file_name = image_tag.attrib["name"]
        src_image_path = os.path.join(current_image_dir, file_name)

        # Read and resize the image
        img = cv2.imread(src_image_path)

        img_width = int(float(image_tag.attrib["width"]))
        img_height = int(float(image_tag.attrib["height"]))

        src_image_path = os.path.join(current_image_dir, file_name)

        with open(output_file, "a") as label_file:

            annotation = AnnotationConverter.read_cvat_by_id(current_annotation_file, file_name)

            bbs = annotation.get_bounding_boxes()
            for bb in bbs:

                label = bb.get_label()
        
                # Calculate YOLO-format bounding box values
                bbox_x_min = int(bb.get_x())  # Top-left corner x
                bbox_y_min = int(bb.get_y())  # Top-left corner y
                bbox_width = int(bb.get_width())  # Width of the bounding box
                bbox_height = int(bb.get_height())  # Height of the bounding box
                
                bb_left = int(bb.get_x()) 
                bb_top = int(bb.get_y())   

                # Center coordinates
                center_x = bbox_x_min + (bbox_width / 2)
                center_y = bbox_y_min + (bbox_height / 2)

                # Normalize all values
                center_x_normalized = center_x / img_width
                center_y_normalized = center_y / img_height
                width_normalized = bbox_width / img_width
                height_normalized = bbox_height / img_height

                i = index_dict[file_name]

                # Ensure the values are within [0, 1]
                if (0 <= center_x_normalized <= 1 and 0 <= center_y_normalized <= 1 and 
                    0 <= width_normalized <= 1 and 0 <= height_normalized <= 1):
                    label_file.write(f"{i}, 0, {bbox_x_min}, {bbox_y_min}, {bbox_width}, {bbox_height}, 1, -1, -1, -1\n")
                else:
                    print(f"Skipping bounding box with out-of-bounds values: {center_x}, {center_y}, {bbox_width}, {bbox_height}")
    sort_txt_file(output_file)

def sort_txt_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Convert each line to a list of values and sort by the first column
    sorted_lines = sorted(lines, key=lambda x: int(x.split(',')[0]))

    # Write sorted data back to file
    with open(file_path, 'w') as f:
        f.writelines(sorted_lines)


def tracking_output2(track_results, location, seq):
    output_hyp_file = f"tracking_annotations/t/{location}/t_{seq}.txt"
    os.makedirs(os.path.dirname(output_hyp_file), exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    with open(output_hyp_file, 'w') as f:   
        for frame, result in enumerate(track_results):
            try:            
                for box in result.boxes:
                    x, y, w, h = box.xywh[0]
                    id = int(box.id)
                    conf = float(box.conf)

                    x = int(x - (w / 2))
                    y = int(y - (h / 2))

                    # Save in MOT16 format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                    f.write(f"{frame+1}, {id}, {x}, {y}, {int(w)}, {int(h)}, {conf:.6f}, -1, -1, -1\n")
            except:
                pass

def tracking_output(model, location, seq, video_path):
    # Load Video
    cap = cv2.VideoCapture(video_path)

    # Create output hypothesis file
    output_hyp_file = f"tracking_annotations/t/{location}/t_{seq}.txt"
    os.makedirs(os.path.dirname(output_hyp_file), exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    
    with open(output_hyp_file, 'w') as f:
        pass  # Clear existing file

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_id += 1
        results = model(frame)  # Run object detection
        
        with open(output_hyp_file, 'a') as f:
            for result in results:
                boxes = result.boxes.xywh.numpy()  # Get bbox [x, y, w, h]
                confs = result.boxes.conf.numpy()  # Confidence
                ids = np.arange(len(boxes))  # Assign IDs
                
                for obj_id, (x_center, y_center, w, h), conf in zip(ids, boxes, confs):
                    # Convert (x_center, y_center, w, h) → (bb_left, bb_top, w, h)
                    bb_left = x_center - (w / 2)
                    bb_top = y_center - (h / 2)

                    # Save in MOT16 format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
                    f.write(f"{frame_id}, {obj_id}, {bb_left:.2f}, {bb_top:.2f}, {w:.2f}, {h:.2f}, {conf:.6f}, -1, -1, -1\n")

    cap.release()
    print(f"Saved hypothesis results to {output_hyp_file}")



def motMetricsAccumulator(gtSource, tSource):
    gt = np.loadtxt(gtSource, delimiter=',')  # Load ground truth
    t = np.loadtxt(tSource, delimiter=',')    # Load predictions

    acc = mm.MOTAccumulator(auto_id=True)

    for frame in range(int(gt[:,0].max())):
        frame += 1  # Frames start from 1

        # Select detections for the current frame
        gt_dets = gt[gt[:,0] == frame, 2:6]  # Get X, Y, Width, Height
        t_dets = t[t[:,0] == frame, 2:6]  # Get X, Y, Width, Height
        
        if gt_dets.shape[0] == 0 or t_dets.shape[0] == 0:
            continue  # Skip if no detections in the frame

        # Compute IoU matrix
        C = mm.distances.iou_matrix(gt_dets.tolist(), t_dets.tolist(), max_iou=0.5)
        C = 1 - C

        # Update accumulator
        acc.update(
            range(len(gt_dets)),  # Ground truth IDs (indices)
            range(len(t_dets)),  # Tracker IDs (indices)
            C  # IoU cost matrix
        )

    mh = mm.metrics.create()

    return mh, acc


def compute_metrics(mh, acc, metrics, output_file):
    summary = mh.compute(acc, metrics=metrics, name='acc')

    strsummary = mm.io.render_summary(
        summary,
        #formatters={'mota' : '{:.2%}'.format},
        namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
                'precision': 'Prcn', 'num_objects': 'GT', \
                'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
                'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
                'num_misses': 'FN', 'num_switches' : 'IDsw', \
                'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP',  \
                }
    )
    
    return strsummary


def write_tracking_results(metric_res, output_file):
    summary = []
    columns = metric_res[0].split('\n')[0].split()

    with open(output_file + '.csv', 'w') as f:
        f.write('Sequence ' + ' '.join(columns) + '\n')

    for index, m in enumerate(metric_res):
        values = [float(i) for i in m.split('\n')[1].split()[1:]]
        summary.append(values)

    summary = np.array(summary)
    avg_MOTP = summary[:, 1].mean()
    avg_MOTA = summary[:, 0].mean()
    avg_IDF1 = summary[:, 2].mean()

    avg_metrics = [round(float(avg_MOTP), 3), round(float(avg_MOTA), 3), round(float(avg_IDF1), 3)]

    with open(output_file + '.csv', 'a') as f:
        for index, m in enumerate(summary):
            m = ' '.join(map(str, m))
            f.write(f'{index} {m}\n')

        f.write('\n')
        a = ' '.join(map(str, avg_metrics))
        f.write(f'Average {a}\n')

    fig, ax = plt.subplots(figsize=(4, 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=[avg_metrics], colLabels=columns, cellLoc='center', loc='center')

    for (i, j), cell in table.get_celld().items():
        cell.set_fontsize(13)
        cell.set_height(0.5)
    
    plt.savefig(f'{output_file}.png', bbox_inches='tight', dpi=300)

    


    



