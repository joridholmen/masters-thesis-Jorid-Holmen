from ultralytics import YOLO
import os
import json
import matplotlib.pyplot as plt
import folium
import cv2
import glob
import shutil
import xml.etree.ElementTree as ET
from annotation_converter.AnnotationConverter import AnnotationConverter
import random
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def images_from_area_to_video(area, source, output_video, sequences='all'):
    dir = f'{source}/{area}'
    if sequences == 'all':
        sequences = [f for f in os.listdir(dir) if f.startswith('seq')]
        sequences = sorted(sequences, key=lambda x: int(x[3:]))
    img = sorted(glob.glob(f"{dir}/{sequences[0]}/imgs/*.png"))
    frame = cv2.imread(img[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'H264'), 2, (width, height))

    images = []
    for seq in sequences:
        image_folder = f'{dir}/{seq}/imgs'
        images = sorted(glob.glob(f"{image_folder}/*.png"))

        # Add images to the video
        for image in images:
            video.write(cv2.imread(image))

    video.release()
    return video

def get_gps_data_from_area(area, source, sequences='all'):
    gps_data = []
    if sequences == 'all':
        sequences = [f for f in os.listdir(f'{source}/{area}') if f.startswith('seq')]
        sequences = sorted(sequences, key=lambda x: int(x[3:]))


    for seq in sequences:
        gps_file = f'{source}/{area}/{seq}/gps.json'
        with open(gps_file, "r") as file:
            data = json.load(file)
        gps = sorted(
            data.items(),
            key=lambda x: (x[1]['header']['stamp']['secs'], x[1]['header']['stamp']['nsecs'])
        )
        gps_data += gps

    latitudes = []
    longitudes = []

    for key, value in gps_data:
        latitude = value['latitude']
        longitude = value['longitude']
        latitudes.append(latitude)
        longitudes.append(longitude)

    return latitudes, longitudes


def get_unique_tracking_ids_with_frames(track_res):
    ids = []

    for frame_idx, result in enumerate(track_res):    
        # Access the bounding box data    
        if result.boxes is not None:
            for box in result.boxes:
                try:
                    ids.append(box.id.item())
                except:
                    pass

    id_name_frame_dict = {}

    for target_tracking_id in set(ids):
        for frame_idx, result in enumerate(track_res):        
            # Access the bounding boxes
            if result.boxes is not None:
                for box in result.boxes:
                    # Check if the tracking ID matches
                    if box.id == target_tracking_id:
                        class_id = box.cls  # Class ID
                        class_name = result.names[int(class_id)]  # Map class ID to class name
                        id_name_frame_dict[target_tracking_id] = [class_name, frame_idx]
                        break  # Stop searching once found

    return id_name_frame_dict

def class_counts_based_on_area(area, latitudes_longitudes_obtus, latitudes_longitudes_crisp, output_file):
    class_counts = {}

    # for tracking_id, value in tracking_id_dict.items():
    #     class_name = value[0]
    #     if class_name not in class_counts:
    #         class_counts[class_name] = 1
    #     else:
    #         class_counts[class_name] += 1

    class_counts['rumex_obtusifolius'] = len(latitudes_longitudes_obtus)
    class_counts['rumex_crispus'] = len(latitudes_longitudes_crisp)

    # Ground truth values based on area
    ground_truth_values = {
        '20210806_hegnstrup': (41, 5),
        '20210806_stengard': (3653, 1296),
        '20211006_stengard': (4358, 1142),
        '20210807_lundholm': (774, 264),
        '20210908_lundholm': (3208, 172)
    }

    # Get the correct ground truth values
    if area in ground_truth_values:
        r_obtusifolius_gt, r_crispus_gt = ground_truth_values[area]
    else:
        raise ValueError(f"Unknown area: {area}")

    det_obtus = class_counts['rumex_obtusifolius']
    det_crisp = class_counts['rumex_crispus']

    data = [
        ["Ground Truth", r_obtusifolius_gt, r_crispus_gt], 
        ["Detected", det_obtus, det_crisp]
    ]

    columns = ['', 'Rumex Obtusifolius', 'Rumex Crispus']

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_file + '.csv')

    fig, ax = plt.subplots(figsize=(8, 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')


    for (i, j), cell in table.get_celld().items():
        cell.set_fontsize(13)
        cell.set_height(0.5)
    
    plt.savefig(output_file + '.png', bbox_inches='tight', dpi=300)
    

def get_gps_coordinates_for_detected_weeds(id_name_frame_dict, latitudes, longitudes):
    r_obtusifolius = 0
    r_crispus = 0
    frames_obtusifolius = []
    frames_crispus = []

    for key, value in id_name_frame_dict.items():
        if value[0] == "rumex_obtusifolius":
            r_obtusifolius += 1
            frames_obtusifolius.append(value[1])
        elif value[0] == "rumex_crispus":
            r_crispus += 1
            frames_crispus.append(value[1])

    obtusifolius_latitudes = [latitudes[i] for i in sorted(frames_obtusifolius)]
    obtusifolius_longitudes = [longitudes[i] for i in sorted(frames_obtusifolius)]

    crispus_latitudes = [latitudes[i] for i in sorted(frames_crispus)]
    crispus_longitudes = [longitudes[i] for i in sorted(frames_crispus)]

    return obtusifolius_latitudes, obtusifolius_longitudes, crispus_latitudes, crispus_longitudes


def plot_gps_coordinates(latitudes_longitudes_obtus, latitudes_longitudes_crisp, output):
    
    obtusifolius_latitudes = [i[0] for i in latitudes_longitudes_obtus]
    obtusifolius_longitudes = [i[1] for i in latitudes_longitudes_obtus]
    crispus_latitudes = [i[0] for i in latitudes_longitudes_crisp]
    crispus_longitudes = [i[1] for i in latitudes_longitudes_crisp]
    

    plt.figure(figsize=(10, 6))
    plt.scatter(obtusifolius_longitudes, obtusifolius_latitudes, marker='o', label='Rumex Obtusifolius')
    plt.scatter(crispus_longitudes, crispus_latitudes, marker='o', label='Rumex Crispus')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GPS Coordinates')
    plt.grid(True)
    plt.legend()
    plt.savefig(output)
    # plt.show()

def plot_gps_coordinates_to_map(latitudes_longitudes_obtus, latitudes_longitudes_crisp, output):
    
    obtusifolius_longitudes = [i[1] for i in latitudes_longitudes_obtus]
    obtusifolius_latitudes = [i[0] for i in latitudes_longitudes_obtus]

    crispus_longitudes = [i[1] for i in latitudes_longitudes_crisp]
    crispus_latitudes = [i[0] for i in latitudes_longitudes_crisp]

    center = (obtusifolius_latitudes[len(obtusifolius_latitudes) // 2], obtusifolius_longitudes[len(obtusifolius_latitudes) // 2])
    satelite_tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    satelite_map = folium.Map(center, tiles=satelite_tiles, attr="Esri", name="Esri Satellite", zoom_start=23, max_zoom=20)

    obtusifolius_group_sat = folium.FeatureGroup("Rumex obtusifolius").add_to(satelite_map)
    crispus_group_sat = folium.FeatureGroup("Rumex crispus").add_to(satelite_map)

    folium.LayerControl().add_to(satelite_map)

    for lat, lon in zip(obtusifolius_latitudes, obtusifolius_longitudes):
        folium.Marker(
            location=[lat, lon],
            popup="Rumex Obtusifolius",
            icon=folium.Icon(color='green')
        ).add_to(obtusifolius_group_sat)

    for lat, lon in zip(crispus_latitudes, crispus_longitudes):
        folium.Marker(
            location=[lat, lon],
            popup="Rumex Crispus",
            icon=folium.Icon(color='red')
        ).add_to(crispus_group_sat)

    satelite_map.save(output)

def clear_directories(source):
    if not os.path.exists(source):
        print(f"The directory '{source}' does not exist.")
        return
    
    # Iterate over all the contents in the directory
    for item_name in os.listdir(source):
        item_path = os.path.join(source, item_name)
        
        # Check if it is a file or a directory
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # Delete file or symbolic link
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Delete the directory and its contents
    
    print(f"All contents of '{source}' have been deleted.")


def parse_cvat_annotation(main_dir, location_folders, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    labels_dir = os.path.join(output_dir, f"labels/train")
    images_dir = os.path.join(output_dir, f"images/train")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    for folder in location_folders:
        full_path = os.path.join(main_dir, folder)
        print(f"Location folder: {folder}")
        for entry in os.listdir(full_path):

            full_path = os.path.join(main_dir, folder, entry)
            if os.path.isdir(full_path):  # Check if it's a folder
                print(f"Sequence folder: {entry}")
                current_annotation_file = f"{main_dir}/{folder}/{entry}/annotations.xml"
                current_image_dir = f"{main_dir}/{folder}/{entry}/imgs"

                tree = ET.parse(current_annotation_file)
                root = tree.getroot()

                img_files = glob.glob(f"{current_image_dir}/*.png")

                for index, image_tag in enumerate(root.findall("image")):
                    file_name = image_tag.attrib["name"]
                    src_image_path = os.path.join(current_image_dir, file_name)

                    # Read and resize the image
                    img = cv2.imread(src_image_path)

                    img_width = int(float(image_tag.attrib["width"]))
                    img_height = int(float(image_tag.attrib["height"]))

                    # Save resized image to the output directory
                    dst_image_path = os.path.join(images_dir, file_name)
                    os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
                    cv2.imwrite(dst_image_path, img) 

                    # Copy image to YOLO images directory
                    src_image_path = os.path.join(current_image_dir, file_name)
                    dst_image_path = os.path.join(images_dir, file_name)
                    if not os.path.exists(dst_image_path):
                        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
                        os.system(f"cp {src_image_path} {dst_image_path}")

                            # Create corresponding label file
                    label_path = os.path.join(labels_dir, os.path.splitext(file_name)[0] + ".txt")
                    with open(label_path, "a") as label_file:
                        # for box in image_tag.findall("box"):
                            # label = box.attrib["label"]

                        annotation = AnnotationConverter.read_cvat_by_id(current_annotation_file, file_name)

                        bbs = annotation.get_bounding_boxes()
                        for bb in bbs:
                            label = bb.get_label()
                    
                            # Calculate YOLO-format bounding box values
                            bbox_x_min = int(bb.get_x())  # Top-left corner x
                            bbox_y_min = int(bb.get_y())  # Top-left corner y
                            bbox_width = int(bb.get_width())  # Width of the bounding box
                            bbox_height = int(bb.get_height())  # Height of the bounding box

                            # Center coordinates
                            center_x = bbox_x_min + (bbox_width / 2)
                            center_y = bbox_y_min + (bbox_height / 2)

                            # Normalize all values
                            center_x_normalized = center_x / img_width
                            center_y_normalized = center_y / img_height
                            width_normalized = bbox_width / img_width
                            height_normalized = bbox_height / img_height

                            # Ensure the values are within [0, 1]
                            if (0 <= center_x_normalized <= 1 and 0 <= center_y_normalized <= 1 and 
                                0 <= width_normalized <= 1 and 0 <= height_normalized <= 1):
                                if label == 'rumex_obtusifolius':
                                    label_file.write(f"{0} {center_x_normalized} {center_y_normalized} {width_normalized} {height_normalized}\n")
                                elif label == 'rumex_crispus':
                                    label_file.write(f"{1} {center_x_normalized} {center_y_normalized} {width_normalized} {height_normalized}\n")
                            else:
                                print(f"Skipping bounding box with out-of-bounds values: {center_x}, {center_y}, {bbox_width}, {bbox_height}")


def train_test_split1(main_dir, train_ratio, val_ratio, test_ratio):
    # Ensure the ratios add up to 1
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("Train, validation, and test ratios must sum up to 1.")
    
    # Paths for images and labels
    image_folder = os.path.join(main_dir, 'images')
    label_folder = os.path.join(main_dir, 'labels')
    
    # Define subfolders for train, val, and test
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(image_folder, split), exist_ok=True)
        os.makedirs(os.path.join(label_folder, split), exist_ok=True)
    
    # List all files in the image source folder
    source_images = [f for f in os.listdir(os.path.join(image_folder, 'train')) 
                     if os.path.isfile(os.path.join(image_folder, 'train', f))]
    
    # Shuffle the files
    random.shuffle(source_images)
    
    # Calculate the split indices
    total_files = len(source_images)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    
    # Divide the files into splits
    train_files = source_images[:train_end]
    val_files = source_images[train_end:val_end]
    test_files = source_images[val_end:]
    
    split_mapping = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # Move files and corresponding labels
    for split, file_list in split_mapping.items():
        for file_name in file_list:
            # Define source and destination for images
            image_source_path = os.path.join(image_folder, 'train', file_name)
            image_dest_path = os.path.join(image_folder, split, file_name)

            # Define source and destination for labels (assuming labels are .txt files)
            label_source_path = os.path.join(label_folder, 'train', file_name.rsplit('.', 1)[0] + '.txt')
            label_dest_path = os.path.join(label_folder, split, file_name.rsplit('.', 1)[0] + '.txt')

            # Move the image file
            if os.path.exists(image_source_path):
                shutil.move(image_source_path, image_dest_path)
            
            # Move the corresponding label file
            if os.path.exists(label_source_path):
                shutil.move(label_source_path, label_dest_path)
    
    print("Dataset split completed.")
    print(f"Train: {len(train_files)} files")
    print(f"Validation: {len(val_files)} files")
    print(f"Test: {len(test_files)} files")


def train_test_split_stratify(main_dir, train_ratio, val_ratio, test_ratio):
    # Ensure the ratios add up to 1
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("Train, validation, and test ratios must sum up to 1.")
    
    # Paths for images and labels
    image_folder = os.path.join(main_dir, 'images')
    label_folder = os.path.join(main_dir, 'labels')
    
    # Define subfolders for train, val, and test
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(image_folder, split), exist_ok=True)
        os.makedirs(os.path.join(label_folder, split), exist_ok=True)
    
    # List all files in the image source folder
    source_images = [f for f in os.listdir(os.path.join(image_folder, 'train')) 
                     if os.path.isfile(os.path.join(image_folder, 'train', f))]
    
    # Calculate the split indices
    total_files = len(source_images)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)


    data = []
    for img_file in source_images:
        label_file = os.path.join(label_folder, 'train', img_file.rsplit('.', 1)[0] + ".txt")
        if os.path.exists(label_file):
            main_class = get_main_class(label_file)
            if main_class is not None:
                data.append((img_file, label_file, main_class))
            else: 
                data.append((img_file, label_file, -1))

    df = pd.DataFrame(data, columns=["image", "label", "class"])
    print(f"Sum: {len(df)} files")

    train_df, temp_df = train_test_split(df, test_size=(val_ratio + test_ratio), stratify=df["class"])
    val_df, test_df = train_test_split(temp_df, test_size=(test_ratio / (val_ratio + test_ratio)), stratify=temp_df["class"])


    split_mapping = {
        'train': train_df['image'].tolist(),
        'val': val_df['image'].tolist(),
        'test': test_df['image'].tolist()
    }

    # Move files and corresponding labels
    for split, file_list in split_mapping.items():
        for file_name in file_list:
            # Define source and destination for images
            image_source_path = os.path.join(image_folder, 'train', file_name)
            image_dest_path = os.path.join(image_folder, split, file_name)

            # Define source and destination for labels (assuming labels are .txt files)
            label_source_path = os.path.join(label_folder, 'train', file_name.rsplit('.', 1)[0] + '.txt')
            label_dest_path = os.path.join(label_folder, split, file_name.rsplit('.', 1)[0] + '.txt')

            # Move the image file
            if os.path.exists(image_source_path):
                shutil.move(image_source_path, image_dest_path)
            
            # Move the corresponding label file
            if os.path.exists(label_source_path):
                shutil.move(label_source_path, label_dest_path)
    
        
    print("Dataset split completed.")
    print(f"Train: {len(train_df['image'].tolist())} files")
    print(f"Validation: {len(val_df['image'].tolist())} files")
    print(f"Test: {len(test_df['image'].tolist())} files")

# def move_files(df, split_type, image_folder, label_folder):
#     for _, row in df.iterrows():
#         # Define source and destination for images
#         image_source_path = os.path.join(image_folder, 'train', row["image"])
#         image_dest_path = os.path.join(image_folder, split_type, row["image"])

#         # Define source and destination for labels
#         label_source_path = os.path.join(label_folder, 'train', row["label"].rsplit('/', 1)[-1])
#         label_dest_path = os.path.join(label_folder, split_type, row["label"].rsplit('/', 1)[-1])

#         # Move image and label
#         if os.path.exists(image_source_path):
#             shutil.move(image_source_path, image_dest_path)
#         if os.path.exists(label_source_path):
#             shutil.move(label_source_path, label_dest_path)


def get_main_class(label_file):
    with open(label_file, "r") as f:
        lines = f.readlines()
        if len(lines) == 0:
            return None  # Empty label file
        classes = [int(line.split()[0]) for line in lines]  # Extract class IDs

        s = sum(classes)/len(classes)

        if s > 0.5:
            return 1
        else:
            return 0


def select_random_images(folder_path, num_images=100):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    
    # Get all images in the folder
    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(tuple(image_extensions))]

    # Ensure we don't request more images than available
    num_images = min(num_images, len(image_files))

    # Randomly select images
    random_images = random.sample(image_files, num_images)

    return [os.path.join(folder_path, img) for img in random_images]  # Return full paths



def create_table(val_results, output_image, output_file):
    # Create a DataFrame from the validation output

    # Define the column names
    columns = ['Inference speed (ms)', 'Precision', 'Recall', 'mAP50', 'mAP50-95']

    values = [val_results.speed['inference'], 
              val_results.results_dict['metrics/precision(B)'], 
              val_results.results_dict['metrics/recall(B)'], 
              val_results.results_dict['metrics/mAP50(B)'], 
              val_results.results_dict['metrics/mAP50-95(B)']]
    
    values = [round(i, 3) for i in values]

    df = pd.DataFrame([values], columns=columns)

    fig, ax = plt.subplots(figsize=(8, 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    for (i, j), cell in table.get_celld().items():
        cell.set_fontsize(13)
        cell.set_height(0.5)

    plt.savefig(output_image, bbox_inches='tight', dpi=300)
    df.to_csv(output_file, index=False)

    # plt.show()


# def train_plots(train_dir, output_image):
#     # Load CSV only once
#     train_res = train_dir + '/results.csv'
#     df = pd.read_csv(train_res)

#     # Extract epochs
#     epochs = df['epoch']

#     # Create figure and subplots
#     fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))  # Adjust figure size

#     # Plot metrics
#     metrics = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
#     titles = ["Precision Over Epochs", "Recall Over Epochs", "mAP50 Over Epochs", "mAP50-95 Over Epochs"]
#     labels = ["Precision", "Recall", "mAP50", "mAP50-95"]

#     for i, ax in enumerate(axes):
#         ax.plot(epochs, df[metrics[i]], label=labels[i])  # Use pre-loaded df
#         ax.set_title(titles[i])
#         ax.set_xlabel("Epochs")
#         ax.legend()
#         ax.grid()

#     # Adjust layout
#     plt.tight_layout()

#     # Save the figure
#     plt.savefig(output_image, dpi=300)
#     plt.show()


def train_plots(train_dir, output_image):

    train_res = train_dir + '/results.csv'
    epochs = pd.read_csv(train_res)['epoch']

    # Create figure and subplots (4 columns, 2 rows)
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))  # Adjust size as needed

    axes[0].plot(epochs, pd.read_csv(train_res)['metrics/precision(B)'], label='Precision')
    axes[1].plot(epochs, pd.read_csv(train_res)['metrics/recall(B)'], label='Recall')
    axes[2].plot(epochs, pd.read_csv(train_res)['metrics/mAP50(B)'], label='mAP50')
    axes[3].plot(epochs, pd.read_csv(train_res)['metrics/mAP50-95(B)'], label='mAP50-95')

    axes[0].set_title("Precision Over Epochs")
    axes[1].set_title("Recall Over Epochs")
    axes[2].set_title("mAP50 Over Epochs")
    axes[3].set_title("mAP50-95 Over Epochs")

    for ax in axes.flat:
        ax.set_box_aspect(1)
        ax.set(xlabel='Epochs')
        ax.legend()
        ax.grid()

    # Adjust layout to avoid overlapping
    plt.tight_layout()

    # Save the figure (optional)
    plt.savefig(output_image, dpi=300)

def gather_curve_plots(val_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = ['R_curve.png', 'P_curve.png', 'PR_curve.png']

    for file in files:
        source_path = os.path.join(val_dir, file)
        destination_path = os.path.join(output_dir, file)

        # Move file only if it exists
        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
            print(f"✅ Moved: {file} → {output_dir}")
        else:
            print(f"⚠️ File not found: {source_path}")
