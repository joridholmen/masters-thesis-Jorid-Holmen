from OPPD_utils import *
import os
import cv2
from utilities import *

output_dir = 'COCOCrispusInBox'
labels_dir = os.path.join(output_dir, 'labels/test')
images_dir = f'{output_dir}/images/test'
path_imgs = 'crispusInBox'


images = sorted(glob.glob(f"{images_dir}/*.jpg"))
labels = []

for img in images:
    labels.append(img.replace('images', 'labels').replace('.jpg', '.txt'))

annos = getImagesInFolder(path_imgs, 'json')

for anno in annos:
    anno_path = os.path.join(anno[0], anno[1])
    image_path = os.path.join(anno[0], anno[1].replace('.json', '.jpg'))
    new_anno_path = os.path.join(labels_dir, anno[1].replace('.json', '.txt'))
    if new_anno_path in labels:
        anno_dict = readJSONAnnotation(anno_path)

        image = cv2.imread(image_path)
        img_height, img_width, _ = image.shape

        with open(new_anno_path, 'w') as anno_file:

            for plant in anno_dict['plants']:
                xmin, ymin, xmax, ymax = plant['bndbox'].values()
                w = xmax - xmin
                h = ymax - ymin

                # Convert to YOLO format (center_x, center_y, width, height)
                center_x = xmin + (w / 2)
                center_y = ymin + (h / 2)

                # Normalize values
                center_x /= img_width
                center_y /= img_height
                w /= img_width
                h /= img_height

                # Ensure values are within [0, 1]
                if 0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                    anno_file.write(f"1 {center_x} {center_y} {w} {h}\n")
                else:
                    print(f"Skipping bounding box with out-of-bounds values: {xmin}, {ymin}, {xmax}, {ymax}")

