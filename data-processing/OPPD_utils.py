import os
import json
import platform
import pandas as pd
import numpy as np
import cv2


def createIfNotExist(directory):
    """ Create directory if it does not exist

    params:
        directory (string): directort to be created
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def getImagesInFolder(path, extensions = ('.jpg','.png')):
    """ Get a list of all images in a folder, including subfolders

    params:
        path: path to be searched
        extensions: extensions to be searched for
    
    returns:
        lstImages: list of tuples containing folder and filename for each image
    """
    lstImages = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(extensions):
                lstImages.append((root,name))
    return lstImages

	
def readJSONAnnotation(pathAnno):
    """ Load JSON annotation file

    params:
        pathAnno: Where the JSON annotation file is located

    returns:
        anno: Dictionary containing the annotation data
    """
    
    with open(pathAnno) as data_file:
        anno = json.load(data_file)
    
    return anno

def writeJSONAnnotation(pathAnno, anno):
    """ Write annotation to JSON file

    params:
        pathAnno: Where to save the JSON annotation file
        anno: Dictionary containing the annotation data
    """
    with open(pathAnno, 'w') as data_file:
        json.dump(anno, data_file)


def printWithStyle(msg, colour=None, formating=None):
    """ Print to terminal with colour and formating style,

    params:
        msg: Message to print
        colour: string with color for msg. Supported values {'GRAY','RED','GREEN','YELLOW','BLUE','MAGENTA','CYAN'}
        formating: list of strings with formating options. Supported values: {'BOLD','ITALIC','UNDERLINE'}
    """

    class pcolors:
        ENDC = '\033[0m'
        GRAY = '\033[90m'
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN = '\33[96m'

    class pformats:
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        ITALIC = '\033[3m'
        UNDERLINE = '\033[4m'

    # ADD colour
    if colour == 'GRAY':
        msg = pcolors.GRAY + msg + pcolors.ENDC
    elif colour == 'RED':
        msg = pcolors.RED + msg + pcolors.ENDC
    elif colour == 'GREEN':
        msg = pcolors.GREEN + msg + pcolors.ENDC
    elif colour == 'YELLOW':
        msg = pcolors.YELLOW + msg + pcolors.ENDC
    elif colour == 'BLUE':
        msg = pcolors.BLUE + msg + pcolors.ENDC
    elif colour == 'MAGENTA':
        msg = pcolors.MAGENTA + msg + pcolors.ENDC
    elif colour == 'CYAN':
        msg = pcolors.CYAN + msg + pcolors.ENDC
    else:
        msg = msg
    
    # ADD formating
    if formating:
        if "BOLD" in formating:
            msg = pformats.BOLD + msg + pformats.ENDC
        if "ITALIC" in formating:
            msg = pformats.ITALIC + msg + pformats.ENDC
        if "UNDERLINE" in formating:
            msg = pformats.UNDERLINE + msg + pformats.ENDC

    print(msg)


def unique_lst(lst): 
    # intilize a null list
    lst_unique = []
      
    # traverse for all elements 
    for element in lst:
        if element not in lst_unique:
            lst_unique.append(element)

    return lst_unique

def addPolygon2Image(img, polygon, color=(255,0,0), alpha=0.2, thickness=5):
    """ Add polygon to an image

    params:
        img: cv2 image
        polygon: array of corner coordinates
        color: color for polygon
        alpha: transparency of polygon fill
        thickness: thickness of polygon lines

    returns:
        output: image with added polygon
    """
    overlay = img.copy()
    output = img.copy()
    polygon = [np.int32(polygon)]

    # cv2.polylines(overlay, [corners], 1, color)
    cv2.fillPoly(overlay, polygon, color)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    cv2.polylines(output, polygon, 1, color, thickness)

    return output

def bndbox2polygon(bndbox):
    polygon = np.array([[bndbox['xmin'],bndbox['ymin']],
                        [bndbox['xmin'],bndbox['ymax']],
                        [bndbox['xmax'],bndbox['ymax']],
                        [bndbox['xmax'],bndbox['ymin']]],dtype=np.int32)
    return polygon

def addBndBoxes2Image(path_img, color=(255,0,0), alpha=0.2, thickness=5):
    """ Add a polygon for each bounding box annotation associated with the image

    params:
        path_img: path to image
        color: color of the polygons
        alpha: transparency of polygons fill
        thickness: thickness of polygons lines

    returns:
        output: image with added polygons for each bounding box
    """
    path_anno = path_img.replace('.jpg','.json')
    image = cv2.imread(path_img)
    anno = readJSONAnnotation(path_anno)

    output = image.copy()
    for plant in anno['plants']:
        bndbox = plant['bndbox']
        coor_bndbox = bndbox2polygon(bndbox)

        output = addPolygon2Image(output,coor_bndbox, color,alpha, thickness)
    
    return output
