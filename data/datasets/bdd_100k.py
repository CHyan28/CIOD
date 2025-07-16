# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET
from random import shuffle

from iod.structures import BoxMode
from iod.data import DatasetCatalog, MetadataCatalog
from iod.utils.logger import setup_logger
import glob
logger = setup_logger(name=__name__)

__all__ = ["register_pascal_voc"]


# fmt: off
CLASS_NAMES = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic sign', 'traffic light']


def load_bdd_instances(dirname: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    
    Args:
       split = 'train'/ 'val'; dirname = '/root/data/iOD/datasets/BDD100k/bdd100k'
    """
    # with PathManager.open(os.path.join(dirname, "txt_labels", split, "*.txt")) as f:
    #     fileids = np.loadtxt(f, dtype=np.str)
    path = os.path.join(dirname, "txt_labels", split)
    fileids = []
    for file in os.listdir(path):
        fileids.append(file.split('.')[0])
    # shuffle(CLASS_NAMES)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", split, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "images/100k", split, fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts

# name = 'bdd_100k_train'; split = 'train'; dirname = '/root/data/iOD/datasets/BDD100k/bdd100k'

def register_bdd_100k(name, dirname, split):
    DatasetCatalog.register(name, lambda: load_bdd_instances(dirname, split)) # split = 'trainval'; dirname = '/root/data/iOD/datasets/BDD100k/bdd100k'
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, split=split
    )
# thing_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', ...]