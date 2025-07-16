from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET
from random import shuffle

from iod.structures import BoxMode
from iod.data import DatasetCatalog, MetadataCatalog
from iod.utils.logger import setup_logger

logger = setup_logger(name=__name__)

__all__ = ["register_kitti"]


# CLASS_NAMES= [
#     "person", "car", "truck", 
# ]


CLASS_NAMES= [
    "truck", "car", "misc", "cyclist","pedestrian","person_sitting","tram","van",
]


def load_kitti_instances(dirname: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # shuffle(CLASS_NAMES)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".png")

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

def register_kitti(name, dirname, split):
    DatasetCatalog.register(name, lambda: load_kitti_instances(dirname, split)) # split = 'trainval';dirname = '/storage1/syy/OWOD/datasets/VOC2007'
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, split=split
    )
    

