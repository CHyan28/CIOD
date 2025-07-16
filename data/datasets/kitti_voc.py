import xml.etree.ElementTree as ET
import os
import json

# KITTI类别列表
KITTI_CLASSES = ['car', 'pedestrian', 'cyclist']

# KITTI标注文件夹路径
kitti_ann_dir = '/root/data/KITTI/data_object_label_2/training/label_2'

# Pascal VOC标注文件夹路径
voc_ann_dir = '/root/data/iOD/datasets/KITTI/annotations'

# 将KITTI标注文件转换为Pascal VOC格式
for filename in os.listdir(kitti_ann_dir):
    if filename.endswith('.txt'):
        # 读取KITTI标注文件
        with open(os.path.join(kitti_ann_dir, filename), 'r') as f:
            lines = f.readlines()

        # 创建Pascal VOC标注文件
        root = ET.Element('annotation')
        filename_elem = ET.SubElement(root, 'filename')
        filename_elem.text = filename.replace('.txt', '.png')

        for line in lines:
            # 解析KITTI标注信息
            data = line.strip().split()
            class_name = KITTI_CLASSES[int(data[0])]
            bbox = [float(x) for x in data[4:8]]

            # 创建Pascal VOC目标元素
            obj_elem = ET.SubElement(root, 'object')
            name_elem = ET.SubElement(obj_elem, 'name')
            name_elem.text = class_name
            bbox_elem = ET.SubElement(obj_elem, 'bndbox')
            xmin_elem = ET.SubElement(bbox_elem, 'xmin')
            xmin_elem.text = str(int(bbox[0]))
            ymin_elem = ET.SubElement(bbox_elem, 'ymin')
            ymin_elem.text = str(int(bbox[1]))
            xmax_elem = ET.SubElement(bbox_elem, 'xmax')
            xmax_elem.text = str(int(bbox[2]))
            ymax_elem = ET.SubElement(bbox_elem, 'ymax')
            ymax_elem.text = str(int(bbox[3]))

        # 保存Pascal VOC标注文件
        tree = ET.ElementTree(root)
        tree.write(os.path.join(voc_ann_dir, filename.replace('.txt', '.xml')))




import torch
import glob
import os
import cv2
from PIL import Image
from iod.structures import BoxMode
from iod.data import DatasetCatalog, MetadataCatalog

CLASS_NAMES = []

class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # 读取图像列表和标注文件
        self.image_list = sorted(glob.glob(os.path.join(self.root_dir, 'image_2', self.split, '*.png')))
        self.label_list = sorted(glob.glob(os.path.join(self.root_dir, 'label_2', self.split, '*.txt')))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # 读取图像和标注信息
        image_path = self.image_list[index]
        label_path = self.label_list[index]
        image = Image.open(image_path)
        label = read_kitti_label(label_path)

        # 对图像进行预处理
        if self.transform is not None:
            image = self.transform(image)

        return image, label



def get_kitti_dicts(img_dir, label_dir):
    """
    将 KITTI 数据集的标注信息转换为 Detectron2 支持的格式
    """
    dataset_dicts = []
    label_files = os.listdir(label_dir)
    for label_file in label_files:
        img_file = os.path.join(img_dir, label_file[:-4] + '.png')
        record = {}
        record['file_name'] = img_file
        record['image_id'] = label_file[:-4]
        record['height'], record['width'] = cv2.imread(img_file).shape[:2]

        objs = []
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()
        for line in lines:
            data = line.strip().split(' ')
            obj = {}
            obj['bbox'] = [float(data[4]), float(data[5]), float(data[6]), float(data[7])]
            obj['bbox_mode'] = BoxMode.XYXY_ABS
            obj['category_id'] = int(data[0])
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)

    return dataset_dicts

def read_kitti_label(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    objects = []
    for line in lines:
        # 解析每一行的标注信息
        object_data = {}
        data = line.strip().split(' ')
        object_data['type'] = data[0]
        object_data['truncated'] = float(data[1])
        object_data['occluded'] = int(data[2])
        object_data['alpha'] = float(data[3])
        object_data['bbox'] = [float(x) for x in data[4:8]]
        object_data['dimensions'] = [float(x) for x in data[8:11]]
        object_data['location'] = [float(x) for x in data[11:14]]
        object_data['rotation_y'] = float(data[14])
        objects.append(object_data)
    return objects
