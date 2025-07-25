import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# 将KITTI数据集中一个图像的标注文件转换为YOLO格式的标注文件

sets=['train', 'val','test']

classes = ["person", "bicycle" , "car" , "motorcycle", "bus" , "truck"]

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    in_file = open('/root/data/iOD/datasets/KITTI/Annotations/%s.xml'%(image_id))
    out_file = open('/root/data/iOD/datasets/KITTI/ImageSets/labels/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for image_set in sets:
    if not os.path.exists('/root/data/iOD/datasets/KITTI/ImageSets/labels/'):
        os.makedirs('/root/data/iOD/datasets/KITTI/ImageSets/labels/')
    image_ids = open('/root/data/iOD/datasets/KITTI/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    list_file = open('%s.txt'%(image_set), 'w')
    for image_id in image_ids:
        list_file.write('/root/data/iOD/datasets/KITTI/JPEGImages/%s.png\n'%(image_id))
        convert_annotation(image_id)
    list_file.close()

os.system("cat train.txt val.txt > train_val.txt")
os.system("cat train.txt val.txt test.txt > train.all.txt")