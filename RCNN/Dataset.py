import os
import torch
import torchvision
import xml.etree.ElementTree as ET

from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


#https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=W9e7thfLcwS0

class CustomVOCDataset(torch.utils.data.Dataset):
    def __init__(self, concepts, target_path):
        data_dir = os.path.join('..', 'datasets', 'sea_data')

        self.concepts = concepts

        self.image_dir = os.path.join(data_dir, 'images', target_path)
        self.label_dir = os.path.join(data_dir, 'labels', target_path)

        self.image_files = [f for f in sorted(os.listdir(self.image_dir)) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = torchvision.io.read_image(img_name)
        img = tv_tensors.Image(image)

        annotation_file = os.path.join(self.label_dir, os.path.splitext(self.image_files[idx])[0] + '.xml')
        labels, boxes = zip(*parse_xml(annotation_file))
        labels = [self.concepts.index(l) for l in labels]
    
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "labels": torch.tensor(labels)
        }

        return img, target
    
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bbox_pair = []

    for obj in root.findall('object'):
        name_key = obj.find('name').text
    
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        bbox_pair.append((name_key, (xmin, ymin, xmax, ymax)))
    
    return bbox_pair        
