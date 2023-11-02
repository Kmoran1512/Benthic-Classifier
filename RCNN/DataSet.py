import os
import torch
import xml.etree.ElementTree as ET

from torchvision.io import read_image

#https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=W9e7thfLcwS0

class CustomVOCDataset(torch.utils.data.Dataset):
    def __init__(self, target_path):
        data_dir = os.path.join('..', 'datasets', 'sea_data')

        self.image_dir = os.path.join(data_dir, 'images', target_path)
        self.label_dir = os.path.join(data_dir, 'labels', target_path)

        self.image_files = [f for f in sorted(os.listdir(self.image_dir)) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        
        image = read_image(img_name)

        annotation_file = os.path.join(self.label_dir, os.path.splitext(self.image_files[idx])[0] + '.xml')
        boxes, labels = zip(*parse_xml(annotation_file))


        padded_boxes = torch.tensor([0, 0, 224, 224], dtype=torch.float32).repeat(self.max_size, 1)
        padded_labels = torch.zeros(self.max_size, dtype=torch.int64)
    
        # Copy the original boxes and labels into the padded tensors
        padded_boxes[-len(boxes):] = torch.tensor(boxes, device=device, dtype=torch.float)
        padded_labels[-len(boxes):] = torch.tensor([int(concepts_to_include.index(label) + 1) for label in labels], device=device, dtype=torch.int)
        
        target = {
            "boxes": padded_boxes,
            "labels": padded_labels
        }

        return torch.from_numpy(image / 255.0), target
    
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
