import os
import cv2
import numpy as np
import torch
import xml.etree.ElementTree as ET
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry

def generate_sam_masks(img_dir, xml_dir, output_dir, sam_checkpoint):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM model to {device}...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)
    
    os.makedirs(output_dir, exist_ok=True)
    classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

    for xml_file in tqdm(xml_files, desc="Generating Masks"):
        name = os.path.splitext(xml_file)[0]
        img_path = os.path.join(img_dir, f"{name}.jpg")
        if not os.path.exists(img_path): continue

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Set image once (Runs the heavy Vision Transformer encoder)
        predictor.set_image(img_rgb)
        
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        boxes = []
        cls_ids = []

        for obj in tree.findall('object'):
            cls_name = obj.find('name').text
            difficult = obj.find('difficult')
            is_difficult = difficult is not None and difficult.text == '1'
            
            if cls_name not in classes or is_difficult:
                cid = 255
            else:
                cid = classes.index(cls_name)
            
            box_node = obj.find('bndbox')
            # Fix float error and collect boxes
            coords = [
                float(box_node.find('xmin').text),
                float(box_node.find('ymin').text),
                float(box_node.find('xmax').text),
                float(box_node.find('ymax').text)
            ]
            boxes.append(coords)
            cls_ids.append(cid)

        if not boxes:
            # If no objects, save empty mask and move on
            cv2.imwrite(os.path.join(output_dir, f"{name}.png"), np.zeros(img.shape[:2], dtype=np.uint8))
            continue

        # --- BATCHED PREDICTION ---
        # Convert boxes to a Torch Tensor
        input_boxes = torch.tensor(boxes, device=device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, img_rgb.shape[:2])
        
        # predict_torch is MUCH faster for multiple boxes
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # Combine masks into one image
        # masks is shape [N, 1, H, W] where N is number of objects
        full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Move masks to CPU for numpy assignment
        masks = masks.cpu().numpy() # [N, 1, H, W]
        
        for i in range(len(cls_ids)):
            # masks[i, 0] is a boolean array
            full_mask[masks[i, 0]] = cls_ids[i]
            
        cv2.imwrite(os.path.join(output_dir, f"{name}.png"), full_mask)

if __name__ == "__main__":
    BASE_PATH = "/scratch/m25cse012/DL_Assignment_PSSI_COPY/DL_Mini_project/VOC2012_train_val/VOC2012_train_val"
    generate_sam_masks(
        os.path.join(BASE_PATH, "JPEGImages"),
        os.path.join(BASE_PATH, "Annotations"),
        os.path.join(BASE_PATH, "SAM_Segmentation_Maps1"),
        os.path.join(BASE_PATH, "sam_vit_h_4b8939.pth")
    )