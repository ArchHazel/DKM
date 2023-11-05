from mobile_sam import SamAutomaticMaskGenerator,sam_model_registry
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
from pathlib import Path
import os
import torch
import time
from pathlib import Path
from tqdm import tqdm

# pre-trained weights of mobileSAM
weights_path = Path("weights")

sam_checkpoint = weights_path/"mobile_sam.pt"
model_type = "vit_t"
device = "cuda" 
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()
# sam = torch.compile(sam)

mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=1
)
           

def save_anns_matrix(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    categories_contour = np.zeros_like(sorted_anns[0]["segmentation"], dtype=np.int32) 
    # print(categories.shape)
    for ann_i, ann in enumerate( sorted_anns):
        categories = np.zeros_like(sorted_anns[0]["segmentation"], dtype=np.int32)
        m = ann["segmentation"]
        categories[m] = ann_i + 1
        h, w = categories.shape
        categories = np.float32(categories.reshape(h, w, 1))

        # categories = cv2.resize(categories, dsize=None, fx=0.5, fy=0.5)
        categories_dx = np.abs(cv2.Sobel(categories, ddepth=-1, dx=1, dy=0, ksize=3))
        categories_dy = np.abs(cv2.Sobel(categories, ddepth=-1, dx=0, dy=1, ksize=3))
        categories_contour = categories_contour + categories_dx + categories_dy

    np.save("ann.npy",sorted_anns,allow_pickle=True)





megadepth_indices_train = Path("data/Megadepth/train-data/megadepth_indices/prep_scene_info")
sceneInfoFile = os.listdir(megadepth_indices_train)
megadepth_image_path = Path("data/Megadepth/phoenix/S6/zl548")
megadepth_masks_path = Path("data/Megadepth/masks")




for  f in sceneInfoFile:
    img_paths = np.load(megadepth_indices_train/f,allow_pickle = True)
    img_paths = img_paths.item()


    for p in tqdm(img_paths["image_paths"]):
        if p is None:
            continue
        
        im = cv2.imread(str(megadepth_image_path / p))
        # change color channel from RGB to BGR
        im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        im = cv2.resize(im, None, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)

        # init im_Mask with the same size as the original image
        im_Mask = np.zeros(im.shape[:2], dtype=np.uint16)
        print(im_Mask.shape)

        masks = mask_generator_2.generate(im)

        # sort masks by area
        sorted_masks = sorted(masks,key = (lambda x:x["area"]),reverse = True)

        for id, mask in enumerate(sorted_masks):
            m = mask["segmentation"]
            im_Mask[m] = id + 1

        cv2.imwrite(str(megadepth_masks_path/os.path.basename(p)),im_Mask)
        break
    break



