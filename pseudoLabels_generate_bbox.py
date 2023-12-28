"""
    create pseudo labels by bbox annotations
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sam import sam_model_registry as adapter_sam_model_registry
from sam import SamPredictor as adapter_SamPredictor
import utilis
from tqdm import tqdm

model_type_tuned = 'vit_b'
checkpoint_tuned = 'adapter_sam_model.pth'
device = 'cuda:0'
sam_model_tuned = adapter_sam_model_registry[model_type_tuned](checkpoint=checkpoint_tuned)
sam_model_tuned.to(device)
predictor_tuned = adapter_SamPredictor(sam_model_tuned)
print("=====================================model loaded========================================")

data_split = r'/PASTIS_AgriculturalMeadow/dataset_split.xlsx'
data_split = pd.read_excel(data_split)
id_patches = data_split['Image Number'].to_list()

image_dir = r'/PASTIS_AgriculturalMeadow/image'
instance_dir = r'/PASTIS_AgriculturalMeadow/instance'
point_dir = r'/PASTIS_AgriculturalMeadow/heatmap'
semantic_dir = r'/PASTIS_AgriculturalMeadow/semantic'

pseudo_saveDir = r'bbox_pseudo'
if not os.path.exists(pseudo_saveDir):
    os.makedirs(pseudo_saveDir)

for image_id in tqdm(id_patches):
    gee_image = os.path.join(image_dir, f'{image_id}.npy')
    point_label = os.path.join(point_dir, f'heatmap_{image_id}.npy')
    instance_label = os.path.join(instance_dir, f'instance_{image_id}.npy')
    semantic_label = os.path.join(semantic_dir, f'semantic_{image_id}.npy')

    if os.path.exists(gee_image) and os.path.exists(point_label) and os.path.exists(instance_label) and os.path.exists(semantic_label):
        point_label = np.load(point_label)
        instance_label = np.load(instance_label)
        semantic_label = np.load(semantic_label)

        bbox_prompts = []
        for instance_id in np.unique(instance_label):
            positions = np.where(instance_label == instance_id)

            semantic_id = np.unique(semantic_label[positions])[0]

            if semantic_id == 1:
                parcel = np.zeros_like(semantic_label)
                parcel[positions] = 1

                rows, cols = np.where(parcel == 1)
                min_row, max_row, min_col, max_col = min(rows), max(rows), min(cols), max(cols)
                expansion = 2
                min_row = max(0, min_row - expansion)
                max_row = min(parcel.shape[0] - 1, max_row + expansion)
                min_col = max(0, min_col - expansion)
                max_col = min(parcel.shape[1] - 1, max_col + expansion)
                min_rect = np.zeros_like(semantic_label)
                min_rect[min_row:max_row + 1, min_col:max_col + 1] = 1

                bbox_prompts.append(min_rect)

        img = utilis.convert_uint8(np.load(gee_image)[[2, 1, 0], :, :])
        predictor_tuned.set_image(img)
        output_mask = []
        for prompt_mask in bbox_prompts:
            prompt = utilis.get_bbox(prompt_mask)

            pseudo_mask, _, _ = predictor_tuned.predict(
                point_coords=None,
                box=prompt,
                multimask_output=False,
            )
            output_mask.append(np.squeeze(pseudo_mask))
        final_mask = np.logical_or.reduce(output_mask)

        final_ground_truth = final_mask.astype(int)

        if not isinstance(final_ground_truth, np.ndarray):
            final_ground_truth = np.zeros((128, 128))
            print(f'-----------------{image_id}------------------')

        np.save(os.path.join(pseudo_saveDir, f'{image_id}.npy'), final_ground_truth)
