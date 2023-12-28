"""
    General segmentation mode of finetuned SAM
    create superpixels
"""
import pandas as pd
from sam import sam_model_registry as adapter_sam_model_registry
from sam import SamAutomaticMaskGenerator
import os
import utilis
import numpy as np
from tqdm import tqdm

save_dir = r'SAMgenerator'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

split_file = r'/PASTIS_AgriculturalMeadow/dataset_split.xlsx'
data_split = pd.read_excel(split_file)
ids = data_split.loc[(data_split['Dataset'] == 'Train') & (data_split['Labels'] == 1)]['Image Number'].to_list()

img_dir = r'/PASTIS_AgriculturalMeadow/image'

model_type_tuned = 'vit_b'
checkpoint_tuned = 'adapter_sam_model.pth'
device = 'cuda'
sam_model_tuned = adapter_sam_model_registry[model_type_tuned](checkpoint=checkpoint_tuned)
sam_model_tuned.to(device)
mask_generator = SamAutomaticMaskGenerator(sam_model_tuned, points_per_side=128)

for id_patch in tqdm(ids):
    img_path = os.path.join(img_dir, f'{id_patch}.npy')
    img = utilis.convert_uint8(np.load(img_path)[[2, 1, 0], :, :])

    masks = mask_generator.generate(img)

    save_mask = []
    for mask in masks:
        mask = mask['segmentation'].astype(int)
        save_mask.append(mask)
    save_mask = np.array(save_mask)
    np.save(os.path.join(save_dir, f'{id_patch}.npy'), save_mask)
