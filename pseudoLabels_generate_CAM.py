"""
    create pseudo labels by CAM
"""
import numpy as np
import os
from glob import glob
from tqdm import tqdm


save_dir = r'image-level_pseudo'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

generator_dir = r'SAMgenerator'
cam_dir = r''  # the CAM dir

cam_paths = glob(cam_dir + '/*.npy')
id_patches = []
for path in cam_paths:
    id_patch = os.path.basename(path)[:5]
    if id_patch not in id_patches:
        id_patches.append(id_patch)

for id_patch in tqdm(id_patches):
    cam_1 = np.load(os.path.join(cam_dir, f'{id_patch}_1.npy'))
    cam_0 = np.load(os.path.join(cam_dir, f'{id_patch}_0.npy'))
    generator = np.load(os.path.join(generator_dir, f'{id_patch}.npy'))

    foreground_pseudo = np.zeros_like(cam_1).astype(np.float32)
    background_pseudo = np.zeros_like(cam_0).astype(np.float32)
    for mask in generator:
        indices = np.where(mask == 1)
        mean_value_fg = np.mean(cam_1[indices])
        mean_value_bg = np.mean(cam_0[indices])
        foreground_pseudo[indices] = mean_value_fg
        background_pseudo[indices] = mean_value_bg
    pseudo = np.argmax(np.array([background_pseudo, foreground_pseudo]), axis=0)
    np.save(os.path.join(save_dir, f'{id_patch}.npy'), pseudo)
