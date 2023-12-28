"""
    Use adapters to finetune SAM
    Code is written by Jialin Sun
"""
import numpy as np
import re
import torch
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
import warnings
from time import time
import os
import pandas as pd
import utilis
from tqdm import tqdm
from loss_function import Myloss

warnings.filterwarnings("ignore", category=UserWarning)

# set seeds
torch.manual_seed(2023)
np.random.seed(2023)

start = time()

# -----------------load orignial SAM parameters to preprocess points and bbox-----------------
model_type = 'vit_b'
checkpoint = 'sam_vit_b_01ec64.pth'
device = 'cuda:0'
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)

# ----------------------load finetuning dataset-------------------
data_root = r'PASTIS_Finetune_dataset'
pd_split = os.path.join(data_root, 'dataset_split.xlsx')
img_dir = os.path.join(data_root, 'img')
bbox_dir = os.path.join(data_root, 'bbox')
mask_dir = os.path.join(data_root, 'mask')
point_dir = os.path.join(data_root, 'point')
pd_split = pd.read_excel(pd_split)

train_ids = pd_split.loc[pd_split['Dataset'] == 'Train']['Image Number'].to_list()
validation_ids = pd_split.loc[pd_split['Dataset'] == 'Validation']['Image Number'].to_list()

img_train = []
bbox_train = []
mask_train = []
point_train = []
for id_ in train_ids:
    img_path = os.path.join(img_dir, f'img_{id_}.npy')
    bbox_path = os.path.join(bbox_dir, f'bbox_{id_}.npy')
    mask_path = os.path.join(mask_dir, f'mask_{id_}.npy')
    point_path = os.path.join(point_dir, f'point_{id_}.npy')

    temp_point = utilis.get_point(np.load(point_path))
    point_train.append(temp_point)

    img_train.append(utilis.convert_uint8(np.load(img_path)[[2, 1, 0], :, :]))  # convert type to uint8 as SAM requires, and(H,W,C)
    bbox_train.append(utilis.get_bbox(np.load(bbox_path)))  # get coordinates of bbox
    mask_train.append(np.load(mask_path))
bbox_train = np.array(bbox_train)
mask_train = np.array(mask_train).astype(np.int32)
point_train = np.array(point_train)
# [b, 3, 1024, 1024]
input_image_train = []
for i in range(len(img_train)):
    embedding = utilis.preprocess_input_image(img_train[i], sam_model)
    input_image_train.append(embedding)

# preprocess format of bbox to what SAM needs
input_boxes_train = utilis.preprocess_input_bbox(bbox_train, sam_model)
# preprocess format of points to what SAM needs
input_points_train, input_labels_train = utilis.preprocess_input_points(point_train, sam_model)
img_train, bbox_train, point_train = None, None, None

img_val = []
bbox_val = []
mask_val = []
point_val = []
for id_ in validation_ids:
    img_path = os.path.join(img_dir, f'img_{id_}.npy')
    bbox_path = os.path.join(bbox_dir, f'bbox_{id_}.npy')
    mask_path = os.path.join(mask_dir, f'mask_{id_}.npy')
    point_path = os.path.join(point_dir, f'point_{id_}.npy')

    temp_point = utilis.get_point(np.load(point_path))
    point_val.append(temp_point)

    img_val.append(utilis.convert_uint8(np.load(img_path)[[2, 1, 0], :, :]))
    bbox_val.append(utilis.get_bbox(np.load(bbox_path)))
    mask_val.append(np.load(mask_path))
bbox_val = np.array(bbox_val)
mask_val = np.array(mask_val).astype(np.int32)
point_val = np.array(point_val)
# [b, 3, 1024, 1024]
input_image_val = []
for i in range(len(img_val)):
    embedding = utilis.preprocess_input_image(img_val[i], sam_model)
    input_image_val.append(embedding)

input_boxes_val = utilis.preprocess_input_bbox(bbox_val, sam_model)
input_points_val, input_labels_val = utilis.preprocess_input_points(point_val, sam_model)
img_val, bbox_val, point_val = None, None, None

# ==================================================dataloader======================================================
train_dataset = utilis.customedDS(input_image_train, input_boxes_train, mask_train, input_points_train, input_labels_train)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

val_dataset = utilis.customedDS(input_image_val, input_boxes_val, mask_val, input_points_val, input_labels_val)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

for img_embed, gt2D, bboxes, points, labels in train_dataloader:
    print(f"{img_embed.shape=}, {gt2D.shape=}, {bboxes.shape=}, {points.shape=}, {labels.shape=}")
    break

sam_model = None
end = time()
time_elapsed = end - start
t = f'Data reading time:**{time_elapsed // 60:.0f}m {time_elapsed % 60:.2f}s** or **{time_elapsed:.2f}s**'
print(t)

# ================================================train===================================================
from sam import sam_model_registry as adapter_sam_model_registry
import monai


seg_loss = Myloss(issigmoid=True)
net = adapter_sam_model_registry['vit_b'](checkpoint=checkpoint).to(device)
start_epoch = 0
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=0)

num_epochs = 20
loss_train = []
loss_val = []
best_loss = 1e10
for epoch in range(start_epoch, num_epochs):
    time_1 = time()
    epoch_loss_train = 0
    # train
    net.train()
    for step, (image_embed, gt2D, boxes, points, labels) in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()
        # just train prarameters of adapters
        for n, value in net.image_encoder.named_parameters():
            if "Adapter" not in n:
                value.requires_grad = False
        image_coded = net.image_encoder(image_embed.to(device))
        required_points = (points.to(device), labels.to(device))

        with torch.no_grad():
            # train with bbox prompts
            sparse_embeddings, dense_embeddings = net.prompt_encoder(
                points=None,
                boxes=boxes.to(device),
                masks=None,
            )
            # train with point prompts
            # sparse_embeddings, dense_embeddings = net.prompt_encoder(
            #     points=required_points,
            #     boxes=None,
            #     masks=None,
            # )
        low_res_masks, _ = net.mask_decoder(
            image_embeddings=image_coded.to(device),  # (B, 256, 64, 64)
            image_pe=net.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        upscaled_masks = net.postprocess_masks(low_res_masks, (1024, 1024), (128, 128)).to(device)
        loss = seg_loss(upscaled_masks, gt2D.to(device).float())
        loss.backward()
        optimizer.step()
        epoch_loss_train += loss.item()
    epoch_loss_train /= step
    loss_train.append(epoch_loss_train)

    # validation
    epoch_loss_val = 0
    if epoch % 1 == 0:
        net.eval()
        for step, (image_embed, gt2D, boxes, points, labels) in tqdm(enumerate(val_dataloader)):
            with torch.no_grad():
                image_coded = net.image_encoder(image_embed.to(device))
                required_points = (points.to(device), labels.to(device))
                sparse_embeddings, dense_embeddings = net.prompt_encoder(
                    points=None,
                    boxes=boxes.to(device),
                    masks=None,
                )
                # sparse_embeddings, dense_embeddings = net.prompt_encoder(
                #     points=required_points,
                #     boxes=None,
                #     masks=None,
                # )
                low_res_masks, _ = net.mask_decoder(
                    image_embeddings=image_coded.to(device),  # (B, 256, 64, 64)
                    image_pe=net.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                    sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                    dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                    multimask_output=False,
                )
                upscaled_masks = net.postprocess_masks(low_res_masks, (1024, 1024), (128, 128)).to(device)
                #         upscaled_masks = upscaled_masks > 0
                # print(upscaled_masks.shape, gt2D.shape)
                loss = seg_loss(upscaled_masks, gt2D.to(device).float())
            epoch_loss_val += loss.item()
        epoch_loss_val /= step
        loss_val.append(epoch_loss_val)
        print(f'EPOCH: {epoch}, Train Loss: {epoch_loss_train} , validation loss: {epoch_loss_val} , lr:{optimizer.param_groups[0]["lr"]}')
    else:
        print(f'EPOCH: {epoch}, Train Loss: {epoch_loss_train} , lr:{optimizer.param_groups[0]["lr"]}')

    # save the best model
    if epoch_loss_val < best_loss:
        best_loss = epoch_loss_val
        torch.save(net.state_dict(), 'adapter_sam_model.pth')
    time_2 = time()
    epoch_time = time_2 - time_1
    t = f'epoch time:**{epoch_time // 60:.0f}m {epoch_time % 60:.2f}s** or **{epoch_time:.2f}s**'
    print(t)

end = time()
time_elapsed = end - start
t = f'Training time consuming:**{time_elapsed // 60:.0f}m {time_elapsed % 60:.2f}s** or **{time_elapsed:.2f}s**'
print(t)
