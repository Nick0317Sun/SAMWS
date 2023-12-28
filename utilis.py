"""
    some functions used in adapter_finetune.py
"""
import numpy as np
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def convert_uint8(im):
    """
        preprocess image to uint8
    """
    mx = im.max(axis=(1, 2))
    mi = im.min(axis=(1, 2))
    im = (im - mi[:, None, None]) / (mx - mi)[:, None, None] * 255
    im = np.clip(im, a_max=255, a_min=0)
    im = np.uint8(im)
    return im.transpose(1, 2, 0)


def preprocess_input_image(img, sam_model, device='cuda:0'):
    """
        preprocess image to [3,1024,1024], which is the requirement of SAM
        inputting format [H, W, C]
    """
    # image embedding 3*1024*1024
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    image = transform.apply_image(img)
    image_torch = torch.as_tensor(image, device=device)
    transformed_image = image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    embedding = sam_model.preprocess(transformed_image)

    return embedding.cpu().numpy()[0]


def preprocess_input_bbox(bboxes, sam_model, img_H=128, img_W=128, device='cuda:0'):
    """
        预处理bbox,处理成模型需要的格式,可以批处理，输入为[B, 4]， 需要原始图片大小
        本质上应该是将坐标转化为了1024*1024中对应的坐标
    """
    # image embedding 3*1024*1024
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    box = transform.apply_boxes(bboxes, (img_H, img_W))
    input_boxes = torch.as_tensor(box, dtype=torch.float, device=device)
    if len(input_boxes.shape) == 2:
        input_boxes = input_boxes[:, None, :]  # (B, 1, 4)

    return input_boxes.cpu().numpy()


def preprocess_input_points(points, sam_model, img_H=128, img_W=128, device='cuda:0'):
    """
        process point to [b,2]
    """
    # 转化为image embedding 3*1024*1024
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    point_trans = transform.apply_coords(points, (img_H, img_W))
    point_torch = torch.as_tensor(point_trans, dtype=torch.float, device=device)
    if len(point_torch.shape) == 2:
        point_torch = point_torch[:, None, :]  # (B, 1, 2)
    point_labels = torch.ones((point_torch.shape[0], 1), dtype=torch.long)  # [b,1]

    return point_torch.cpu().numpy(), point_labels.cpu().numpy()


def get_bbox(img):
    """
        get the left bottom and right top coordinates
    """
    nonzero_pixels = np.argwhere(img == 1)

    min_coords = nonzero_pixels.min(axis=0)[::-1]
    max_coords = nonzero_pixels.max(axis=0)[::-1]

    return np.concatenate((min_coords, max_coords))


def get_point(img):
    """
        get point coordinate
    """
    coordinates = np.where(img == 1)

    x_coordinates = coordinates[0]
    y_coordinates = coordinates[1]
    #     return np.concatenate((x_coordinates, y_coordinates))
    return np.concatenate((y_coordinates, x_coordinates))


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_points(coords, ax, marker_size=375):
    ax.scatter(coords[0], coords[1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255 / 255, 0 / 255, 0 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def normalize_rgb(im):
    """Utility function to get a displayable rgb image
    from a Sentinel-2 time series.
    """
    mx = im.max(axis=(1, 2))
    mi = im.min(axis=(1, 2))
    im = (im - mi[:, None, None]) / (mx - mi)[:, None, None]
    im = im.swapaxes(0, 2).swapaxes(0, 1)
    im = np.clip(im, a_max=1, a_min=0)
    return im


class customedDS(Dataset):
    def __init__(self, img, bbox, mask, point, label):
        self.img = img
        self.bbox = bbox
        self.mask = mask
        self.point = point
        self.label = label

    def __len__(self):
        # return self.img.shape[0]
        return len(self.img)

    def __getitem__(self, index):
        img_embedding = self.img[index]
        gt_mask = self.mask[index]
        bbox = self.bbox[index]
        point = self.point[index]
        label = self.label[index]

        return torch.tensor(img_embedding), torch.tensor(gt_mask[None, :, :]), torch.tensor(bbox), torch.tensor(point), torch.tensor(label)
