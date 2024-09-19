# SAMWS
This is the repository for "A Segment Anything Model based weakly supervised learning method for crop mapping using Sentinel-2 time series images".

## Finetuning with adapters
Download SAM [checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints), we used [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth).

The main finetuning code is in adapter_finetune.py. Some example data is provided in the repository.

## Generating pseudo labels
The codes of generating pseudo labels by different weak annotations lie in pseudoLabels_generate_XXX.py.
Our image classification model--attention-based U-Net is in cls_model directory.

## Training a segmentation model
The segmentation model this study uses is [U-TAE](https://github.com/VSainteuf/utae-paps).

## Citation
```
@article{sun2024segment,
  title={A Segment Anything Model based weakly supervised learning method for crop mapping using Sentinel-2 time series images},
  author={Sun, Jialin and Yan, Shuai and Yao, Xiaochuang and Gao, Bingbo and Yang, Jianyu},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={133},
  pages={104085},
  year={2024},
  publisher={Elsevier}
}
```

## Acknowledgement
Spectial thanks to scholars from [Meta](https://github.com/facebookresearch/segment-anything) and [Junde Wu](https://github.com/KidsWithTokens/Medical-SAM-Adapter), as well as his fellows, for their implementation of adapters in SAM.
