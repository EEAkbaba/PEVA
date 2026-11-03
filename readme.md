# PEVA: Parameter-Efficient Volumetric Adaptation of 2D Foundation Models for Medical Image Segmentation

Official implementation of **"Parameter-Efficient Volumetric Adaptation of 2D Foundation Models for Medical Image Segmentation"**, submitted to ISBI 2027.

PEVA adapts pretrained 2D vision foundation models for volumetric medical image segmentation by introducing volumetric information into feature representations before slice-wise decoding.

The framework consists of two components:

- **Local Depth Fusion (LDF):** Incorporates information from neighboring slices using lightweight low-rank adapters.
- **Cross-View Fusion (CVF):** Integrates complementary axial, sagittal, and coronal representations through learned feature projections.

PEVA supports DINOv3 and MedDINOv3 encoders and is evaluated on BTCV (CT) and BraTS 2021 (MRI).

## Setup

The implementation builds on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and uses [DINOv3](https://github.com/facebookresearch/dinov3).

Install the project dependencies and configure the nnU-Net dataset paths using `setting_path.sh`. Download the required pretrained encoder weights separately and place them in the `pretrained_models/` directory.

Pretrained weights and datasets are not included in this repository.

## Training

Training uses the nnU-Net framework with the `3d_fullres` configuration.

Example command for PEVA with a frozen DINOv3 encoder:

```bash
export MODE=2D
export USE_LDF=True
export USE_CROSS_VIEW_FUSION=True
export FULL_FT=False

NNUNET_RUN_ID=PEVA_BTCV \
python -u -m nnunetv2.run.run_training 3 3d_fullres 0 \
  -tr nnUNetTrainer_DINOv3_multilayer \
  -p nnUNetPlans
```

The dataset ID is `3` for BTCV and `500` for BraTS 2021 in our experimental setup. Dataset preprocessing and nnU-Net plans must be prepared before training.

The full PEVA configuration uses a cosine projection alignment loss with weight λ = 0.5. Training is performed for 500 epochs with 250 iterations per epoch.

For full encoder fine-tuning, set `FULL_FT=True`. To disable individual adaptation components for ablation experiments, modify `USE_LDF` or `USE_CROSS_VIEW_FUSION`.

## Acknowledgments

This implementation builds upon nnU-Net and incorporates components from DINOv3. We acknowledge the authors of these projects and their contributions.

## Citation

Citation information will be updated following publication.
