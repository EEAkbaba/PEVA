###################
# Written by EEA
###################

from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

if __name__ == '__main__':
    """
    BTCV (Synapse) -> nnU-Net v2 raw format conversion

    1) Run this script to convert:
       BTCV_RawData/Training/img/img0001.nii.gz
       BTCV_RawData/Training/label/label0001.nii.gz
       into:
       nnUNet_raw/Dataset003_Synapse/imagesTr/synapse_0001_0000.nii.gz
       nnUNet_raw/Dataset003_Synapse/labelsTr/synapse_0001.nii.gz

    2) Then run planning & preprocessing:
       nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity
       nnUNetv2_plan_and_preprocess -d 3
    """

    # --------- CHANGED: paths & dataset id/name ----------
    base = '/mnt/storage1/elif/Medical3D/BTCV_RawData'
    img_dir = join(base, 'Training', 'img')
    lbl_dir = join(base, 'Training', 'label')

    target_dataset_id = 3
    target_dataset_name = f'Dataset{target_dataset_id:03d}_Synapse'

    raw_dir = '/mnt/storage1/elif/Medical3D/nnUNet_raw'
    # -----------------------------------------------------

    # Create target folders
    maybe_mkdir_p(join(raw_dir, target_dataset_name))
    imagesTr = join(raw_dir, target_dataset_name, 'imagesTr')
    labelsTr = join(raw_dir, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    # --------- CHANGED: collect cases from img_dir ----------
    # We expect files like img0001.nii.gz in Training/img
    img_files = subfiles(img_dir, join=False, suffix='.nii.gz', prefix='img')
    img_files.sort()
    # Extract numeric ids like 0001, 0002...
    case_ids = [f.replace('img', '').replace('.nii.gz', '') for f in img_files]
    # --------------------------------------------------------

    # --------- OPTIONAL SAFETY: refuse overwrite if already filled ----------
    # If you want to allow re-run, comment these two lines out.
    if len(subfiles(imagesTr, suffix='.nii.gz')) > 0 or len(subfiles(labelsTr, suffix='.nii.gz')) > 0:
        raise RuntimeError(
            f"Destination not empty:\n  {imagesTr}\n  {labelsTr}\n"
            "Refusing to overwrite. Empty these folders or comment out the safety check."
        )
    # ----------------------------------------------------------------------

    # Copy files with nnU-Net naming
    for cid in case_ids:
        src_img = join(img_dir, f'img{cid}.nii.gz')
        src_lbl = join(lbl_dir, f'label{cid}.nii.gz')

        # --------- CHANGED: output naming ----------
        out_case = f'btcv_{cid}'
        dst_img = join(imagesTr, out_case + '_0000.nii.gz')
        dst_lbl = join(labelsTr, out_case + '.nii.gz')
        # -------------------------------------------

        if not isfile(src_img):
            raise FileNotFoundError(src_img)
        if not isfile(src_lbl):
            raise FileNotFoundError(src_lbl)

        shutil.copy(src_img, dst_img)
        shutil.copy(src_lbl, dst_lbl)

    # --------- CHANGED: labels mapping for BTCV/Synapse (13 organs + background) ----------
    labels = {
        "background": 0,
        "spleen": 1,
        "right_kidney": 2,
        "left_kidney": 3,
        "gallbladder": 4,
        "esophagus": 5,
        "liver": 6,
        "stomach": 7,
        "aorta": 8,
        "inferior_vena_cava": 9,
        "portal_vein_and_splenic_vein": 10,
        "pancreas": 11,
        "right_adrenal_gland": 12,
        "left_adrenal_gland": 13
    }
    # ------------------------------------------------------------------------------------

    # --------- CHANGED: channel name (CT) + dataset.json output folder ----------
    generate_dataset_json(
        join(raw_dir, target_dataset_name),
        {0: 'CT'},
        labels,
        len(case_ids),
        '.nii.gz',
        None,
        target_dataset_name,
        overwrite_image_reader_writer='NibabelIOWithReorient'
    )
    # -------------------------------------------------------------------------

    print(f"Done. Converted {len(case_ids)} cases to: {join(raw_dir, target_dataset_name)}")