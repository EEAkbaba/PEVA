from __future__ import annotations
import multiprocessing
import os
from typing import List
from pathlib import Path
from warnings import warn

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
from nnunetv2.configuration import default_num_processes


def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                    verify_npy: bool = False, fail_ctr: int = 0) -> None:
    data_npy = npz_file[:-3] + "npy"
    seg_npy = npz_file[:-4] + "_seg.npy"
    try:
        npz_content = None  # will only be opened on demand

        if overwrite_existing or not isfile(data_npy):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(data_npy, npz_content['data'])

        if unpack_segmentation and (overwrite_existing or not isfile(seg_npy)):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(npz_file[:-4] + "_seg.npy", npz_content['seg'])

        if verify_npy:
            try:
                np.load(data_npy, mmap_mode='r')
                if isfile(seg_npy):
                    np.load(seg_npy, mmap_mode='r')
            except ValueError:
                os.remove(data_npy)
                os.remove(seg_npy)
                print(f"Error when checking {data_npy} and {seg_npy}, fixing...")
                if fail_ctr < 2:
                    _convert_to_npy(npz_file, unpack_segmentation, overwrite_existing, verify_npy, fail_ctr+1)
                else:
                    raise RuntimeError("Unable to fix unpacking. Please check your system or rerun nnUNetv2_preprocess")

    except KeyboardInterrupt:
        if isfile(data_npy):
            os.remove(data_npy)
        if isfile(seg_npy):
            os.remove(seg_npy)
        raise KeyboardInterrupt


def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = default_num_processes,
                   verify_npy: bool = False):
    """
    all npz files in this folder belong to the dataset, unpack them all
    """
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        npz_files = subfiles(folder, True, None, ".npz", True)
        p.starmap(_convert_to_npy, zip(npz_files,
                                       [unpack_segmentation] * len(npz_files),
                                       [overwrite_existing] * len(npz_files),
                                       [verify_npy] * len(npz_files))
                  )

#----------------------------EEA
def get_case_identifiers(folder: str) -> List[str]:
    """
    finds all b2nd files in the given folder and reconstructs the training case names from them
    (compatible with nnUNet v2 blosc2 .b2nd format)
    """
    case_identifiers = []

    for fname in os.listdir(folder):
        # we only want data files, not segmentation files
        if fname.endswith(".b2nd") and not fname.endswith("_seg.b2nd") and ("segFromPrevStage" not in fname):
            # remove ".b2nd"
            case_identifiers.append(fname[:-5])

    case_identifiers.sort()
    return case_identifiers
#----------------------------EEA

def get_case_identifiers_org(folder: str) -> List[str]:
    """
    finds all npz files in the given folder and reconstructs the training case names from them
    """
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith("npz") and (i.find("segFromPrevStage") == -1)]
    return case_identifiers


if __name__ == '__main__':
    unpack_dataset('/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/2d')