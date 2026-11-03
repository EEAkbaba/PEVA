from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.custom_model.DINO.DINO import MedDINOv3_PEVA_multilayer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class nnUNetTrainer_MedDINOv3_PEVA_multilayer(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name,
                                   arch_init_kwargs,
                                   arch_init_kwargs_req_import,
                                   num_input_channels,
                                   num_output_channels,
                                   enable_deep_supervision=True):
        return MedDINOv3_PEVA_multilayer(
            embed_dim=768,
            patch_size=16,
            input_dim=num_input_channels,
            output_dim=num_output_channels
        )