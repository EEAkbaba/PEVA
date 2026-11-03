import pydoc
import warnings
from typing import Union

from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.custom_model.UNETR.UNETR import UNETR, UNETR_FREEZE
from nnunetv2.custom_model.DINO.DINO import DINOUNETR, DINOUNETR_ENC
from nnunetv2.custom_model.BeiT.BeiT import BeiTUNETR, BeiTUNETR_ENC
from nnunetv2.custom_model.BLIP.BLIP import BLIPUNETR, BLIPUNETR_ENC
from nnunetv2.custom_model.OpenClip.OpenClip import OpenClipUNETR, OpenClipUNETR_ENC
from nnunetv2.custom_model.SAM.SAM import SAMUNETR, SAMUNETR_ENC
from nnunetv2.custom_model.MedSAM.MedSAM import MedSAMUNETR, MedSAMUNETR_ENC



def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)

    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    if deep_supervision is not None:
            architecture_kwargs['deep_supervision'] = deep_supervision


    if network_class == "UNETR":
        print("Successfully modified network_class to UNETR")
        nw_class = UNETR
        network = nw_class(
            in_channels=input_channels,
            out_channels=output_channels,
            img_size=(32, 224, 224),
        )
    elif network_class == "UNETR_FREEZE":
        print("Successfully modified network_class to UNETR_FREEZE")
        nw_class = UNETR_FREEZE
        network = nw_class(
            in_channels=input_channels,
            out_channels=output_channels,
            img_size=(32, 224, 224),
        )
    elif network_class == "DINOUNETR":
        print("Successfully modified network_class to DINOUNETR")
        nw_class = DINOUNETR
        network = nw_class(
            input_dim=input_channels,
            output_dim=output_channels,
        )
    elif network_class == "DINOUNETR_ENC":
        print("Successfully modified network_class to DINOUNETR_ENC")
        nw_class = DINOUNETR_ENC
        network = nw_class(
            input_dim=input_channels,
            output_dim=output_channels,
        )
    elif network_class == "BeiTUNETR":
        print("Successfully modified network_class to BeiTUNETR")
        nw_class = BeiTUNETR
        network = nw_class(
            input_dim=input_channels,
            output_dim=output_channels,
        )
    elif network_class == "BeiTUNETR_ENC":
        print("Successfully modified network_class to BeiTUNETR_ENC")
        nw_class = BeiTUNETR_ENC
        network = nw_class(
            input_dim=input_channels,
            output_dim=output_channels,
        )
    elif network_class == "BLIPUNETR":
        print("Successfully modified network_class to BLIPUNETR")
        nw_class = BLIPUNETR
        network = nw_class(
            input_dim=input_channels,
            output_dim=output_channels,
        )
    elif network_class == "BLIPUNETR_ENC":
        print("Successfully modified network_class to BLIPUNETR_ENC")
        nw_class = BLIPUNETR_ENC
        network = nw_class(
            input_dim=input_channels,
            output_dim=output_channels,
        )
    elif network_class == "OpenClipUNETR":
        print("Successfully modified network_class to OpenClipUNETR")
        nw_class = OpenClipUNETR
        network = nw_class(
            input_dim=input_channels,
            output_dim=output_channels
        )
    elif network_class == "OpenClipUNETR_ENC":
        print("Successfully modified network_class to OpenClipUNETR_ENC")
        nw_class = OpenClipUNETR_ENC
        network = nw_class(
            input_dim=input_channels,
            output_dim=output_channels
        )
    elif network_class == "SAMUNETR":
        print("Successfully modified network_class to SAMUNETR")
        nw_class = SAMUNETR
        network = nw_class(
            input_dim=input_channels,
            output_dim=output_channels
        )
    elif network_class == "SAMUNETR_ENC":
        print("Successfully modified network_class to SAMUNETR_ENC")
        nw_class = SAMUNETR_ENC
        network = nw_class(
            input_dim=input_channels,
            output_dim=output_channels
        )
    elif network_class == "MEDSAMUNETR":
        print("Successfully modified network_class to MEDSAMUNETR")
        nw_class = MedSAMUNETR
        network = nw_class(
            input_dim=input_channels,
            output_dim=output_channels
        )
    elif network_class == "MEDSAMUNETR_ENC":
        print("Successfully modified network_class to MEDSAMUNETR_ENC")
        nw_class = MedSAMUNETR_ENC
        network = nw_class(
            input_dim=input_channels,
            output_dim=output_channels
        )
    else:
        nw_class = pydoc.locate(network_class)
        # sometimes things move around, this makes it so that we can at least recover some of that
        if nw_class is None:
            warnings.warn(f'Network class {network_class} not found. Attempting to locate it within '
                        f'dynamic_network_architectures.architectures...')
            import dynamic_network_architectures
            nw_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                                network_class.split(".")[-1],
                                                'dynamic_network_architectures.architectures')
            if nw_class is not None:
                print(f'FOUND IT: {nw_class}')
            else:
                raise ImportError('Network class could not be found, please check/correct your plans file')

        network = nw_class(
            input_channels=input_channels,
            num_classes=output_channels,
            **architecture_kwargs
        )



    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network

if __name__ == "__main__":
    import torch

    model = get_network_from_plans(
        arch_class_name="dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
        arch_kwargs={
            "n_stages": 7,
            "features_per_stage": [32, 64, 128, 256, 512, 512, 512],
            "conv_op": "torch.nn.modules.conv.Conv2d",
            "kernel_sizes": [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
            "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
            "n_blocks_per_stage": [1, 3, 4, 6, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1, 1],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
        input_channels=1,
        output_channels=4,
        allow_init=True,
        deep_supervision=True,
    )
    data = torch.rand((8, 1, 256, 256))
    target = torch.rand(size=(8, 1, 256, 256))
    outputs = model(data) # this should be a list of torch.Tensor