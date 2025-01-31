# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import torch
import torch.nn as nn

from .nn import (
    ConvLayer,
    DSConv,
    EfficientViTBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
    ResidualBlock,
)
# from efficientvit.models.utils import build_kwargs_from_config
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from .registry import register_backbone
import os

__all__ = [
    "EfficientViTBackbone",
    "efficientvit_backbone_b0",
    "efficientvit_backbone_b1",
    "efficientvit_backbone_b2",
    "efficientvit_backbone_b3",
    "EfficientViTLargeBackbone",
    "efficientvit_backbone_l0",
    "efficientvit_backbone_l1",
    "efficientvit_backbone_l2",
    "efficientvit_backbone_l3",
]


class EfficientViTBackbone(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
    ) -> None:
        super().__init__()

        self.width_list = []
        # input stem
        self.input_stem = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # stages
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_dict = {"input": x}
        output_dict["stage0"] = x = self.input_stem(x)
        for stage_id, stage in enumerate(self.stages, 1):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


def efficientvit_backbone_b0(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b1(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b2(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


def efficientvit_backbone_b3(**kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    return backbone


class EfficientViTLargeBackbone(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        block_list: list[str] or None = None,
        expand_list: list[float] or None = None,
        fewer_norm_list: list[bool] or None = None,
        in_channels=3,
        qkv_dim=32,
        norm="bn2d",
        act_func="gelu",
    ) -> None:
        super().__init__()
        block_list = block_list or ["res", "fmb", "fmb", "mb", "att"]
        expand_list = expand_list or [1, 4, 4, 4, 6]
        fewer_norm_list = fewer_norm_list or [False, False, False, True, True]

        self.width_list = []
        self.stages = []
        # stage 0
        stage0 = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                block=block_list[0],
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=expand_list[0],
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[0],
            )
            stage0.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.stages.append(OpSequential(stage0))
        self.width_list.append(in_channels)

        for stage_id, (w, d) in enumerate(zip(width_list[1:], depth_list[1:]), start=1):
            stage = []
            block = self.build_local_block(
                block="mb" if block_list[stage_id] not in ["mb", "fmb"] else block_list[stage_id],
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_list[stage_id] * 4,
                norm=norm,
                act_func=act_func,
                fewer_norm=fewer_norm_list[stage_id],
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                if block_list[stage_id].startswith("att"):
                    stage.append(
                        EfficientViTBlock(
                            in_channels=in_channels,
                            dim=qkv_dim,
                            expand_ratio=expand_list[stage_id],
                            scales=(3,) if block_list[stage_id] == "att@3" else (5,),
                            norm=norm,
                            act_func=act_func,
                        )
                    )
                else:
                    block = self.build_local_block(
                        block=block_list[stage_id],
                        in_channels=in_channels,
                        out_channels=in_channels,
                        stride=1,
                        expand_ratio=expand_list[stage_id],
                        norm=norm,
                        act_func=act_func,
                        fewer_norm=fewer_norm_list[stage_id],
                    )
                    block = ResidualBlock(block, IdentityLayer())
                    stage.append(block)
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        block: str,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if block == "res":
            block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "fmb":
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        elif block == "mb":
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        else:
            raise ValueError(block)
        return block

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_dict = {"input": x}
        for stage_id, stage in enumerate(self.stages):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict


def efficientvit_backbone_l0(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 4, 4],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l1(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 1, 1, 6, 6],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l2(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


def efficientvit_backbone_l3(**kwargs) -> EfficientViTLargeBackbone:
    backbone = EfficientViTLargeBackbone(
        width_list=[64, 128, 256, 512, 1024],
        depth_list=[1, 2, 2, 8, 8],
        **build_kwargs_from_config(kwargs, EfficientViTLargeBackbone),
    )
    return backbone


@BACKBONE_REGISTRY.register()
class D2EfficientViT(EfficientViTLargeBackbone, Backbone):
    def __init__(self, cfg, input_shape):
        # width_list=[32, 64, 128, 256, 512],
        # depth_list=[1, 2, 2, 8, 8],
        width_list = cfg.MODEL.EFFIVIT.WIDTH_LIST
        depth_list = cfg.MODEL.EFFIVIT.DEPTH_LIST

        super().__init__(
            width_list=width_list,
            depth_list=depth_list
        )

        weights = load_state_dict_from_file(cfg.MODEL.EFFIVIT.WEIGHTS)
        weights = preprocess_ckpt(weights)
        self.load_state_dict(weights)

        self._out_features = cfg.MODEL.EFFIVIT.OUT_FEATURES

        self.num_features = width_list[1:]
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }
        self.map_dict = {"res2": "stage1", "res3": "stage2", "res4": "stage3", "res5": "stage4"}

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        
        for k_o in self._out_features:
            k_i = self.map_dict[k_o]
            outputs[k_o] = y[k_i]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32


@BACKBONE_REGISTRY.register()
class D2EfficientViTBase(EfficientViTBackbone, Backbone):
    def __init__(self, cfg, input_shape):
        # width_list=[32, 64, 128, 256, 512],
        # depth_list=[1, 2, 2, 8, 8],
        width_list = cfg.MODEL.EFFIVIT.WIDTH_LIST
        depth_list = cfg.MODEL.EFFIVIT.DEPTH_LIST
        dim = cfg.MODEL.EFFIVIT.DIM

        super().__init__(
            width_list=width_list,
            depth_list=depth_list,
            dim=dim
        )

        weights = load_state_dict_from_file(cfg.MODEL.EFFIVIT.WEIGHTS)
        weights = preprocess_ckpt(weights)
        self.load_state_dict(weights)

        self._out_features = cfg.MODEL.EFFIVIT.OUT_FEATURES

        self.num_features = width_list[1:]
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }
        self.map_dict = {"res2": "stage1", "res3": "stage2", "res4": "stage3", "res5": "stage4"}

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        
        for k_o in self._out_features:
            k_i = self.map_dict[k_o]
            outputs[k_o] = y[k_i]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32


def load_state_dict_from_file(file: str, only_state_dict=True) -> dict[str, torch.Tensor]:
    file = os.path.realpath(os.path.expanduser(file))
    checkpoint = torch.load(file, map_location="cpu")
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint


def preprocess_ckpt(ckpt):
    from copy import deepcopy
    ckpt_new = deepcopy(ckpt)
    for k, v in ckpt.items():
        if k.startswith("backbone."):
            k_new = k.replace("backbone.", "")
            ckpt_new.pop(k)
            ckpt_new[k_new] = v
        if k.startswith("head."):
            ckpt_new.pop(k)
    return ckpt_new
