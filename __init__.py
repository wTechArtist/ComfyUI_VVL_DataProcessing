"""
ComfyUI VVL Data Processing Plugin
数据处理工具集插件，包含点云处理、JSON合并、遮罩坐标生成等功能
"""

from .glb_point_cloud_processor import (
    GLBPointCloudProcessor, 
    GLBPointCloudBounds, 
    GLBPointCloudOriginAdjuster, 
    GLBPointCloudDensityFilter
)
from .json_merger import JSONMerger, UESceneGenerator
from .mask_to_coordinates import MaskToCoordinates

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "GLBPointCloudProcessor": GLBPointCloudProcessor,
    "GLBPointCloudBounds": GLBPointCloudBounds,
    "GLBPointCloudOriginAdjuster": GLBPointCloudOriginAdjuster,
    "GLBPointCloudDensityFilter": GLBPointCloudDensityFilter,
    "JSONMerger": JSONMerger,
    "UESceneGenerator": UESceneGenerator,
    "MaskToCoordinates": MaskToCoordinates,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "GLBPointCloudProcessor": "VVL GLB 点云黑色点清理",
    "GLBPointCloudBounds": "VVL GLB 点云包围盒计算",
    "GLBPointCloudOriginAdjuster": "VVL GLB 点云原点调整",
    "GLBPointCloudDensityFilter": "VVL GLB 点云密度过滤",
    "JSONMerger": "VVL JSON数据合并器",
    "UESceneGenerator": "VVL UE场景生成器",
    "MaskToCoordinates": "VVL Mask转坐标",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# 版本信息
__version__ = "1.0.0" 