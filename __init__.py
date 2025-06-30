"""
ComfyUI VVL Data Processing Plugin
数据处理工具集插件，包含点云处理、JSON合并、遮罩坐标生成、视频处理等功能
"""

from .glb_point_cloud_processor.glb_point_cloud_black_delete import *
from .glb_point_cloud_processor.glb_point_cloud_bounds import *
from .glb_point_cloud_processor.glb_point_cloud_origin_adjuster import *
from .glb_point_cloud_processor.glb_point_cloud_density_filter import *
from .glb_point_cloud_processor.glb_point_cloud_rotation_corrector import *
from .glb_point_cloud_processor.glb_point_cloud_transform_applier import *
from .glb_point_cloud_processor.glb_point_cloud_rotator import *
from .json_merger import JSONMerger, UESceneGenerator
from .mask_to_coordinates import MaskToCoordinates
from .video_processor.save_video import SaveVideo

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "GLBPointCloudBlackDelete": GLBPointCloudBlackDelete,
    "GLBPointCloudBounds": GLBPointCloudBounds,
    "GLBPointCloudOriginAdjuster": GLBPointCloudOriginAdjuster,
    "GLBPointCloudDensityFilter": GLBPointCloudDensityFilter,
    "GLBPointCloudRotationCorrector": GLBPointCloudRotationCorrector,
    "GLBPointCloudTransformApplier": GLBPointCloudTransformApplier,
    "GLBPointCloudRotator": GLBPointCloudRotator,
    "JSONMerger": JSONMerger,
    "UESceneGenerator": UESceneGenerator,
    "MaskToCoordinates": MaskToCoordinates,
    "VVL_SaveVideo": SaveVideo,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "GLBPointCloudBlackDelete": "VVL GLB 点云黑色点清理",
    "GLBPointCloudBounds": "VVL GLB 点云包围盒计算",
    "GLBPointCloudOriginAdjuster": "VVL GLB 点云原点调整",
    "GLBPointCloudDensityFilter": "VVL GLB 点云密度过滤",
    "GLBPointCloudRotationCorrector": "VVL GLB 点云旋转校正",
    "GLBPointCloudTransformApplier": "VVL GLB 点云变换应用",
    "GLBPointCloudRotator": "VVL GLB 点云旋转器",
    "JSONMerger": "VVL JSON数据合并器",
    "UESceneGenerator": "VVL UE场景生成器",
    "MaskToCoordinates": "VVL Mask转坐标",
    "VVL_SaveVideo": "VVL 视频保存器",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# 版本信息
__version__ = "1.0.0"