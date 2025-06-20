"""
Mask To Coordinates Node
从 mask 生成 coordinates_positive 和 coordinates_negative 坐标信息
"""

import torch
import numpy as np
import json
import cv2
from typing import Tuple, List, Optional


class MaskToCoordinates:
    """
    从 mask 生成坐标信息的节点
    白色区域 (值 >= threshold) 生成 coordinates_positive
    黑色区域 (值 < threshold) 生成 coordinates_negative
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {
                    "tooltip": "输入的遮罩图像。白色区域将生成正样本坐标，黑色区域将生成负样本坐标。"
                }),
                "sample_method": (["random", "grid", "contour"], {
                    "default": "random",
                    "tooltip": "坐标采样方法：\n• random - 随机采样，在有效区域内随机选择点\n• grid - 网格采样，按网格模式均匀分布采样点\n• contour - 轮廓采样，沿着区域轮廓边缘采样"
                }),
                "positive_points": ("INT", {
                    "default": 10, 
                    "min": 1, 
                    "max": 1000, 
                    "step": 1,
                    "tooltip": "从白色区域(值≥阈值)生成的正样本坐标点数量。这些坐标将作为 coordinates_positive 输出到 SAM2 分割节点。"
                }),
                "negative_points": ("INT", {
                    "default": 10, 
                    "min": 0, 
                    "max": 1000, 
                    "step": 1,
                    "tooltip": "从黑色区域(值<阈值)生成的负样本坐标点数量。这些坐标将作为 coordinates_negative 输出到 SAM2 分割节点。设为0则不生成负样本点。"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "二值化阈值，用于区分正负样本区域。像素值≥此阈值的区域被视为白色(正样本)，<此阈值的区域被视为黑色(负样本)。"
                }),
            },
            "optional": {
                "min_distance": ("INT", {
                    "default": 10, 
                    "min": 1, 
                    "max": 100, 
                    "step": 1,
                    "tooltip": "采样点之间的最小距离(像素)。用于避免点过于密集，确保采样点分布更均匀。仅在random和contour采样方法中有效。"
                }),
                "edge_margin": ("INT", {
                    "default": 5, 
                    "min": 0, 
                    "max": 50, 
                    "step": 1,
                    "tooltip": "图像边缘留白像素数。在图像边缘指定像素范围内不会生成采样点，避免边界效应影响分割效果。设为0则不留白。"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("coordinates_positive", "coordinates_negative", "visualization")
    FUNCTION = "mask_to_coordinates"
    CATEGORY = "💃VVL/Mask Processing"
    DESCRIPTION = "从 mask 生成正负坐标信息，用于 SAM2 分割"

    def mask_to_coordinates(
        self, 
        mask: torch.Tensor, 
        sample_method: str = "random",
        positive_points: int = 10, 
        negative_points: int = 10, 
        threshold: float = 0.5,
        min_distance: int = 10,
        edge_margin: int = 5
    ) -> Tuple[str, str, torch.Tensor]:
        """
        从 mask 生成坐标信息
        
        Args:
            mask: 输入的 mask 张量
            sample_method: 采样方法 ("random", "grid", "contour")
            positive_points: 正样本点数量
            negative_points: 负样本点数量
            threshold: 二值化阈值
            min_distance: 点之间最小距离
            edge_margin: 边缘留白
            
        Returns:
            coordinates_positive: 正坐标 JSON 字符串
            coordinates_negative: 负坐标 JSON 字符串
            visualization: 可视化图像
        """
        
        # 处理 mask 张量
        if mask.dim() == 3:
            mask = mask[0]  # 取第一帧
        elif mask.dim() == 4:
            mask = mask[0, 0]  # 取第一帧第一通道
        
        # 转换为 numpy 数组
        mask_np = mask.cpu().numpy()
        height, width = mask_np.shape
        
        # 二值化处理
        positive_mask = (mask_np >= threshold).astype(np.uint8)
        negative_mask = (mask_np < threshold).astype(np.uint8)
        
        # 应用边缘留白
        if edge_margin > 0:
            positive_mask = self._apply_edge_margin(positive_mask, edge_margin)
            negative_mask = self._apply_edge_margin(negative_mask, edge_margin)
        
        # 生成坐标点
        positive_coords = self._sample_points(
            positive_mask, positive_points, sample_method, min_distance
        )
        negative_coords = self._sample_points(
            negative_mask, negative_points, sample_method, min_distance
        )
        
        # 转换为 JSON 格式
        coordinates_positive = json.dumps([
            {"x": int(x), "y": int(y)} for x, y in positive_coords
        ])
        coordinates_negative = json.dumps([
            {"x": int(x), "y": int(y)} for x, y in negative_coords
        ]) if negative_coords else json.dumps([])
        
        # 生成可视化图像
        visualization = self._create_visualization(
            mask_np, positive_coords, negative_coords, threshold
        )
        
        return coordinates_positive, coordinates_negative, visualization
    
    def _apply_edge_margin(self, mask: np.ndarray, margin: int) -> np.ndarray:
        """应用边缘留白"""
        if margin <= 0:
            return mask
        
        result = mask.copy()
        result[:margin, :] = 0
        result[-margin:, :] = 0
        result[:, :margin] = 0
        result[:, -margin:] = 0
        return result
    
    def _sample_points(
        self, 
        mask: np.ndarray, 
        num_points: int, 
        method: str, 
        min_distance: int
    ) -> List[Tuple[int, int]]:
        """从 mask 中采样点"""
        
        if num_points <= 0:
            return []
        
        # 找到所有有效像素位置
        valid_positions = np.where(mask > 0)
        if len(valid_positions[0]) == 0:
            return []
        
        valid_coords = list(zip(valid_positions[1], valid_positions[0]))  # (x, y)
        
        if len(valid_coords) == 0:
            return []
        
        if method == "random":
            return self._random_sampling(valid_coords, num_points, min_distance)
        elif method == "grid":
            return self._grid_sampling(mask, num_points)
        elif method == "contour":
            return self._contour_sampling(mask, num_points, min_distance)
        else:
            return self._random_sampling(valid_coords, num_points, min_distance)
    
    def _random_sampling(
        self, 
        valid_coords: List[Tuple[int, int]], 
        num_points: int, 
        min_distance: int
    ) -> List[Tuple[int, int]]:
        """随机采样"""
        if not valid_coords:
            return []
        
        selected_points = []
        available_coords = valid_coords.copy()
        
        for _ in range(min(num_points, len(available_coords))):
            if not available_coords:
                break
            
            # 随机选择一个点
            idx = np.random.randint(len(available_coords))
            point = available_coords[idx]
            selected_points.append(point)
            
            # 移除距离太近的点
            if min_distance > 0:
                available_coords = [
                    coord for coord in available_coords
                    if np.sqrt((coord[0] - point[0])**2 + (coord[1] - point[1])**2) >= min_distance
                ]
            else:
                available_coords.remove(point)
        
        return selected_points
    
    def _grid_sampling(self, mask: np.ndarray, num_points: int) -> List[Tuple[int, int]]:
        """网格采样"""
        height, width = mask.shape
        
        # 计算网格大小
        grid_size = int(np.sqrt(num_points))
        if grid_size == 0:
            grid_size = 1
        
        points = []
        step_y = height // (grid_size + 1)
        step_x = width // (grid_size + 1)
        
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                y = i * step_y
                x = j * step_x
                
                # 检查该位置是否有效
                if y < height and x < width and mask[y, x] > 0:
                    points.append((x, y))
                    
                if len(points) >= num_points:
                    break
            if len(points) >= num_points:
                break
        
        return points
    
    def _contour_sampling(
        self, 
        mask: np.ndarray, 
        num_points: int, 
        min_distance: int
    ) -> List[Tuple[int, int]]:
        """轮廓采样"""
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # 选择最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < num_points:
            # 如果轮廓点数不够，返回所有轮廓点
            return [(int(point[0][0]), int(point[0][1])) for point in largest_contour]
        
        # 均匀采样轮廓点
        step = len(largest_contour) // num_points
        sampled_points = []
        
        for i in range(0, len(largest_contour), step):
            if len(sampled_points) >= num_points:
                break
            point = largest_contour[i][0]
            sampled_points.append((int(point[0]), int(point[1])))
        
        return sampled_points
    
    def _create_visualization(
        self, 
        mask: np.ndarray, 
        positive_coords: List[Tuple[int, int]], 
        negative_coords: List[Tuple[int, int]], 
        threshold: float
    ) -> torch.Tensor:
        """创建可视化图像"""
        height, width = mask.shape
        
        # 创建彩色图像
        vis_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 绘制 mask 背景
        mask_norm = (mask * 255).astype(np.uint8)
        vis_image[:, :, 0] = mask_norm  # 红色通道显示 mask
        vis_image[:, :, 1] = mask_norm  # 绿色通道显示 mask
        vis_image[:, :, 2] = mask_norm  # 蓝色通道显示 mask
        
        # 绘制正样本点 (绿色)
        for x, y in positive_coords:
            cv2.circle(vis_image, (x, y), 3, (0, 255, 0), -1)
            cv2.circle(vis_image, (x, y), 5, (0, 255, 0), 1)
        
        # 绘制负样本点 (红色)
        for x, y in negative_coords:
            cv2.circle(vis_image, (x, y), 3, (0, 0, 255), -1)
            cv2.circle(vis_image, (x, y), 5, (0, 0, 255), 1)
        
        # 转换为 torch 张量
        vis_tensor = torch.from_numpy(vis_image).float() / 255.0
        vis_tensor = vis_tensor.unsqueeze(0)  # 添加 batch 维度
        
        return vis_tensor


# 节点类映射
NODE_CLASS_MAPPINGS = {
    "MaskToCoordinates": MaskToCoordinates,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskToCoordinates": "VVL Mask转坐标",
} 