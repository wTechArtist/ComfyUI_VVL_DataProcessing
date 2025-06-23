from .common import *


class GLBPointCloudRotator:
    """GLB点云旋转器 - 围绕XYZ任意轴旋转任意角度"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_file_path": ("STRING", {
                    "default": "",
                    "tooltip": "GLB点云文件路径：输入需要旋转的GLB格式点云文件"
                }),
                "rotation_axis": (["X", "Y", "Z"], {
                    "default": "Y",
                    "tooltip": "旋转轴：选择围绕哪个轴进行旋转。X=绕X轴旋转(俯仰)；Y=绕Y轴旋转(偏航)；Z=绕Z轴旋转(翻滚)"
                }),
                "rotation_angle": ("FLOAT", {
                    "default": 90.0, "min": -360.0, "max": 360.0, "step": 0.1,
                    "tooltip": "旋转角度：旋转的角度，单位为度。正值为逆时针旋转，负值为顺时针旋转"
                }),
                "output_filename": ("STRING", {
                    "default": "rotated_pointcloud",
                    "tooltip": "输出GLB文件名：旋转后点云的保存文件名，系统会自动添加.glb扩展名"
                }),
            },
            "optional": {
                "rotation_center": (["origin", "bbox_center"], {
                    "default": "origin",
                    "tooltip": "旋转中心：origin=原点(0,0,0)；bbox_center=点云包围盒中心"
                }),
                "preserve_original_transform": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "保留原始变换：是否保留模型原有的变换矩阵，然后在此基础上应用旋转"
                }),
            }
        }

    RETURN_TYPES = (
        "STRING",    # 旋转后的GLB文件路径
        "STRING",    # 旋转变换信息JSON
    )
    RETURN_NAMES = (
        "rotated_glb_path",
        "rotation_info",
    )
    OUTPUT_TOOLTIPS = [
        "旋转后的GLB文件完整路径",
        "旋转变换信息JSON：包含旋转轴、角度、变换矩阵等信息",
    ]
    OUTPUT_NODE = True
    FUNCTION = "rotate_pointcloud"
    CATEGORY = "💃VVL/Point Cloud Transform"

    def rotate_pointcloud(self,
                         glb_file_path: str,
                         rotation_axis: str = "Y",
                         rotation_angle: float = 0.0,
                         output_filename: str = "rotated_pointcloud",
                         rotation_center: str = "origin",
                         preserve_original_transform: bool = True):
        """
        围绕指定轴旋转GLB点云指定角度
        """
        
        processing_log = []
        processing_log.append("开始GLB点云旋转...")
        
        # 检查依赖
        if not TRIMESH_AVAILABLE:
            error_msg = "trimesh库不可用，无法处理GLB文件"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("", "")
        
        # 验证输入文件路径
        if not glb_file_path or not glb_file_path.strip():
            error_msg = "GLB文件路径为空"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("", "")
        
        # 处理文件路径
        input_path = self._resolve_file_path(glb_file_path.strip())
        processing_log.append(f"输入文件路径: {input_path}")
        
        if not os.path.exists(input_path):
            error_msg = f"GLB文件不存在: {input_path}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("", "")
        
        try:
            # 加载GLB文件
            processing_log.append("正在加载GLB文件...")
            scene = trimesh.load(input_path)
            
            # 收集所有顶点信息（用于计算包围盒中心）
            all_vertices = []
            geometries_info = []
            
            if isinstance(scene, trimesh.Scene):
                for node_name in scene.graph.nodes:
                    if node_name in scene.geometry:
                        geometry = scene.geometry[node_name]
                        transform_matrix = scene.graph[node_name][0]
                        geometries_info.append((node_name, geometry, transform_matrix))
                        
                        if hasattr(geometry, 'vertices') and geometry.vertices is not None:
                            # 应用变换矩阵获取世界坐标
                            if transform_matrix is not None and not np.allclose(transform_matrix, np.eye(4)):
                                world_vertices = trimesh.transformations.transform_points(geometry.vertices, transform_matrix)
                            else:
                                world_vertices = geometry.vertices.copy()
                            
                            all_vertices.append(world_vertices)
                            processing_log.append(f"发现几何体: {node_name}, 顶点数: {len(geometry.vertices)}")
            else:
                # 单个几何体
                geometries_info.append(("main_geometry", scene, np.eye(4)))
                if hasattr(scene, 'vertices') and scene.vertices is not None:
                    all_vertices.append(scene.vertices)
                    processing_log.append(f"发现几何体: main_geometry, 顶点数: {len(scene.vertices)}")
            
            if not all_vertices:
                error_msg = "GLB文件中没有可用的顶点数据"
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
            
            # 合并所有顶点
            combined_vertices = np.vstack(all_vertices)
            total_points = len(combined_vertices)
            processing_log.append(f"总顶点数: {total_points:,}")
            
            # 计算旋转中心
            if rotation_center == "bbox_center":
                bbox_min = np.min(combined_vertices, axis=0)
                bbox_max = np.max(combined_vertices, axis=0)
                center = (bbox_min + bbox_max) / 2.0
                processing_log.append(f"使用包围盒中心作为旋转中心: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
            else:
                center = np.array([0.0, 0.0, 0.0])
                processing_log.append(f"使用原点作为旋转中心: [0, 0, 0]")
            
            # 计算旋转变换
            processing_log.append(f"计算旋转变换: 围绕{rotation_axis}轴旋转{rotation_angle:.2f}度")
            rotation_matrix, transform_matrix = self._calculate_rotation_matrix(
                rotation_axis, rotation_angle, center, processing_log
            )
            
            # 应用旋转变换到所有几何体
            processing_log.append("应用旋转变换...")
            rotated_scene = self._apply_rotation_transform(
                scene, geometries_info, transform_matrix, preserve_original_transform, processing_log
            )
            
            # 生成输出路径并保存
            output_path = self._generate_output_path(output_filename)
            processing_log.append(f"输出文件路径: {output_path}")
            
            # 保存旋转后的GLB文件
            processing_log.append("正在保存旋转后的GLB文件...")
            rotated_scene.export(output_path)
            
            if not os.path.exists(output_path):
                error_msg = "输出文件保存失败"
                logger.error(error_msg)
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
            
            file_size = os.path.getsize(output_path)
            processing_log.append(f"GLB文件保存成功，大小: {file_size:,} bytes")
            
            # 生成旋转信息JSON
            rotation_info = {
                "rotation_axis": rotation_axis,
                "rotation_angle_degrees": rotation_angle,
                "rotation_angle_radians": np.radians(rotation_angle),
                "rotation_center": center.tolist(),
                "rotation_matrix": rotation_matrix.tolist(),
                "transform_matrix": transform_matrix.tolist(),
                "preserve_original_transform": preserve_original_transform,
                "input_file": input_path,
                "output_file": output_path,
                "total_vertices": total_points,
                "processing_log": processing_log,
                "timestamp": time.time()
            }
            
            processing_log.append("GLB点云旋转完成!")
            logger.info("GLB点云旋转成功")
            
            return (output_path, json.dumps(rotation_info, indent=2))
            
        except Exception as e:
            error_msg = f"GLB点云旋转失败: {str(e)}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("", "")

    def _calculate_rotation_matrix(self, axis: str, angle_degrees: float, center: np.ndarray, processing_log: List[str]):
        """计算旋转矩阵"""
        
        angle_radians = np.radians(angle_degrees)
        processing_log.append(f"旋转角度: {angle_degrees:.2f}度 = {angle_radians:.4f}弧度")
        
        # 根据轴创建旋转矩阵
        if axis == "X":
            # 绕X轴旋转
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle_radians), -np.sin(angle_radians)],
                [0, np.sin(angle_radians), np.cos(angle_radians)]
            ])
        elif axis == "Y":
            # 绕Y轴旋转
            rotation_matrix = np.array([
                [np.cos(angle_radians), 0, np.sin(angle_radians)],
                [0, 1, 0],
                [-np.sin(angle_radians), 0, np.cos(angle_radians)]
            ])
        elif axis == "Z":
            # 绕Z轴旋转
            rotation_matrix = np.array([
                [np.cos(angle_radians), -np.sin(angle_radians), 0],
                [np.sin(angle_radians), np.cos(angle_radians), 0],
                [0, 0, 1]
            ])
        else:
            raise ValueError(f"不支持的旋转轴: {axis}")
        
        # 创建4x4齐次变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        
        # 如果旋转中心不是原点，需要先平移到原点，旋转，再平移回去
        if not np.allclose(center, 0):
            # T = T_back * R * T_to_origin
            # T_to_origin: 平移到原点
            T_to_origin = np.eye(4)
            T_to_origin[:3, 3] = -center
            
            # T_back: 平移回原位置
            T_back = np.eye(4)
            T_back[:3, 3] = center
            
            # 组合变换
            transform_matrix = T_back @ transform_matrix @ T_to_origin
            processing_log.append(f"应用了平移补偿，旋转中心: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
        
        return rotation_matrix, transform_matrix

    def _apply_rotation_transform(self, scene, geometries_info, transform_matrix, preserve_original_transform, processing_log):
        """应用旋转变换到场景中的所有几何体"""
        
        rotated_scene = trimesh.Scene()
        
        if isinstance(scene, trimesh.Scene):
            for node_name, geometry, original_transform in geometries_info:
                new_geometry = geometry.copy()
                
                if hasattr(new_geometry, 'vertices') and new_geometry.vertices is not None:
                    vertices = geometry.vertices.copy()
                    
                    if preserve_original_transform and original_transform is not None and not np.allclose(original_transform, np.eye(4)):
                        # 保留原始变换：先应用原始变换，再应用旋转
                        if original_transform.shape == (3, 4):
                            full_original_transform = np.eye(4)
                            full_original_transform[:3, :] = original_transform
                            original_transform = full_original_transform
                        
                        # 组合变换：新变换 * 原始变换
                        combined_transform = transform_matrix @ original_transform
                        vertices_transformed = trimesh.transformations.transform_points(vertices, combined_transform)
                        
                        # 设置为单位矩阵，因为变换已应用到顶点
                        node_transform = np.eye(4)
                    else:
                        # 不保留原始变换或原始变换是单位矩阵
                        vertices_transformed = trimesh.transformations.transform_points(vertices, transform_matrix)
                        node_transform = np.eye(4)
                    
                    new_geometry.vertices = vertices_transformed
                else:
                    # 对于不含显式顶点的几何体，在节点级别应用变换
                    if preserve_original_transform and original_transform is not None:
                        node_transform = transform_matrix @ original_transform
                    else:
                        node_transform = transform_matrix
                
                rotated_scene.add_geometry(new_geometry, node_name=node_name, transform=node_transform)
                processing_log.append(f"已变换几何体: {node_name}")
        else:
            # 单个几何体
            rotated_scene = scene.copy()
            rotated_scene.apply_transform(transform_matrix)
            processing_log.append("已变换单个几何体")
        
        return rotated_scene

    def _resolve_file_path(self, file_path: str) -> str:
        """解析文件路径，支持绝对/相对路径以及ComfyUI输出目录"""
        if os.path.isabs(file_path):
            return file_path
        
        # 尝试ComfyUI输出目录
        if FOLDER_PATHS_AVAILABLE:
            try:
                output_dir = folder_paths.get_output_directory()
                candidate = os.path.join(output_dir, file_path)
                if os.path.exists(candidate):
                    return candidate
            except Exception:
                pass
        
        # 相对于当前工作目录
        if os.path.exists(file_path):
            return os.path.abspath(file_path)
        
        return file_path

    def _generate_output_path(self, filename: str) -> str:
        """生成输出文件完整路径，自动处理扩展名"""
        if FOLDER_PATHS_AVAILABLE:
            try:
                output_dir = folder_paths.get_output_directory()
            except Exception:
                output_dir = "output"
        else:
            output_dir = "output"
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename.lower().endswith('.glb'):
            filename += '.glb'
        
        timestamp = int(time.time())
        filename_base, ext = os.path.splitext(filename)
        unique_filename = f"{filename_base}_{timestamp}{ext}"
        
        return os.path.join(output_dir, unique_filename)


# 导出节点类
NODE_CLASS_MAPPINGS = {
    "GLBPointCloudRotator": GLBPointCloudRotator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GLBPointCloudRotator": "GLB点云旋转器"
} 