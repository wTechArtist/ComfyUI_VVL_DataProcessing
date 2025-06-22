from .common import *


class GLBPointCloudOriginAdjuster:
    """GLB点云原点调整计算器 - 计算原点调整信息并生成带坐标轴的预览文件（原模型数据保持不变）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_file_path": ("STRING", {
                    "default": "",
                    "tooltip": "GLB点云文件路径：输入需要调整原点的GLB格式点云文件。注意：此节点在原模型基础上添加坐标轴预览，原模型数据保持不变"
                }),
            },
            "optional": {
                "origin_mode": (["center", "bottom_center"], {
                    "default": "bottom_center",
                    "tooltip": "原点模式：center=点云几何中心；bottom_center=点云底部中心(脚底)"
                }),
                "output_units": (["meters", "centimeters", "millimeters"], {
                    "default": "centimeters",
                    "tooltip": "输出单位：变换信息的输出单位"
                }),
                "add_coordinate_axes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "添加坐标轴：是否添加RGB坐标轴(X=红色，Y=绿色，Z=蓝色)到输出点云中，显示调整后的原点位置"
                }),
                "wireframe_density": ("INT", {
                    "default": 100, "min": 20, "max": 200, "step": 10,
                    "tooltip": "坐标轴密度：每条坐标轴的点数，越高坐标轴越清晰但点数越多。推荐80-120获得清晰效果"
                }),
                "output_filename": ("STRING", {
                    "default": "adjusted_pointcloud",
                    "tooltip": "输出文件名：调整后的点云文件名，系统会自动添加.glb扩展名"
                }),
            }
        }

    RETURN_TYPES = (
        "STRING",    # 处理后的GLB文件路径
        "STRING",    # 变换信息JSON (position)
    )
    RETURN_NAMES = (
        "adjusted_glb_path",
        "transform_info",
    )
    OUTPUT_TOOLTIPS = [
        "带预览的GLB文件路径 - 包含原始点云+坐标轴预览的完整点云文件",
        "纯净的position信息JSON: {\"position\": [x, y, z]} - 仅包含位置信息，可直接用于UE等引擎",
    ]
    OUTPUT_NODE = True
    FUNCTION = "adjust_origin"
    CATEGORY = "💃VVL/Point Cloud Transform"

    def adjust_origin(self,
                     glb_file_path: str,
                     origin_mode: str = "bottom_center",
                     output_units: str = "centimeters",
                     add_coordinate_axes: bool = False,
                     wireframe_density: int = 100,
                     output_filename: str = "adjusted_pointcloud"):
        """
        调整GLB点云的原点位置和旋转
        """
        
        processing_log = []
        processing_log.append("开始GLB点云原点调整...")
        
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
            
            # 收集所有几何体和点云，考虑变换矩阵
            geometries_to_transform = []
            all_vertices = []
            total_points = 0
            
            if isinstance(scene, trimesh.Scene):
                for node_name in scene.graph.nodes:
                    if node_name in scene.geometry:
                        geometry = scene.geometry[node_name]
                        transform_matrix = scene.graph[node_name][0]  # 获取变换矩阵
                        geometries_to_transform.append((node_name, geometry))
                        
                        if hasattr(geometry, 'vertices') and geometry.vertices is not None:
                            # 应用变换矩阵获取真实世界坐标
                            if transform_matrix is not None and not np.allclose(transform_matrix, np.eye(4)):
                                world_vertices = trimesh.transformations.transform_points(geometry.vertices, transform_matrix)
                                processing_log.append(f"发现几何体: {node_name}, 顶点数: {len(geometry.vertices)}, 已应用变换矩阵")
                            else:
                                world_vertices = geometry.vertices.copy()
                                processing_log.append(f"发现几何体: {node_name}, 顶点数: {len(geometry.vertices)}, 无变换")
                            
                            all_vertices.append(world_vertices)
                            total_points += len(geometry.vertices)
            else:
                # 单个几何体
                geometries_to_transform.append(("main_geometry", scene))
                if hasattr(scene, 'vertices') and scene.vertices is not None:
                    all_vertices.append(scene.vertices)
                    total_points = len(scene.vertices)
                    processing_log.append(f"发现几何体: main_geometry, 顶点数: {total_points}")
            
            if not all_vertices:
                error_msg = "GLB文件中没有可用的顶点数据"
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
            
            # 合并所有顶点计算包围盒（已经是世界坐标）
            combined_vertices = np.vstack(all_vertices)
            processing_log.append(f"总顶点数: {total_points:,} (已应用变换矩阵)")
            
            # 计算原始包围盒
            original_min = np.min(combined_vertices, axis=0)
            original_max = np.max(combined_vertices, axis=0)
            original_center = (original_min + original_max) / 2
            original_size = original_max - original_min
            
            processing_log.append(f"原始包围盒:")
            processing_log.append(f"  最小点: [{original_min[0]:.6f}, {original_min[1]:.6f}, {original_min[2]:.6f}]")
            processing_log.append(f"  最大点: [{original_max[0]:.6f}, {original_max[1]:.6f}, {original_max[2]:.6f}]")
            processing_log.append(f"  中心点: [{original_center[0]:.6f}, {original_center[1]:.6f}, {original_center[2]:.6f}]")
            processing_log.append(f"  尺寸: [{original_size[0]:.6f}, {original_size[1]:.6f}, {original_size[2]:.6f}]")
            
            # 确定新的原点位置
            if origin_mode == "center":
                new_origin = original_center.copy()
                processing_log.append("原点模式: 几何中心")
            else:  # bottom_center
                new_origin = np.array([original_center[0], original_center[1], original_min[2]])
                processing_log.append("原点模式: 底部中心(脚底)")
            
            # 计算平移向量
            translation = -new_origin
            processing_log.append(f"平移向量: [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]")
            
            # 创建平移变换矩阵（仅平移，无旋转）
            transform_matrix = np.eye(4)
            transform_matrix[:3, 3] = translation
            
            # 应用变换到所有几何体
            new_scene = trimesh.Scene()
            transformed_vertices_list = []
            
            for name, geometry in geometries_to_transform:
                # 复制几何体
                new_geometry = geometry.copy()
                
                # 应用变换
                new_geometry.apply_transform(transform_matrix)
                
                # 添加到新场景
                new_scene.add_geometry(new_geometry, node_name=name)
                
                # 收集变换后的顶点用于统计
                if hasattr(new_geometry, 'vertices') and new_geometry.vertices is not None:
                    transformed_vertices_list.append(new_geometry.vertices)
                
                processing_log.append(f"已变换几何体: {name}")
            
            # 添加坐标轴预览（在变换后的原点位置，即(0,0,0)）
            if add_coordinate_axes:
                try:
                    # 计算变换后的包围盒来确定坐标轴长度
                    if transformed_vertices_list:
                        all_transformed_vertices = np.vstack(transformed_vertices_list)
                        transformed_extents = np.max(all_transformed_vertices, axis=0) - np.min(all_transformed_vertices, axis=0)
                        axis_length = np.max(transformed_extents) * 0.4
                    else:
                        axis_length = 1.0  # 默认长度
                    
                    # 在新的原点(0,0,0)处生成坐标轴
                    axes_vertices, axes_colors = self._create_coordinate_axes_pointcloud_at_origin(
                        axis_length, wireframe_density, processing_log
                    )
                    
                    if len(axes_vertices) > 0:
                        # 创建坐标轴点云
                        axes_pointcloud = trimesh.PointCloud(vertices=axes_vertices, colors=axes_colors)
                        new_scene.add_geometry(axes_pointcloud, node_name="coordinate_axes")
                        processing_log.append(f"坐标轴预览: {len(axes_vertices):,} 个点 (显示新的原点位置)")
                    
                except Exception as e:
                    processing_log.append(f"坐标轴生成失败: {str(e)}")
            
            # 计算变换后的包围盒
            if transformed_vertices_list:
                transformed_vertices = np.vstack(transformed_vertices_list)
                new_min = np.min(transformed_vertices, axis=0)
                new_max = np.max(transformed_vertices, axis=0)
                new_center = (new_min + new_max) / 2
                new_size = new_max - new_min
                
                processing_log.append(f"变换后包围盒:")
                processing_log.append(f"  最小点: [{new_min[0]:.6f}, {new_min[1]:.6f}, {new_min[2]:.6f}]")
                processing_log.append(f"  最大点: [{new_max[0]:.6f}, {new_max[1]:.6f}, {new_max[2]:.6f}]")
                processing_log.append(f"  中心点: [{new_center[0]:.6f}, {new_center[1]:.6f}, {new_center[2]:.6f}]")
                processing_log.append(f"  尺寸: [{new_size[0]:.6f}, {new_size[1]:.6f}, {new_size[2]:.6f}]")
            
            # 生成带预览的输出文件（原模型数据不变，仅添加坐标轴预览）
            output_path = self._generate_output_path(output_filename)
            processing_log.append(f"输出文件路径: {output_path}")
            
            # 创建包含原始点云和坐标轴的预览场景，完全保持原始结构
            original_scene = trimesh.load(input_path)  # 重新加载原始场景
            
            # 创建预览场景，完全复制原始场景的结构
            if isinstance(original_scene, trimesh.Scene):
                preview_scene = trimesh.Scene()
                # 完全复制原始场景的所有几何体和变换矩阵
                for node_name in original_scene.graph.nodes:
                    if node_name in original_scene.geometry:
                        geometry = original_scene.geometry[node_name]
                        transform_matrix = original_scene.graph[node_name][0]  # 获取变换矩阵
                        
                        # 完全复制几何体，保持所有属性
                        copied_geometry = geometry.copy()
                        
                        # 添加到预览场景，保持原始的变换矩阵
                        preview_scene.add_geometry(copied_geometry, node_name=node_name, transform=transform_matrix)
                        processing_log.append(f"完全保持原始几何体: {node_name}, 变换矩阵和朝向已保留")
            else:
                # 单个几何体的情况
                preview_scene = trimesh.Scene()
                preview_scene.add_geometry(original_scene.copy(), node_name="main_geometry")
                processing_log.append("完全保持原始几何体: main_geometry, 所有属性已保留")
            
            # 添加坐标轴预览（如果需要）
            if add_coordinate_axes:
                try:
                    # 使用原始包围盒来确定坐标轴长度
                    axis_length = np.max(original_size) * 0.4
                    
                    # 在计算出的新原点位置生成坐标轴
                    axes_vertices, axes_colors = self._create_coordinate_axes_pointcloud_at_position(
                        new_origin, axis_length, wireframe_density, processing_log
                    )
                    
                    if len(axes_vertices) > 0:
                        # 创建坐标轴点云并添加到场景
                        axes_pointcloud = trimesh.PointCloud(vertices=axes_vertices, colors=axes_colors)
                        preview_scene.add_geometry(axes_pointcloud, node_name="coordinate_axes_preview")
                        processing_log.append(f"坐标轴预览: {len(axes_vertices):,} 个点 (显示建议的新原点位置)")
                    
                except Exception as e:
                    processing_log.append(f"坐标轴生成失败: {str(e)}")
            
            # 保存预览文件
            processing_log.append("正在保存带预览的GLB文件...")
            preview_scene.export(output_path)
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                processing_log.append(f"预览GLB文件保存成功，文件大小: {file_size} bytes")
            else:
                error_msg = "预览GLB文件保存失败"
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
            
            # 应用单位转换
            unit_scale = {"meters": 1.0, "centimeters": 100.0, "millimeters": 1000.0}
            scale_factor = unit_scale.get(output_units, 1.0)
            
            # 生成变换信息（UE格式）
            final_position = new_origin * scale_factor  # 相对于原始坐标系的位置
            transform_info = {
                "name": output_filename,
                "position": [float(final_position[0]), float(final_position[1]), float(final_position[2])]
            }
            
            processing_log.append("")
            processing_log.append(f"变换信息 (单位: {output_units}):")
            processing_log.append(f"  Position: [{transform_info['position'][0]:.2f}, {transform_info['position'][1]:.2f}, {transform_info['position'][2]:.2f}]")
            processing_log.append("原点调整预览生成完成!")
            
            return (
                output_path,
                json.dumps(transform_info, indent=2),
            )
                
        except Exception as e:
            error_msg = f"调整原点时发生错误: {str(e)}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return ("", "")
    
    def _resolve_file_path(self, file_path: str) -> str:
        """解析文件路径，支持绝对路径和相对路径"""
        if os.path.isabs(file_path):
            return file_path
        
        # 尝试相对于ComfyUI输出目录
        if FOLDER_PATHS_AVAILABLE:
            output_dir = folder_paths.get_output_directory()
            candidate_path = os.path.join(output_dir, file_path)
            if os.path.exists(candidate_path):
                return candidate_path
        
        # 尝试相对于当前工作目录
        if os.path.exists(file_path):
            return os.path.abspath(file_path)
        
        # 返回原始路径（让后续检查处理错误）
        return file_path
    
    def _generate_output_path(self, filename: str) -> str:
        """生成输出文件路径"""
        # 确保文件名有正确的扩展名
        if not filename.lower().endswith('.glb'):
            filename += '.glb'
        
        # 生成输出路径
        output_dir = folder_paths.get_output_directory() if FOLDER_PATHS_AVAILABLE else "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 添加时间戳避免文件名冲突
        timestamp = str(int(time.time()))
        name_parts = filename.rsplit('.', 1)
        if len(name_parts) == 2:
            timestamped_filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
        else:
            timestamped_filename = f"{filename}_{timestamp}"
        
        return os.path.join(output_dir, timestamped_filename)
    

    
    def _create_coordinate_axes_pointcloud_at_position(self, position, axis_length, wireframe_density, processing_log):
        """在指定位置创建坐标轴的点云表示"""
        try:
            center = np.array(position)  # 指定的位置
            axis_thickness = axis_length * 0.01  # 坐标轴厚度
            axes_points = []
            axes_colors = []
            
            # 计算坐标轴密度
            axis_line_density = wireframe_density  # 使用参数控制密度
            axis_thickness_points = max(3, axis_line_density // 15)  # 厚度方向的点数
            
            # X轴 - 红色（从指定位置开始）
            for i in range(axis_line_density):
                t = i / (axis_line_density - 1) if axis_line_density > 1 else 0
                base_point = [center[0] + t * axis_length, center[1], center[2]]
                
                # 主轴线
                axes_points.append(base_point)
                axes_colors.append([255, 0, 0, 255])
                
                # 增加厚度（在YZ平面上添加点）
                for j in range(axis_thickness_points):
                    for k in range(axis_thickness_points):
                        offset_y = (j - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        offset_z = (k - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        thick_point = [base_point[0], base_point[1] + offset_y, base_point[2] + offset_z]
                        axes_points.append(thick_point)
                        axes_colors.append([255, 0, 0, 255])
            
            # X轴箭头头部
            arrow_length = axis_length * 0.1
            arrow_base = axis_length * 0.9
            for i in range(axis_line_density // 2):
                t = i / (axis_line_density // 2 - 1) if axis_line_density > 2 else 0
                arrow_x = center[0] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([arrow_x, center[1] + arrow_offset, center[2]])
                axes_points.append([arrow_x, center[1] - arrow_offset, center[2]])
                axes_points.append([arrow_x, center[1], center[2] + arrow_offset])
                axes_points.append([arrow_x, center[1], center[2] - arrow_offset])
                axes_colors.extend([[255, 0, 0, 255]] * 4)
            
            # Y轴 - 绿色（从指定位置开始）
            for i in range(axis_line_density):
                t = i / (axis_line_density - 1) if axis_line_density > 1 else 0
                base_point = [center[0], center[1] + t * axis_length, center[2]]
                
                # 主轴线
                axes_points.append(base_point)
                axes_colors.append([0, 255, 0, 255])
                
                # 增加厚度（在XZ平面上添加点）
                for j in range(axis_thickness_points):
                    for k in range(axis_thickness_points):
                        offset_x = (j - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        offset_z = (k - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        thick_point = [base_point[0] + offset_x, base_point[1], base_point[2] + offset_z]
                        axes_points.append(thick_point)
                        axes_colors.append([0, 255, 0, 255])
            
            # Y轴箭头头部
            for i in range(axis_line_density // 2):
                t = i / (axis_line_density // 2 - 1) if axis_line_density > 2 else 0
                arrow_y = center[1] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([center[0] + arrow_offset, arrow_y, center[2]])
                axes_points.append([center[0] - arrow_offset, arrow_y, center[2]])
                axes_points.append([center[0], arrow_y, center[2] + arrow_offset])
                axes_points.append([center[0], arrow_y, center[2] - arrow_offset])
                axes_colors.extend([[0, 255, 0, 255]] * 4)
            
            # Z轴 - 蓝色（从指定位置开始）
            for i in range(axis_line_density):
                t = i / (axis_line_density - 1) if axis_line_density > 1 else 0
                base_point = [center[0], center[1], center[2] + t * axis_length]
                
                # 主轴线
                axes_points.append(base_point)
                axes_colors.append([0, 0, 255, 255])
                
                # 增加厚度（在XY平面上添加点）
                for j in range(axis_thickness_points):
                    for k in range(axis_thickness_points):
                        offset_x = (j - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        offset_y = (k - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        thick_point = [base_point[0] + offset_x, base_point[1] + offset_y, base_point[2]]
                        axes_points.append(thick_point)
                        axes_colors.append([0, 0, 255, 255])
            
            # Z轴箭头头部
            for i in range(axis_line_density // 2):
                t = i / (axis_line_density // 2 - 1) if axis_line_density > 2 else 0
                arrow_z = center[2] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([center[0] + arrow_offset, center[1], arrow_z])
                axes_points.append([center[0] - arrow_offset, center[1], arrow_z])
                axes_points.append([center[0], center[1] + arrow_offset, arrow_z])
                axes_points.append([center[0], center[1] - arrow_offset, arrow_z])
                axes_colors.extend([[0, 0, 255, 255]] * 4)
            
            axes_vertices = np.array(axes_points)
            axes_colors_array = np.array(axes_colors)
            
            processing_log.append(f"坐标轴生成: {len(axes_vertices):,} 个点 (长度={axis_length:.3f}, 密度={wireframe_density}, 位置=[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}])")
            
            return axes_vertices, axes_colors_array
            
        except Exception as e:
            processing_log.append(f"创建坐标轴点云失败: {str(e)}")
            return np.array([]), np.array([])

    def _create_coordinate_axes_pointcloud_at_origin(self, axis_length, wireframe_density, processing_log):
        """在原点(0,0,0)创建坐标轴的点云表示"""
        try:
            origin_center = np.array([0.0, 0.0, 0.0])  # 新的原点位置
            axis_thickness = axis_length * 0.01  # 坐标轴厚度
            axes_points = []
            axes_colors = []
            
            # 计算坐标轴密度
            axis_line_density = wireframe_density  # 使用参数控制密度
            axis_thickness_points = max(3, axis_line_density // 15)  # 厚度方向的点数
            
            # X轴 - 红色（从原点开始）
            for i in range(axis_line_density):
                t = i / (axis_line_density - 1) if axis_line_density > 1 else 0
                base_point = [origin_center[0] + t * axis_length, origin_center[1], origin_center[2]]
                
                # 主轴线
                axes_points.append(base_point)
                axes_colors.append([255, 0, 0, 255])
                
                # 增加厚度（在YZ平面上添加点）
                for j in range(axis_thickness_points):
                    for k in range(axis_thickness_points):
                        offset_y = (j - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        offset_z = (k - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        thick_point = [base_point[0], base_point[1] + offset_y, base_point[2] + offset_z]
                        axes_points.append(thick_point)
                        axes_colors.append([255, 0, 0, 255])
            
            # X轴箭头头部
            arrow_length = axis_length * 0.1
            arrow_base = axis_length * 0.9
            for i in range(axis_line_density // 2):
                t = i / (axis_line_density // 2 - 1) if axis_line_density > 2 else 0
                arrow_x = origin_center[0] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([arrow_x, origin_center[1] + arrow_offset, origin_center[2]])
                axes_points.append([arrow_x, origin_center[1] - arrow_offset, origin_center[2]])
                axes_points.append([arrow_x, origin_center[1], origin_center[2] + arrow_offset])
                axes_points.append([arrow_x, origin_center[1], origin_center[2] - arrow_offset])
                axes_colors.extend([[255, 0, 0, 255]] * 4)
            
            # Y轴 - 绿色（从原点开始）
            for i in range(axis_line_density):
                t = i / (axis_line_density - 1) if axis_line_density > 1 else 0
                base_point = [origin_center[0], origin_center[1] + t * axis_length, origin_center[2]]
                
                # 主轴线
                axes_points.append(base_point)
                axes_colors.append([0, 255, 0, 255])
                
                # 增加厚度（在XZ平面上添加点）
                for j in range(axis_thickness_points):
                    for k in range(axis_thickness_points):
                        offset_x = (j - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        offset_z = (k - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        thick_point = [base_point[0] + offset_x, base_point[1], base_point[2] + offset_z]
                        axes_points.append(thick_point)
                        axes_colors.append([0, 255, 0, 255])
            
            # Y轴箭头头部
            for i in range(axis_line_density // 2):
                t = i / (axis_line_density // 2 - 1) if axis_line_density > 2 else 0
                arrow_y = origin_center[1] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([origin_center[0] + arrow_offset, arrow_y, origin_center[2]])
                axes_points.append([origin_center[0] - arrow_offset, arrow_y, origin_center[2]])
                axes_points.append([origin_center[0], arrow_y, origin_center[2] + arrow_offset])
                axes_points.append([origin_center[0], arrow_y, origin_center[2] - arrow_offset])
                axes_colors.extend([[0, 255, 0, 255]] * 4)
            
            # Z轴 - 蓝色（从原点开始）
            for i in range(axis_line_density):
                t = i / (axis_line_density - 1) if axis_line_density > 1 else 0
                base_point = [origin_center[0], origin_center[1], origin_center[2] + t * axis_length]
                
                # 主轴线
                axes_points.append(base_point)
                axes_colors.append([0, 0, 255, 255])
                
                # 增加厚度（在XY平面上添加点）
                for j in range(axis_thickness_points):
                    for k in range(axis_thickness_points):
                        offset_x = (j - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        offset_y = (k - axis_thickness_points//2) * axis_thickness / axis_thickness_points
                        thick_point = [base_point[0] + offset_x, base_point[1] + offset_y, base_point[2]]
                        axes_points.append(thick_point)
                        axes_colors.append([0, 0, 255, 255])
            
            # Z轴箭头头部
            for i in range(axis_line_density // 2):
                t = i / (axis_line_density // 2 - 1) if axis_line_density > 2 else 0
                arrow_z = origin_center[2] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([origin_center[0] + arrow_offset, origin_center[1], arrow_z])
                axes_points.append([origin_center[0] - arrow_offset, origin_center[1], arrow_z])
                axes_points.append([origin_center[0], origin_center[1] + arrow_offset, arrow_z])
                axes_points.append([origin_center[0], origin_center[1] - arrow_offset, arrow_z])
                axes_colors.extend([[0, 0, 255, 255]] * 4)
            
            axes_vertices = np.array(axes_points)
            axes_colors_array = np.array(axes_colors)
            
            processing_log.append(f"坐标轴生成: {len(axes_vertices):,} 个点 (长度={axis_length:.3f}, 密度={wireframe_density}, 原点=[0.000, 0.000, 0.000])")
            
            return axes_vertices, axes_colors_array
            
        except Exception as e:
            processing_log.append(f"创建坐标轴点云失败: {str(e)}")
            return np.array([]), np.array([])

