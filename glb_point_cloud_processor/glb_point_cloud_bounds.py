from .common import *

class GLBPointCloudBounds:
    """GLB点云包围盒计算器 - 计算包围盒并生成带可视化的预览文件（原模型数据保持不变）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_file_path": ("STRING", {
                    "default": "",
                    "tooltip": "GLB点云文件路径：输入需要计算包围盒的GLB格式点云文件。支持绝对路径或相对于ComfyUI output目录的相对路径。注意：此节点在原模型基础上添加预览信息，原模型数据保持不变"
                }),
            },
            "optional": {
                "bounding_box_type": (["axis_aligned", "oriented"], {
                    "default": "axis_aligned",
                    "tooltip": "包围盒类型：axis_aligned=轴对齐包围盒(AABB)，计算快速；oriented=有向包围盒(OBB)，体积最小"
                }),
                "units": ([1, 100, 1000, 10000, 100000], {
                    "default": 1, 
                    "tooltip": "输出单位倍数：1=米，100=厘米，1000=毫米，10000=0.1毫米，100000=微米"
                }),
                "add_bounding_box_visualization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "添加包围盒可视化：是否在输出的点云文件中添加红色包围盒线框点云，便于直接预览验证"
                }),
                "add_coordinate_axes": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "添加坐标轴：是否添加RGB坐标轴(X=红色，Y=绿色，Z=蓝色)到输出点云中"
                }),
                "wireframe_density": ("INT", {
                    "default": 150, "min": 20, "max": 200, "step": 10,
                    "tooltip": "线框密度：每条包围盒边的点数，越高线框越清晰但点数越多。推荐50-100获得清晰效果"
                }),
                "enhance_visibility": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "增强可见性：添加顶点高亮和面中心标记，让包围盒更容易识别"
                }),
                "output_filename": ("STRING", {
                    "default": "pointcloud_with_bounds",
                    "tooltip": "输出文件名：带包围盒可视化的点云文件名，系统会自动添加.glb扩展名"
                }),
            }
        }

    RETURN_TYPES = (
        "STRING",    # 输出的GLB文件路径
        "STRING",    # JSON格式的scale数组
    )
    RETURN_NAMES = (
        "output_glb_path",
        "scale_json",
    )
    OUTPUT_TOOLTIPS = [
        "带预览的GLB文件路径 - 包含原始点云+包围盒可视化+坐标轴的完整点云文件",
        "纯净的scale数组JSON: {\"scale\": [长, 宽, 高]} - 仅包含scale信息，可直接用于3D引擎缩放",
    ]
    OUTPUT_NODE = True
    FUNCTION = "calculate_bounds_and_visualize"
    CATEGORY = "💃VVL/Point Cloud Analysis"

    def calculate_bounds_and_visualize(self,
                        glb_file_path: str,
                        bounding_box_type: str = "axis_aligned",
                        units: int = 1,
                        add_bounding_box_visualization: bool = True,
                        add_coordinate_axes: bool = True,
                        wireframe_density: int = 50,
                        enhance_visibility: bool = True,
                        output_filename: str = "pointcloud_with_bounds"):
        """
        计算GLB点云文件的包围盒尺寸并生成带可视化的点云文件
        """
        
        processing_log = []
        processing_log.append("开始GLB点云包围盒计算和可视化生成...")
        
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
        self._current_input_path = input_path  # 保存当前输入路径供后续使用
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
            
            # 收集所有点云数据，考虑变换矩阵
            all_vertices = []
            point_cloud_count = 0
            
            if isinstance(scene, trimesh.Scene):
                for node_name in scene.graph.nodes:
                    if node_name in scene.geometry:
                        geometry = scene.geometry[node_name]
                        transform_matrix = scene.graph[node_name][0]  # 获取变换矩阵
                        
                        if hasattr(geometry, 'vertices') and geometry.vertices is not None:
                            # 应用变换矩阵获取真实世界坐标
                            if transform_matrix is not None and not np.allclose(transform_matrix, np.eye(4)):
                                world_vertices = trimesh.transformations.transform_points(geometry.vertices, transform_matrix)
                                processing_log.append(f"发现几何体: {node_name}, 点数: {len(geometry.vertices)}, 已应用变换矩阵")
                            else:
                                world_vertices = geometry.vertices.copy()
                                processing_log.append(f"发现几何体: {node_name}, 点数: {len(geometry.vertices)}, 无变换")
                            
                            all_vertices.append(world_vertices)
                            point_cloud_count += 1
            elif isinstance(scene, trimesh.PointCloud):
                all_vertices.append(scene.vertices)
                point_cloud_count = 1
                processing_log.append(f"发现点云: main_pointcloud, 点数: {len(scene.vertices)}")
            elif hasattr(scene, 'vertices') and scene.vertices is not None:
                all_vertices.append(scene.vertices)
                point_cloud_count = 1
                processing_log.append(f"发现几何体(作为点云处理): main_geometry, 点数: {len(scene.vertices)}")
            else:
                error_msg = "GLB文件中未找到点云或几何体数据"
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
            
            if not all_vertices:
                error_msg = "GLB文件中没有可用的顶点数据"
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
            
            # 合并所有顶点（已经是世界坐标）
            combined_vertices = np.vstack(all_vertices)
            total_points = len(combined_vertices)
            processing_log.append(f"合并了 {point_cloud_count} 个几何体，总点数: {total_points} (已应用变换矩阵)")
            
            # 计算包围盒
            processing_log.append(f"计算包围盒类型: {bounding_box_type}")
            
            if bounding_box_type == "axis_aligned":
                # 轴对齐包围盒 (AABB)
                min_point = np.min(combined_vertices, axis=0)
                max_point = np.max(combined_vertices, axis=0)
                extents = max_point - min_point
                center = (min_point + max_point) / 2
                
                processing_log.append(f"AABB计算完成:")
                processing_log.append(f"  最小点: [{min_point[0]:.6f}, {min_point[1]:.6f}, {min_point[2]:.6f}]")
                processing_log.append(f"  最大点: [{max_point[0]:.6f}, {max_point[1]:.6f}, {max_point[2]:.6f}]")
                processing_log.append(f"  中心点: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
                processing_log.append(f"  尺寸: [{extents[0]:.6f}, {extents[1]:.6f}, {extents[2]:.6f}]")
                processing_log.append(f"  体积: {np.prod(extents):.6f}")
                
            else:  # oriented
                # 有向包围盒 (OBB)
                try:
                    to_origin, obb_extents = trimesh.bounds.oriented_bounds(combined_vertices)
                    
                    # 计算OBB的中心点
                    obb_center = -to_origin[:3, 3]  # 变换矩阵的平移部分的负值
                    center = obb_center  # 设置center变量
                    extents = obb_extents
                    
                    processing_log.append(f"OBB计算完成:")
                    processing_log.append(f"  中心点: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
                    processing_log.append(f"  尺寸: [{extents[0]:.6f}, {extents[1]:.6f}, {extents[2]:.6f}]")
                    processing_log.append(f"  体积: {np.prod(extents):.6f}")
                    
                except Exception as e:
                    processing_log.append(f"OBB计算失败，回退到AABB: {str(e)}")
                    # 回退到AABB
                    min_point = np.min(combined_vertices, axis=0)
                    max_point = np.max(combined_vertices, axis=0)
                    extents = max_point - min_point
                    center = (min_point + max_point) / 2
                    

            
            # 应用单位转换
            scale_factor = float(units)
            
            scaled_extents = extents * scale_factor
            
            # 生成scale JSON (按照 [长, 宽, 高] 的顺序，通常是 [X, Y, Z])
            scale_array = [float(scaled_extents[0]), float(scaled_extents[1]), float(scaled_extents[2])]
            scale_json = {
                "name": output_filename,
                "scale": scale_array
            }
            

            
            processing_log.append(f"")
            processing_log.append(f"输出结果 (单位倍数: {units}):")
            processing_log.append(f"  Scale数组: [{scale_array[0]:.6f}, {scale_array[1]:.6f}, {scale_array[2]:.6f}]")
            processing_log.append(f"  长度(X): {scale_array[0]:.6f} (x{units})")
            processing_log.append(f"  宽度(Y): {scale_array[1]:.6f} (x{units})")
            processing_log.append(f"  高度(Z): {scale_array[2]:.6f} (x{units})")
            
            # 生成带可视化的点云文件（原模型数据不变，仅添加预览）
            output_glb_path = ""
            
            try:
                processing_log.append("正在生成带包围盒可视化的点云文件...")
                output_glb_path = self._generate_visualization_pointcloud(
                    combined_vertices, extents, center, bounding_box_type,
                    add_bounding_box_visualization, add_coordinate_axes, 
                    wireframe_density, enhance_visibility, output_filename, processing_log
                )
                processing_log.append("可视化点云文件生成完成!")
            except Exception as e:
                processing_log.append(f"可视化点云生成失败: {str(e)}")
                import traceback
                traceback.print_exc()
            
            processing_log.append("GLB点云包围盒计算和可视化完成!")
            
            return (
                output_glb_path,
                json.dumps(scale_json, indent=2),
            )
                
        except Exception as e:
            error_msg = f"计算包围盒时发生错误: {str(e)}"
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

    def _get_current_input_path(self):
        """获取当前输入文件路径"""
        return getattr(self, '_current_input_path', None)

    def _normalize_colors(self, colors):
        """标准化颜色数组为RGBA格式，0-255范围"""
        if colors is None:
            return None
        
        colors = np.array(colors)
        
        # 如果是0-1范围，转换为0-255
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
        
        # 确保是RGBA格式
        if colors.shape[1] == 3:  # RGB -> RGBA
            alpha_channel = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
            colors = np.hstack([colors, alpha_channel])
        elif colors.shape[1] != 4:
            # 如果不是3或4通道，创建白色RGBA
            colors = np.tile([255, 255, 255, 255], (colors.shape[0], 1)).astype(np.uint8)
        
        return colors

    def _generate_visualization_pointcloud(self, combined_vertices, extents, center, bounds_type,
                                          add_bounding_box_visualization, add_coordinate_axes,
                                          wireframe_density, enhance_visibility, output_filename, processing_log):
        """生成包含原始点云、包围盒线框和坐标轴的可视化点云文件"""
        try:
            # 直接使用已加载的原始场景数据，避免重新加载可能改变朝向
            input_path = self._get_current_input_path()
            
            # 提取原始点云数据（完全保持原始状态）
            all_vertices = []
            all_colors = []
            original_point_count = 0
            
            # 使用之前已经加载的combined_vertices作为基础，重新从原始场景提取完整数据
            original_scene = trimesh.load(input_path)
            
            # 创建一个新的场景，完全保持原始场景的结构和变换
            visualization_scene = trimesh.Scene()
            
            # 保持原始场景的完整结构
            if isinstance(original_scene, trimesh.Scene):
                # 完全复制原始场景的所有几何体和变换
                for node_name in original_scene.graph.nodes:
                    if node_name in original_scene.geometry:
                        geometry = original_scene.geometry[node_name]
                        transform_matrix = original_scene.graph[node_name][0]  # 获取变换矩阵
                        
                        # 完全复制几何体，保持所有属性
                        copied_geometry = geometry.copy()
                        
                        # 添加到新场景，保持原始的变换矩阵
                        visualization_scene.add_geometry(copied_geometry, node_name=node_name, transform=transform_matrix)
                        
                        # 提取顶点用于统计（注意：这里用于显示统计，真实包围盒计算已在前面完成）
                        if hasattr(geometry, 'vertices') and geometry.vertices is not None:
                            # 这里不需要再次应用变换，因为包围盒计算已经在combined_vertices中处理过了
                            all_vertices.append(geometry.vertices)
                            
                            # 保持原始颜色
                            if isinstance(geometry, trimesh.PointCloud):
                                if hasattr(geometry.visual, 'vertex_colors') and geometry.visual.vertex_colors is not None:
                                    colors = geometry.visual.vertex_colors.copy()
                                elif hasattr(geometry, 'colors') and geometry.colors is not None:
                                    colors = geometry.colors.copy()
                                else:
                                    colors = np.tile([255, 255, 255, 255], (len(geometry.vertices), 1))
                            else:
                                if hasattr(geometry.visual, 'vertex_colors') and geometry.visual.vertex_colors is not None:
                                    colors = geometry.visual.vertex_colors.copy()
                                else:
                                    colors = np.tile([200, 200, 200, 255], (len(geometry.vertices), 1))
                            
                            # 确保颜色格式一致 (RGBA, 0-255)
                            colors = self._normalize_colors(colors)
                            all_colors.append(colors)
                            original_point_count += len(geometry.vertices)
                            
                        processing_log.append(f"完全保持原始几何体: {node_name}, 变换矩阵和所有属性已保留")
            
            elif isinstance(original_scene, trimesh.PointCloud):
                # 单个点云，直接复制
                visualization_scene = original_scene.copy()
                all_vertices.append(original_scene.vertices)
                
                if hasattr(original_scene.visual, 'vertex_colors') and original_scene.visual.vertex_colors is not None:
                    colors = original_scene.visual.vertex_colors.copy()
                elif hasattr(original_scene, 'colors') and original_scene.colors is not None:
                    colors = original_scene.colors.copy()
                else:
                    colors = np.tile([255, 255, 255, 255], (len(original_scene.vertices), 1))
                
                colors = self._normalize_colors(colors)
                all_colors.append(colors)
                original_point_count = len(original_scene.vertices)
                processing_log.append(f"完全保持原始点云: {original_point_count:,} 个点")
            
            else:
                # 单个几何体，转换为场景保持结构
                visualization_scene.add_geometry(original_scene.copy(), node_name="main_geometry")
                if hasattr(original_scene, 'vertices') and original_scene.vertices is not None:
                    all_vertices.append(original_scene.vertices)
                    
                    if hasattr(original_scene.visual, 'vertex_colors') and original_scene.visual.vertex_colors is not None:
                        colors = original_scene.visual.vertex_colors.copy()
                    else:
                        colors = np.tile([200, 200, 200, 255], (len(original_scene.vertices), 1))
                    
                    colors = self._normalize_colors(colors)
                    all_colors.append(colors)
                    original_point_count = len(original_scene.vertices)
                    processing_log.append(f"完全保持原始几何体: {original_point_count:,} 个点")
            
            processing_log.append(f"原始模型结构完全保留: {original_point_count:,} 个点，所有变换矩阵和朝向不变")
            
            # 添加包围盒线框点云到场景
            if add_bounding_box_visualization:
                box_vertices, box_colors = self._create_bounding_box_pointcloud(
                    extents, center, bounds_type, wireframe_density, enhance_visibility, processing_log
                )
                if len(box_vertices) > 0:
                    # 创建包围盒点云并添加到场景
                    bounding_box_pointcloud = trimesh.PointCloud(vertices=box_vertices, colors=box_colors)
                    visualization_scene.add_geometry(bounding_box_pointcloud, node_name="bounding_box_visualization")
                    processing_log.append(f"包围盒线框: {len(box_vertices):,} 个点")
            
            # 添加坐标轴点云到场景
            if add_coordinate_axes:
                axes_vertices, axes_colors = self._create_coordinate_axes_pointcloud(
                    extents, center, wireframe_density, processing_log
                )
                if len(axes_vertices) > 0:
                    # 创建坐标轴点云并添加到场景
                    coordinate_axes_pointcloud = trimesh.PointCloud(vertices=axes_vertices, colors=axes_colors)
                    visualization_scene.add_geometry(coordinate_axes_pointcloud, node_name="coordinate_axes")
                    processing_log.append(f"坐标轴: {len(axes_vertices):,} 个点")
            
            # 统计总点数
            total_visualization_points = original_point_count
            box_point_count = 0
            axes_point_count = 0
            
            if add_bounding_box_visualization:
                box_point_count = len(box_vertices) if 'box_vertices' in locals() and len(box_vertices) > 0 else 0
                total_visualization_points += box_point_count
            
            if add_coordinate_axes:
                axes_point_count = len(axes_vertices) if 'axes_vertices' in locals() and len(axes_vertices) > 0 else 0
                total_visualization_points += axes_point_count
            
            processing_log.append(f"总点数: {total_visualization_points:,} 个点")
            
            # 生成输出路径
            output_path = self._generate_output_path(output_filename)
            
            # 保存场景，完全保持原始模型的结构和朝向
            visualization_scene.export(output_path)
            
            processing_log.append(f"可视化点云已保存: {output_path}")
            return output_path
            
        except Exception as e:
            processing_log.append(f"可视化点云生成失败: {str(e)}")
            raise e

    def _create_bounding_box_pointcloud(self, extents, center, bounds_type, wireframe_density, enhance_visibility, processing_log):
        """创建包围盒线框的点云表示"""
        try:
            # 计算包围盒的8个顶点
            min_point = center - extents/2
            max_point = center + extents/2
            
            box_vertices = np.array([
                [min_point[0], min_point[1], min_point[2]],  # 0: min corner
                [max_point[0], min_point[1], min_point[2]],  # 1
                [max_point[0], max_point[1], min_point[2]],  # 2
                [min_point[0], max_point[1], min_point[2]],  # 3
                [min_point[0], min_point[1], max_point[2]],  # 4
                [max_point[0], min_point[1], max_point[2]],  # 5
                [max_point[0], max_point[1], max_point[2]],  # 6: max corner
                [min_point[0], max_point[1], max_point[2]],  # 7
            ])
            
            # 定义包围盒的12条边
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
                (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
                (0, 4), (1, 5), (2, 6), (3, 7),  # 垂直边
            ]
            
            wireframe_points = []
            
            # 1. 生成线框边上的点（更密集）
            for start_idx, end_idx in edges:
                start_point = box_vertices[start_idx]
                end_point = box_vertices[end_idx]
                
                # 在每条边上生成密集的点
                for i in range(wireframe_density):
                    t = i / (wireframe_density - 1) if wireframe_density > 1 else 0
                    point = start_point + t * (end_point - start_point)
                    wireframe_points.append(point)
            
            # 2. 添加顶点高亮和面中心标记（可选增强可见性）
            if enhance_visibility:
                # 顶点高亮（每个顶点周围添加小点云）
                vertex_highlight_radius = np.min(extents) * 0.01  # 顶点高亮半径
                highlight_density = max(5, wireframe_density // 10)  # 顶点周围的点数
                
                for vertex in box_vertices:
                    # 在每个顶点周围创建小的点云球
                    for i in range(highlight_density):
                        # 生成随机方向
                        theta = np.random.uniform(0, 2 * np.pi)
                        phi = np.random.uniform(0, np.pi)
                        r = np.random.uniform(0, vertex_highlight_radius)
                        
                        # 球坐标转换为笛卡尔坐标
                        x = vertex[0] + r * np.sin(phi) * np.cos(theta)
                        y = vertex[1] + r * np.sin(phi) * np.sin(theta)
                        z = vertex[2] + r * np.cos(phi)
                        
                        wireframe_points.append([x, y, z])
                
                # 面中心点标记
                face_centers = [
                    # 底面中心
                    (box_vertices[0] + box_vertices[1] + box_vertices[2] + box_vertices[3]) / 4,
                    # 顶面中心
                    (box_vertices[4] + box_vertices[5] + box_vertices[6] + box_vertices[7]) / 4,
                    # 前面中心
                    (box_vertices[0] + box_vertices[1] + box_vertices[4] + box_vertices[5]) / 4,
                    # 后面中心
                    (box_vertices[2] + box_vertices[3] + box_vertices[6] + box_vertices[7]) / 4,
                    # 左面中心
                    (box_vertices[0] + box_vertices[3] + box_vertices[4] + box_vertices[7]) / 4,
                    # 右面中心
                    (box_vertices[1] + box_vertices[2] + box_vertices[5] + box_vertices[6]) / 4,
                ]
                
                # 在每个面中心添加标记点
                face_mark_density = max(3, wireframe_density // 20)
                face_mark_radius = np.min(extents) * 0.005
                
                for face_center in face_centers:
                    for i in range(face_mark_density):
                        # 在面中心周围添加小的点云
                        theta = np.random.uniform(0, 2 * np.pi)
                        phi = np.random.uniform(0, np.pi)
                        r = np.random.uniform(0, face_mark_radius)
                        
                        x = face_center[0] + r * np.sin(phi) * np.cos(theta)
                        y = face_center[1] + r * np.sin(phi) * np.sin(theta)
                        z = face_center[2] + r * np.cos(phi)
                        
                        wireframe_points.append([x, y, z])
            
            wireframe_vertices = np.array(wireframe_points)
            
            # 为包围盒线框设置红色
            wireframe_colors = np.tile([255, 0, 0, 255], (len(wireframe_vertices), 1))
            
            processing_log.append(f"包围盒线框生成: {len(wireframe_vertices):,} 个点 (密度={wireframe_density})")
            
            return wireframe_vertices, wireframe_colors
            
        except Exception as e:
            processing_log.append(f"创建包围盒点云失败: {str(e)}")
            return np.array([]), np.array([])
    
    def _create_coordinate_axes_pointcloud(self, extents, origin_center, wireframe_density, processing_log):
        """创建坐标轴的点云表示"""
        try:
            axis_length = np.max(extents) * 0.4
            axis_thickness = np.min(extents) * 0.005  # 坐标轴厚度
            axes_points = []
            axes_colors = []
            
            # 计算坐标轴密度
            axis_line_density = wireframe_density
            axis_thickness_points = max(3, wireframe_density // 15)  # 厚度方向的点数
            
            # X轴 - 红色（粗线条）- 从点云中心开始
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
                # 箭头点
                arrow_x = origin_center[0] + arrow_base + t * arrow_length
                arrow_offset = (1 - t) * axis_thickness * 2
                
                axes_points.append([arrow_x, origin_center[1] + arrow_offset, origin_center[2]])
                axes_points.append([arrow_x, origin_center[1] - arrow_offset, origin_center[2]])
                axes_points.append([arrow_x, origin_center[1], origin_center[2] + arrow_offset])
                axes_points.append([arrow_x, origin_center[1], origin_center[2] - arrow_offset])
                axes_colors.extend([[255, 0, 0, 255]] * 4)
            
            # Y轴 - 绿色（粗线条）- 从点云中心开始
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
            
            # Z轴 - 蓝色（粗线条）- 从点云中心开始
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
            
            processing_log.append(f"坐标轴生成: {len(axes_vertices):,} 个点 (长度={axis_length:.3f}, 原点=[{origin_center[0]:.3f}, {origin_center[1]:.3f}, {origin_center[2]:.3f}])")
            
            return axes_vertices, axes_colors_array
            
        except Exception as e:
            processing_log.append(f"创建坐标轴点云失败: {str(e)}")
            return np.array([]), np.array([])
    
    def _generate_output_path(self, output_filename):
        """生成输出文件路径"""
        # 确保文件名有正确的扩展名
        if not output_filename.lower().endswith('.glb'):
            output_filename += '.glb'
        
        # 生成输出路径
        output_dir = folder_paths.get_output_directory() if FOLDER_PATHS_AVAILABLE else "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 添加时间戳避免文件名冲突
        import time
        timestamp = str(int(time.time()))
        name_parts = output_filename.rsplit('.', 1)
        if len(name_parts) == 2:
            timestamped_filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
        else:
            timestamped_filename = f"{output_filename}_{timestamp}"
        
        return os.path.join(output_dir, timestamped_filename)

