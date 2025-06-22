from .common import *

class GLBPointCloudBlackDelete:
    """GLB点云文件处理器 - 专门用于删除黑色点和暗色点优化"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_file_path": ("STRING", {
                    "default": "",
                    "tooltip": "GLB点云文件路径：输入需要处理的GLB格式点云文件路径。支持绝对路径(如C:/path/to/file.glb)或相对于ComfyUI output目录的相对路径(如pointcloud.glb)。GLB文件应包含点云数据，支持带颜色信息的点云处理"
                }),
            },
            "optional": {
                "black_threshold": ("INT", {
                    "default": 30, "min": 0, "max": 255, "step": 1,
                    "tooltip": "黑色阈值控制：RGB三色总和小于此值的点将被删除。0=仅删除纯黑点(0,0,0)；30=删除深黑色点；60=删除较暗点；100=删除中等暗度点；255=删除所有点。调大=删除更多暗色点，点云更亮净；调小=保留更多暗色细节，但可能有噪点"
                }),
                "output_filename": ("STRING", {
                    "default": "clean_pointcloud",
                    "tooltip": "输出GLB文件名：处理后点云的保存文件名，系统会自动添加.glb扩展名。文件将保存到ComfyUI的output目录中。建议使用描述性名称如'clean_pointcloud'、'filtered_scan'等便于识别"
                }),
            }
        }

    RETURN_TYPES = (
        "STRING",    # 处理后的GLB文件路径
    )
    RETURN_NAMES = (
        "processed_glb_path",
    )
    OUTPUT_TOOLTIPS = [
        "处理后的GLB文件完整路径",
    ]
    OUTPUT_NODE = True
    FUNCTION = "process_point_cloud"
    CATEGORY = "💃VVL/Point Cloud Cleaning"

    def process_point_cloud(self,
                          glb_file_path: str,
                          black_threshold: int = 30,
                          output_filename: str = "clean_pointcloud"):
        """
        处理GLB点云文件，删除黑色点和暗色点
        """
        
        processing_log = []
        processing_log.append("开始GLB点云黑色点清理...")
        
        # 检查依赖
        if not TRIMESH_AVAILABLE:
            error_msg = "trimesh库不可用，无法处理GLB文件"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("",)
        
        # 验证输入文件路径
        if not glb_file_path or not glb_file_path.strip():
            error_msg = "GLB文件路径为空"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("",)
        
        # 处理文件路径
        input_path = self._resolve_file_path(glb_file_path.strip())
        processing_log.append(f"输入文件路径: {input_path}")
        
        if not os.path.exists(input_path):
            error_msg = f"GLB文件不存在: {input_path}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            return ("",)
        
        try:
            # 加载GLB文件
            processing_log.append("正在加载GLB文件...")
            scene = trimesh.load(input_path)
            
            # 提取点云数据
            point_clouds = []
            other_geometries = []
            
            if isinstance(scene, trimesh.Scene):
                for name, geometry in scene.geometry.items():
                    if isinstance(geometry, trimesh.PointCloud):
                        point_clouds.append((name, geometry))
                        processing_log.append(f"发现点云: {name}, 点数: {len(geometry.vertices)}")
                    else:
                        other_geometries.append((name, geometry))
                        processing_log.append(f"发现其他几何体: {name}, 类型: {type(geometry).__name__}")
            elif isinstance(scene, trimesh.PointCloud):
                point_clouds.append(("main_pointcloud", scene))
                processing_log.append(f"发现点云: main_pointcloud, 点数: {len(scene.vertices)}")
            else:
                # 尝试转换为点云
                if hasattr(scene, 'vertices'):
                    pc = trimesh.PointCloud(vertices=scene.vertices, colors=getattr(scene.visual, 'vertex_colors', None))
                    point_clouds.append(("converted_pointcloud", pc))
                    processing_log.append(f"转换为点云: converted_pointcloud, 点数: {len(pc.vertices)}")
                else:
                    error_msg = "GLB文件中未找到点云数据"
                    processing_log.append(f"错误: {error_msg}")
                    return ("",)
            
            if not point_clouds:
                error_msg = "GLB文件中没有点云数据"
                processing_log.append(f"错误: {error_msg}")
                return ("",)
            
            # 处理每个点云
            processed_point_clouds = []
            total_original_points = 0
            total_removed_points = 0
            
            for name, point_cloud in point_clouds:
                processing_log.append(f"\n处理点云: {name}")
                original_count = len(point_cloud.vertices)
                total_original_points += original_count
                
                # 获取原始点云的所有属性和状态
                original_vertices = point_cloud.vertices.copy()
                colors = None
                
                # 获取颜色信息
                if hasattr(point_cloud.visual, 'vertex_colors') and point_cloud.visual.vertex_colors is not None:
                    colors = point_cloud.visual.vertex_colors.copy()
                elif hasattr(point_cloud, 'colors') and point_cloud.colors is not None:
                    colors = point_cloud.colors.copy()
                
                # 初始化掩码（所有点都保留）
                keep_mask = np.ones(len(original_vertices), dtype=bool)
                
                # 删除暗色点（仅基于颜色，不改变顶点坐标）
                if colors is not None:
                    removed_count = self._remove_dark_points(original_vertices, colors, keep_mask, black_threshold, processing_log)
                    processing_log.append(f"删除黑色点: {removed_count} 个")
                else:
                    processing_log.append("跳过黑色点过滤: 点云无颜色信息")
                
                # 应用掩码，保留原始的顶点坐标
                filtered_vertices = original_vertices[keep_mask]
                filtered_colors = colors[keep_mask] if colors is not None else None
                
                remaining_count = len(filtered_vertices)
                removed_count = original_count - remaining_count
                total_removed_points += removed_count
                
                processing_log.append(f"点云 {name}: 原始 {original_count} -> 剩余 {remaining_count} (删除 {removed_count})")
                
                # 创建处理后的点云，保留所有原始属性
                if remaining_count > 0:
                    # 使用原始点云作为模板，只更新顶点和颜色
                    processed_pc = point_cloud.copy()
                    processed_pc.vertices = filtered_vertices
                    if filtered_colors is not None:
                        processed_pc.visual.vertex_colors = filtered_colors
                    
                    processed_point_clouds.append((name, processed_pc))
                    processing_log.append(f"保留点云 {name} 的所有原始属性和状态")
                else:
                    processing_log.append(f"警告: 点云 {name} 处理后没有剩余点")
            
            # 直接修改原始场景，保持所有几何信息不变
            if processed_point_clouds:
                # 使用原始场景作为基础，直接替换点云数据
                new_scene = scene
                
                # 直接修改原始场景中的点云几何体，保持变换不变
                for name, processed_pc in processed_point_clouds:
                    if isinstance(scene, trimesh.Scene):
                        # 获取原始几何体
                        if name in scene.geometry:
                            original_geometry = scene.geometry[name]
                            
                            # 如果是点云，直接修改顶点和颜色
                            if isinstance(original_geometry, trimesh.PointCloud):
                                original_geometry.vertices = processed_pc.vertices
                                if hasattr(processed_pc.visual, 'vertex_colors') and processed_pc.visual.vertex_colors is not None:
                                    original_geometry.visual.vertex_colors = processed_pc.visual.vertex_colors
                                
                                processing_log.append(f"直接修改原始点云 {name}，保持所有变换和属性不变")
                            else:
                                # 如果不是点云类型，替换整个几何体但保持变换
                                scene.delete_geometry(name)
                                scene.add_geometry(processed_pc, node_name=name)
                                processing_log.append(f"替换几何体 {name}，尝试保持变换")
                        else:
                            # 新增几何体
                            scene.add_geometry(processed_pc, node_name=name)
                            processing_log.append(f"添加新点云 {name}")
                    else:
                        # 单个几何体的情况
                        new_scene = processed_pc
                        processing_log.append("单个几何体，直接使用处理后的点云")
                
                processing_log.append("直接修改原始场景，所有变换矩阵和几何属性完全保持不变")
                
                # 生成输出路径
                output_path = self._generate_output_path(output_filename)
                processing_log.append(f"输出文件路径: {output_path}")
                
                # 保存处理后的GLB文件
                processing_log.append("正在保存处理后的GLB文件...")
                new_scene.export(output_path)
                
                # 验证文件是否保存成功
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    processing_log.append(f"GLB文件保存成功，文件大小: {file_size} bytes")
                    
                    # 验证输出文件的一致性
                    try:
                        # 加载输出文件进行验证
                        verification_scene = trimesh.load(output_path)
                        processing_log.append("文件验证: 输出文件可正常加载")
                        
                        # 检查是否保留了场景结构
                        if isinstance(scene, trimesh.Scene) and isinstance(verification_scene, trimesh.Scene):
                            processing_log.append(f"场景结构验证: 原始 {len(scene.geometry)} 个对象 -> 输出 {len(verification_scene.geometry)} 个对象")
                        
                    except Exception as e:
                        processing_log.append(f"文件验证警告: {str(e)}")
                    
                    processing_log.append("GLB点云黑色点清理完成! 已保留原始模型的大小、朝向和几何属性")
                    return (output_path,)
                else:
                    error_msg = "GLB文件保存失败"
                    processing_log.append(f"错误: {error_msg}")
                    return ("",)
            else:
                error_msg = "处理后没有剩余的点云数据"
                processing_log.append(f"错误: {error_msg}")
                return ("",)
                
        except Exception as e:
            error_msg = f"处理GLB文件时发生错误: {str(e)}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return ("",)
    
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
        if FOLDER_PATHS_AVAILABLE:
            output_dir = folder_paths.get_output_directory()
        else:
            output_dir = "output"
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理文件名，确保是.glb扩展名
        if not filename.lower().endswith('.glb'):
            filename = f"{filename}.glb"
        
        return os.path.join(output_dir, filename)
    
    def _remove_dark_points(self, vertices: np.ndarray, colors: np.ndarray, keep_mask: np.ndarray, 
                           black_threshold: int, processing_log: List[str]) -> int:
        """删除暗色点"""
        removed_count = 0
        
        # 确保颜色数据格式正确
        if colors.shape[1] == 4:  # RGBA
            rgb_colors = colors[:, :3]
        else:  # RGB
            rgb_colors = colors
        
        # 如果颜色值在0-1范围内，转换为0-255
        if rgb_colors.max() <= 1.0:
            rgb_colors = (rgb_colors * 255).astype(np.uint8)
        else:
            rgb_colors = rgb_colors.astype(np.uint8)
        
        # 计算RGB总和
        rgb_sum = rgb_colors.sum(axis=1)
        
        # 应用黑色阈值过滤
        if black_threshold > 0:
            dark_mask = rgb_sum >= black_threshold
            before_count = keep_mask.sum()
            keep_mask &= dark_mask
            after_count = keep_mask.sum()
            removed_by_black = before_count - after_count
            removed_count += removed_by_black
            processing_log.append(f"  按RGB总和阈值({black_threshold})删除: {removed_by_black} 个点")
        
        return removed_count
    
