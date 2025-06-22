from .common import *

class GLBPointCloudDensityFilter:
    """GLB点云密度过滤器 - 根据局部密度删除稀疏区域，保留密度最高的核心区域"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_file_path": ("STRING", {
                    "default": "",
                    "tooltip": "GLB点云文件路径：输入需要进行密度过滤的GLB格式点云文件。支持绝对路径或相对于ComfyUI output目录的相对路径"
                }),
            },
            "optional": {
                "density_threshold": ("FLOAT", {
                    "default": 0.99, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "密度阈值：保留密度高于此比例的点。0.3=保留密度前70%的点；0.5=保留密度前50%的点。值越大删除越多稀疏点"
                }),
                "neighborhood_radius": ("FLOAT", {
                    "default": 0.3, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "邻域半径：计算密度时的搜索半径。较小值=精细密度检测；较大值=平滑密度检测。建议0.05-0.2"
                }),
                "min_neighbors": ("INT", {
                    "default": 5, "min": 1, "max": 50, "step": 1,
                    "tooltip": "最小邻居数：一个点在邻域内至少需要的邻居点数量才不被视为稀疏点。值越大要求密度越高"
                }),
                "preserve_core_percentage": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "核心区域保留比例：无论密度如何，都会保留密度最高的这部分点。0.8=保证保留最密集的80%区域"
                }),
                "use_adaptive_radius": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "自适应半径：根据点云整体尺寸自动调整邻域半径，确保在不同尺寸的点云上都有良好效果"
                }),
                "output_filename": ("STRING", {
                    "default": "density_filtered_pointcloud",
                    "tooltip": "输出GLB文件名：密度过滤后点云的保存文件名，系统会自动添加.glb扩展名"
                }),
            }
        }

    RETURN_TYPES = (
        "STRING",    # 处理后的GLB文件路径
        "STRING",    # 过滤统计信息JSON
    )
    RETURN_NAMES = (
        "filtered_glb_path",
        "filter_stats",
    )
    OUTPUT_TOOLTIPS = [
        "密度过滤后的GLB文件完整路径",
        "过滤统计信息JSON：包含原始点数、保留点数、删除点数等统计数据",
    ]
    OUTPUT_NODE = True
    FUNCTION = "filter_by_density"
    CATEGORY = "💃VVL/Point Cloud Filtering"

    def filter_by_density(self,
                         glb_file_path: str,
                         density_threshold: float = 0.3,
                         neighborhood_radius: float = 0.1,
                         min_neighbors: int = 5,
                         preserve_core_percentage: float = 0.8,
                         use_adaptive_radius: bool = True,
                         output_filename: str = "density_filtered_pointcloud"):
        """
        根据局部密度过滤GLB点云文件，删除稀疏区域
        """
        
        processing_log = []
        processing_log.append("开始GLB点云密度过滤...")
        
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
                    return ("", "")
            
            if not point_clouds:
                error_msg = "GLB文件中没有点云数据"
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
            
            # 处理每个点云
            filtered_point_clouds = []
            total_original_points = 0
            total_filtered_points = 0
            
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
                
                # 应用密度过滤（仅基于密度，不改变顶点坐标）
                keep_mask, filter_info = self._apply_density_filter(
                    original_vertices, colors, density_threshold, neighborhood_radius, min_neighbors,
                    preserve_core_percentage, use_adaptive_radius, processing_log
                )
                
                # 应用掩码，保留原始的顶点坐标
                filtered_vertices = original_vertices[keep_mask]
                filtered_colors = colors[keep_mask] if colors is not None else None
                
                filtered_count = len(filtered_vertices)
                total_filtered_points += filtered_count
                
                processing_log.append(f"点云 {name}: 原始 {original_count} -> 保留 {filtered_count} (删除 {original_count - filtered_count})")
                
                # 创建过滤后的点云，保留所有原始属性
                if filtered_count > 0:
                    # 使用原始点云作为模板，只更新顶点和颜色
                    filtered_pc = point_cloud.copy()
                    filtered_pc.vertices = filtered_vertices
                    if filtered_colors is not None:
                        filtered_pc.visual.vertex_colors = filtered_colors
                    
                    filtered_point_clouds.append((name, filtered_pc))
                    processing_log.append(f"保留点云 {name} 的所有原始属性和状态")
                else:
                    processing_log.append(f"警告: 点云 {name} 过滤后没有剩余点")
            
            # 直接修改原始场景，保持所有几何信息不变
            if filtered_point_clouds:
                # 使用原始场景作为基础，直接替换点云数据
                new_scene = scene
                
                # 直接修改原始场景中的点云几何体，保持变换不变
                for name, filtered_pc in filtered_point_clouds:
                    if isinstance(scene, trimesh.Scene):
                        # 获取原始几何体
                        if name in scene.geometry:
                            original_geometry = scene.geometry[name]
                            
                            # 如果是点云，直接修改顶点和颜色
                            if isinstance(original_geometry, trimesh.PointCloud):
                                original_geometry.vertices = filtered_pc.vertices
                                if hasattr(filtered_pc.visual, 'vertex_colors') and filtered_pc.visual.vertex_colors is not None:
                                    original_geometry.visual.vertex_colors = filtered_pc.visual.vertex_colors
                                
                                processing_log.append(f"直接修改原始点云 {name}，保持所有变换和属性不变")
                            else:
                                # 如果不是点云类型，替换整个几何体但保持变换
                                scene.delete_geometry(name)
                                scene.add_geometry(filtered_pc, node_name=name)
                                processing_log.append(f"替换几何体 {name}，尝试保持变换")
                        else:
                            # 新增几何体
                            scene.add_geometry(filtered_pc, node_name=name)
                            processing_log.append(f"添加新点云 {name}")
                    else:
                        # 单个几何体的情况
                        new_scene = filtered_pc
                        processing_log.append("单个几何体，直接使用过滤后的点云")
                
                processing_log.append("直接修改原始场景，所有变换矩阵和几何属性完全保持不变")
                
                # 生成输出路径
                output_path = self._generate_output_path(output_filename)
                processing_log.append(f"输出文件路径: {output_path}")
                
                # 保存过滤后的GLB文件
                processing_log.append("正在保存过滤后的GLB文件...")
                new_scene.export(output_path)
                
                # 验证文件是否保存成功
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    processing_log.append(f"GLB文件保存成功，文件大小: {file_size} bytes")
                    
                    # 生成统计信息
                    filter_stats = {
                        "original_points": total_original_points,
                        "filtered_points": total_filtered_points,
                        "removed_points": total_original_points - total_filtered_points,
                        "retention_rate": total_filtered_points / total_original_points if total_original_points > 0 else 0,
                        "parameters": {
                            "density_threshold": density_threshold,
                            "neighborhood_radius": neighborhood_radius,
                            "min_neighbors": min_neighbors,
                            "preserve_core_percentage": preserve_core_percentage,
                            "use_adaptive_radius": use_adaptive_radius
                        }
                    }
                    
                    stats_json = json.dumps(filter_stats, indent=2)
                    
                    processing_log.append("GLB点云密度过滤完成! 已保留原始模型的大小、朝向和几何属性")
                    processing_log.append(f"统计: 原始{total_original_points}点 -> 保留{total_filtered_points}点 (保留率: {filter_stats['retention_rate']:.1%})")
                    
                    return (output_path, stats_json)
                else:
                    error_msg = "GLB文件保存失败"
                    processing_log.append(f"错误: {error_msg}")
                    return ("", "")
            else:
                error_msg = "过滤后没有剩余的点云数据"
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
                
        except Exception as e:
            error_msg = f"处理GLB文件时发生错误: {str(e)}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return ("", "")
    
    def _apply_density_filter(self, vertices, colors, density_threshold, neighborhood_radius, 
                             min_neighbors, preserve_core_percentage, use_adaptive_radius, processing_log):
        """
        应用密度过滤算法，返回保留点的掩码
        """
        
        n_points = len(vertices)
        processing_log.append(f"开始密度分析，总点数: {n_points}")
        
        # 自适应半径调整
        if use_adaptive_radius:
            # 计算点云的包围盒对角线长度
            bbox_min = np.min(vertices, axis=0)
            bbox_max = np.max(vertices, axis=0)
            bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
            
            # 根据包围盒大小调整半径
            adaptive_radius = neighborhood_radius * bbox_diagonal / 10.0  # 可调整的比例因子
            processing_log.append(f"自适应半径: {neighborhood_radius} -> {adaptive_radius:.4f} (基于包围盒对角线 {bbox_diagonal:.4f})")
            neighborhood_radius = adaptive_radius
        
        # 使用KDTree进行高效的邻域搜索
        try:
            from scipy.spatial import cKDTree
            kdtree = cKDTree(vertices)
            processing_log.append("使用scipy.spatial.cKDTree进行邻域搜索")
        except ImportError:
            processing_log.append("scipy不可用，使用简化的距离计算")
            kdtree = None
        
        # 计算每个点的局部密度
        densities = np.zeros(n_points)
        
        BIG_POINT_THRESHOLD = 200000  # 20万点以上使用体素统计
        if n_points > BIG_POINT_THRESHOLD:
            # ------------------------------------------------------------------
            # 大规模点云：使用体素统计近似密度 (O(N) 内存, 非递归, 无KDTree)
            # ------------------------------------------------------------------
            start_t = time.time()
            bbox_min = vertices.min(axis=0)
            voxel_size = neighborhood_radius  # 体素边长与邻域半径一致
            voxel_indices = np.floor((vertices - bbox_min) / voxel_size).astype(np.int32)
            # 使用结构化dtype便于unique
            voxel_keys = voxel_indices.view([('x', np.int32), ('y', np.int32), ('z', np.int32)]).reshape(-1)
            unique_keys, inverse_indices, counts = np.unique(voxel_keys, return_inverse=True, return_counts=True)
            densities = counts[inverse_indices] - 1  # 同体素内点数近似邻居数
            elapsed = time.time() - start_t
            processing_log.append(f"体素密度计算完成，用时 {elapsed:.2f}s (体素数={len(unique_keys):,})")
        elif kdtree is not None:
            # -----------------------------
            # 使用KDTree批量并行查询提高速度
            # -----------------------------
            start_t = time.time()
            neighbors_list = kdtree.query_ball_point(vertices, neighborhood_radius, workers=-1)
            densities = np.fromiter((len(idx) - 1 for idx in neighbors_list), dtype=np.int32, count=n_points)
            elapsed = time.time() - start_t
            processing_log.append(f"KDTree密度计算完成，用时 {elapsed:.2f}s (并行查询)")
        else:
            # -----------------------------
            # 回退到简化的距离计算（仅在小点云或无SciPy时使用）
            # -----------------------------
            start_t = time.time()
            # 采用向量化广播一次性计算距离矩阵（O(N^2) 内存消耗大，仅限<50k点）
            if n_points < 50000:
                dist_matrix = np.linalg.norm(vertices[None, :, :] - vertices[:, None, :], axis=2)
                densities = (dist_matrix <= neighborhood_radius).sum(axis=1) - 1  # 排除自身
            else:
                # 大点云时退化为分批循环
                for i in range(n_points):
                    distances = np.linalg.norm(vertices - vertices[i], axis=1)
                    neighbor_count = np.sum(distances <= neighborhood_radius) - 1
                    densities[i] = neighbor_count
            elapsed = time.time() - start_t
            processing_log.append(f"向量化密度计算完成，用时 {elapsed:.2f}s")
        
        processing_log.append(f"密度统计: 最小={densities.min():.1f}, 最大={densities.max():.1f}, 平均={densities.mean():.1f}")
        
        # 应用多层过滤策略
        keep_mask = np.ones(n_points, dtype=bool)
        
        # 1. 基于最小邻居数的过滤
        min_neighbor_mask = densities >= min_neighbors
        removed_by_min_neighbors = np.sum(~min_neighbor_mask)
        keep_mask &= min_neighbor_mask
        processing_log.append(f"最小邻居数过滤: 删除 {removed_by_min_neighbors} 个点")
        
        # 2. 基于密度阈值的过滤
        if density_threshold > 0 and np.sum(keep_mask) > 0:
            valid_densities = densities[keep_mask]
            density_percentile = np.percentile(valid_densities, (1 - density_threshold) * 100)
            density_mask = densities >= density_percentile
            
            # 保留当前mask中的点，再应用密度过滤
            temp_mask = keep_mask.copy()
            keep_mask[temp_mask] &= density_mask[temp_mask]
            
            removed_by_density = np.sum(temp_mask) - np.sum(keep_mask)
            processing_log.append(f"密度阈值过滤: 删除 {removed_by_density} 个点 (阈值密度: {density_percentile:.1f})")
        
        # 3. 核心区域保护
        if preserve_core_percentage < 1.0 and np.sum(keep_mask) > 0:
            # 计算需要保留的核心点数
            target_core_points = int(n_points * preserve_core_percentage)
            current_points = np.sum(keep_mask)
            
            if current_points < target_core_points:
                # 当前保留的点数少于核心要求，需要添加高密度点
                need_more_points = target_core_points - current_points
                
                # 找到密度最高的点
                excluded_indices = np.where(~keep_mask)[0]
                if len(excluded_indices) > 0:
                    excluded_densities = densities[excluded_indices]
                    # 按密度排序，选择密度最高的点
                    sorted_indices = excluded_indices[np.argsort(-excluded_densities)]
                    points_to_restore = min(need_more_points, len(sorted_indices))
                    
                    keep_mask[sorted_indices[:points_to_restore]] = True
                    processing_log.append(f"核心区域保护: 恢复 {points_to_restore} 个高密度点")
        
        # 统计信息
        original_count = n_points
        filtered_count = np.sum(keep_mask)
        removed_count = original_count - filtered_count
        
        filter_info = {
            "original_points": original_count,
            "filtered_points": filtered_count,
            "removed_points": removed_count,
            "retention_rate": filtered_count / original_count if original_count > 0 else 0,
            "density_stats": {
                "min": float(densities.min()),
                "max": float(densities.max()),
                "mean": float(densities.mean()),
                "std": float(densities.std())
            }
        }
        
        processing_log.append(f"密度过滤完成: 保留 {filtered_count}/{original_count} 点 (保留率: {filter_info['retention_rate']:.1%})")
        
        return keep_mask, filter_info
    
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


