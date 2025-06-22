from .common import *


class GLBPointCloudRotationCorrector:
    """GLB点云旋转校正器 - 检测地面倾斜并自动校正为水平，解决视频生成点云地面倾斜问题"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_file_path": ("STRING", {
                    "default": "",
                    "tooltip": "GLB点云文件路径：输入需要旋转校正的GLB格式点云文件。适用于视频生成的倾斜点云文件"
                }),
            },
            "optional": {
                "ground_detection_method": (["height_based", "ransac_full", "lowest_plane", "debug_mode", "simple_test"], {
                    "default": "ransac_full",
                    "tooltip": "地面检测方法：simple_test=简化测试；lowest_plane=检测最低平面（推荐）；debug_mode=调试模式（输出详细信息）；height_based=基于高度快速检测；ransac_full=全点云RANSAC检测"
                }),
                "ground_height_percentile": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01,
                    "tooltip": "地面高度百分位：选择高度最低的这部分点作为地面候选点。0.1=最低10%的点；0.2=最低20%的点"
                }),
                "ransac_iterations": ("INT", {
                    "default": 1000, "min": 100, "max": 10000, "step": 100,
                    "tooltip": "RANSAC迭代次数：平面拟合的迭代次数，越高越精确但耗时更长。推荐500-2000"
                }),
                "ransac_threshold": ("FLOAT", {
                    "default": 0.05, "min": 0.001, "max": 0.5, "step": 0.001,
                    "tooltip": "RANSAC距离阈值：点到平面的最大距离阈值，用于判断内点。值越小拟合越精确"
                }),
                "min_ground_points": ("INT", {
                    "default": 3, "min": 3, "max": 100, "step": 1,
                    "tooltip": "最少地面点数：RANSAC拟合平面需要的最少点数。至少需要3个点"
                }),
                "add_reference_plane": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "添加参考平面：在输出中添加原始地面平面和校正后水平面的可视化"
                }),
                "plane_visualization_size": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "参考平面尺寸：可视化平面的相对大小，1.0=与点云包围盒同等大小"
                }),
                "up_axis": (["Z", "Y", "-Y"], {
                    "default": "Y",
                    "tooltip": "向上的轴：Z=Z轴向上(标准)；Y=Y轴向上(Unity/Maya)；-Y=Y轴向下(某些扫描系统)"
                }),
                "output_filename": ("STRING", {
                    "default": "rotation_corrected_pointcloud",
                    "tooltip": "输出GLB文件名：旋转校正后点云的保存文件名，系统会自动添加.glb扩展名"
                }),
            }
        }

    RETURN_TYPES = (
        "STRING",    # 校正后的GLB文件路径
        "STRING",    # 旋转变换信息JSON
    )
    RETURN_NAMES = (
        "corrected_glb_path",
        "rotation_info",
    )
    OUTPUT_TOOLTIPS = [
        "旋转校正后的GLB文件完整路径",
        "旋转变换信息JSON：包含检测到的地面法向量、旋转角度、变换矩阵等信息",
    ]
    OUTPUT_NODE = True
    FUNCTION = "correct_rotation"
    CATEGORY = "💃VVL/Point Cloud Transform"

    def correct_rotation(self,
                        glb_file_path: str,
                        ground_detection_method: str = "simple_test",
                        ground_height_percentile: float = 0.1,
                        ransac_iterations: int = 1000,
                        ransac_threshold: float = 0.05,
                        min_ground_points: int = 3,
                        add_reference_plane: bool = True,
                        plane_visualization_size: float = 1.0,
                        up_axis: str = "Y",
                        output_filename: str = "rotation_corrected_pointcloud"):
        """
        检测点云地面倾斜并自动校正为水平
        """
        
        processing_log = []
        processing_log.append("开始GLB点云旋转校正...")
        
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
            
            # 收集所有顶点（考虑变换矩阵）
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
            
            # 检测地面平面
            processing_log.append(f"开始地面检测，方法: {ground_detection_method}, 向上轴: {up_axis}")
            ground_normal, ground_point, ground_indices, detection_info = self._detect_ground_plane(
                combined_vertices, ground_detection_method, ground_height_percentile,
                ransac_iterations, ransac_threshold, min_ground_points, up_axis, processing_log
            )
            
            if ground_normal is None:
                error_msg = "无法检测到有效的地面平面"
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
            
            processing_log.append(f"检测到地面法向量: [{ground_normal[0]:.4f}, {ground_normal[1]:.4f}, {ground_normal[2]:.4f}]")
            processing_log.append(f"地面参考点: [{ground_point[0]:.4f}, {ground_point[1]:.4f}, {ground_point[2]:.4f}]")
            processing_log.append(f"地面点数量: {len(ground_indices) if ground_indices is not None else 'N/A'}")
            
            # 计算旋转变换
            processing_log.append("计算旋转变换...")
            rotation_matrix, rotation_angles, transform_matrix = self._calculate_rotation_transform(
                ground_normal, ground_point, up_axis, processing_log
            )
            
            processing_log.append(f"旋转角度 (度): X={np.degrees(rotation_angles[0]):.2f}, Y={np.degrees(rotation_angles[1]):.2f}, Z={np.degrees(rotation_angles[2]):.2f}")
            
            # 应用旋转变换到所有几何体
            processing_log.append("应用旋转变换...")
            corrected_scene = self._apply_rotation_transform(
                scene, geometries_info, transform_matrix, processing_log
            )
            
            # 验证校正结果：检查校正后的地面是否水平
            processing_log.append("验证校正结果...")
            self._verify_correction_result(corrected_scene, processing_log)
            
            # 添加参考平面可视化（如果需要）
            if add_reference_plane:
                try:
                    self._add_reference_planes(corrected_scene, combined_vertices, ground_normal, 
                                             ground_point, rotation_matrix, plane_visualization_size, processing_log)
                except Exception as e:
                    processing_log.append(f"参考平面添加失败: {str(e)}")
            
            # 生成输出路径并保存
            output_path = self._generate_output_path(output_filename)
            processing_log.append(f"输出文件路径: {output_path}")
            
            processing_log.append("正在保存校正后的GLB文件...")
            corrected_scene.export(output_path)
            
            # 验证保存
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                processing_log.append(f"GLB文件保存成功，文件大小: {file_size} bytes")
            else:
                error_msg = "GLB文件保存失败"
                processing_log.append(f"错误: {error_msg}")
                return ("", "")
            
            # 生成旋转信息JSON
            rotation_info = {
                "detection_method": ground_detection_method,
                "original_ground_normal": ground_normal.tolist(),
                "ground_reference_point": ground_point.tolist(),
                "rotation_angles_degrees": {
                    "x": float(np.degrees(rotation_angles[0])),
                    "y": float(np.degrees(rotation_angles[1])),
                    "z": float(np.degrees(rotation_angles[2]))
                },
                "rotation_matrix": rotation_matrix.tolist(),
                "transform_matrix": transform_matrix.tolist(),
                "detection_stats": detection_info,
                "corrected_to_horizontal": True
            }
            
            rotation_json = json.dumps(rotation_info, indent=2)
            
            processing_log.append("GLB点云旋转校正完成! 地面已校正为水平")
            
            return (output_path, rotation_json)
                
        except Exception as e:
            error_msg = f"旋转校正时发生错误: {str(e)}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return ("", "")
    
    def _detect_ground_plane(self, vertices, method, height_percentile, ransac_iterations, 
                           ransac_threshold, min_ground_points, up_axis, processing_log):
        """检测地面平面"""
        
        # 首先分析点云的分布情况
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        ranges = max_coords - min_coords
        processing_log.append(f"点云包围盒: X=[{min_coords[0]:.3f}, {max_coords[0]:.3f}], Y=[{min_coords[1]:.3f}, {max_coords[1]:.3f}], Z=[{min_coords[2]:.3f}, {max_coords[2]:.3f}]")
        processing_log.append(f"各轴范围: X={ranges[0]:.3f}, Y={ranges[1]:.3f}, Z={ranges[2]:.3f}")
        
        # 根据up_axis确定高度轴索引
        if up_axis == "Z":
            height_axis = 2  # Z轴
            axis_name = "Z"
        elif up_axis == "Y":
            height_axis = 1  # Y轴
            axis_name = "Y"
        elif up_axis == "-Y":
            height_axis = 1  # Y轴，但方向相反
            axis_name = "-Y"
        else:
            height_axis = 2  # 默认Z轴
            axis_name = "Z"
        
        processing_log.append(f"使用 {axis_name} 轴作为高度轴")
        
        if method == "lowest_plane":
            # 最低平面检测方法（推荐）
            processing_log.append("使用最低平面检测方法")
            
            # 找到最低的点集
            height_values = vertices[:, height_axis]
            if up_axis == "-Y":
                # 如果Y轴向下，我们需要找最高的点作为地面
                height_min = np.max(height_values)
                ground_mask = height_values >= (height_min - ranges[height_axis] * 0.05)
            else:
                # 正常情况，找最低的点
                height_min = np.min(height_values)
                ground_mask = height_values <= (height_min + ranges[height_axis] * 0.05)
            
            # 使用更严格的高度阈值来找地面
            height_tolerance = ranges[height_axis] * 0.05  # 5%的高度容差
            ground_candidates = vertices[ground_mask]
            
            processing_log.append(f"最低点高度({axis_name}): {height_min:.4f}")
            processing_log.append(f"高度容差: {height_tolerance:.4f}")
            processing_log.append(f"地面候选点: {len(ground_candidates):,} 个")
            
            if len(ground_candidates) < min_ground_points:
                processing_log.append(f"地面候选点不足，尝试扩大搜索范围")
                # 扩大搜索范围
                height_tolerance = ranges[2] * 0.1  # 10%的高度容差
                ground_mask = z_values <= (z_min + height_tolerance)
                ground_candidates = vertices[ground_mask]
                processing_log.append(f"扩大后地面候选点: {len(ground_candidates):,} 个")
                
                if len(ground_candidates) < min_ground_points:
                    processing_log.append(f"地面候选点仍然不足 (需要至少{min_ground_points}个)")
                    return None, None, None, {}
            
            # 对候选点使用RANSAC拟合平面
            best_normal, best_point, best_inliers, ransac_info = self._ransac_plane_fitting(
                ground_candidates, ransac_iterations, ransac_threshold, min_ground_points, processing_log
            )
            
        elif method == "height_based":
            # 基于高度的快速地面检测
            processing_log.append("使用基于高度的地面检测方法")
            
            # 计算高度范围
            z_values = vertices[:, 2]  # 假设Z是高度轴
            z_min = np.min(z_values)
            z_max = np.max(z_values)
            z_range = z_max - z_min
            
            processing_log.append(f"高度范围: {z_min:.4f} ~ {z_max:.4f} (范围: {z_range:.4f})")
            
            # 选择高度最低的点作为地面候选点
            height_threshold = np.percentile(z_values, height_percentile * 100)
            ground_candidates = vertices[z_values <= height_threshold]
            
            processing_log.append(f"地面候选点 (高度<={height_threshold:.4f}): {len(ground_candidates):,} 个")
            
            if len(ground_candidates) < min_ground_points:
                processing_log.append(f"地面候选点不足 (需要至少{min_ground_points}个)")
                return None, None, None, {}
            
            # 对候选点使用RANSAC拟合平面
            best_normal, best_point, best_inliers, ransac_info = self._ransac_plane_fitting(
                ground_candidates, ransac_iterations, ransac_threshold, min_ground_points, processing_log
            )
            
        elif method == "debug_mode":
            # 调试模式：尝试所有三个主轴方向，选择最合理的
            processing_log.append("使用调试模式地面检测方法")
            
            # 分别试试每个轴作为"高度"轴
            best_result = None
            best_score = 0
            
            for axis_idx, axis_name in enumerate(['X', 'Y', 'Z']):
                processing_log.append(f"\n--- 尝试 {axis_name} 轴作为高度轴 ---")
                
                # 使用当前轴作为高度
                axis_values = vertices[:, axis_idx]
                min_val = np.min(axis_values)
                max_val = np.max(axis_values)
                range_val = max_val - min_val
                
                processing_log.append(f"{axis_name}轴范围: {min_val:.4f} ~ {max_val:.4f} (范围: {range_val:.4f})")
                
                # 选择最低的5%作为地面候选
                height_tolerance = range_val * 0.05
                ground_mask = axis_values <= (min_val + height_tolerance)
                ground_candidates = vertices[ground_mask]
                
                processing_log.append(f"{axis_name}轴地面候选点: {len(ground_candidates):,} 个")
                
                if len(ground_candidates) >= min_ground_points:
                    # 尝试RANSAC拟合
                    normal, point, inliers, ransac_info = self._ransac_plane_fitting(
                        ground_candidates, ransac_iterations//3, ransac_threshold, min_ground_points, processing_log
                    )
                    
                    if normal is not None:
                        # 评估这个结果的质量
                        score = ransac_info.get('best_inliers_count', 0)
                        processing_log.append(f"{axis_name}轴拟合结果: {score} 个内点, 法向量: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
                        
                        if score > best_score:
                            best_score = score
                            best_result = (normal, point, inliers, ransac_info, axis_name)
                            processing_log.append(f"*** {axis_name}轴目前是最佳结果 ***")
            
            if best_result:
                best_normal, best_point, best_inliers, ransac_info, best_axis = best_result
                processing_log.append(f"\n最终选择: {best_axis}轴, {best_score} 个内点")
            else:
                processing_log.append("\n调试模式：所有轴都无法找到有效地面")
                return None, None, None, {}
                
        elif method == "simple_test":
            # 简化测试：假设地面就是最低的10%点，强制设定法向量
            processing_log.append("使用简化测试模式")
            
            z_values = vertices[:, 2]
            z_min = np.min(z_values)
            height_tolerance = ranges[2] * 0.1
            ground_mask = z_values <= (z_min + height_tolerance)
            ground_candidates = vertices[ground_mask]
            
            processing_log.append(f"简化测试：选择最低10%的点作为地面: {len(ground_candidates):,} 个")
            
            if len(ground_candidates) >= 3:
                # 使用PCA找主方向
                centroid = np.mean(ground_candidates, axis=0)
                centered = ground_candidates - centroid
                cov_matrix = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                
                # 最小特征值对应的特征向量就是法向量
                best_normal = eigenvectors[:, np.argmin(eigenvalues)]
                best_point = centroid
                
                processing_log.append(f"PCA地面法向量: [{best_normal[0]:.4f}, {best_normal[1]:.4f}, {best_normal[2]:.4f}]")
                
                ransac_info = {
                    "method": "simple_pca",
                    "ground_points": len(ground_candidates),
                    "eigenvalues": eigenvalues.tolist()
                }
                
                best_inliers = np.arange(len(ground_candidates))
            else:
                processing_log.append("简化测试：地面点不足")
                return None, None, None, {}
                
        else:  # ransac_full
            # 全点云RANSAC检测
            processing_log.append("使用全点云RANSAC地面检测方法")
            
            best_normal, best_point, best_inliers, ransac_info = self._ransac_plane_fitting(
                vertices, ransac_iterations, ransac_threshold, min_ground_points, processing_log
            )
        
        if best_normal is None:
            return None, None, None, {}
        
        # 确保法向量向上 (如果法向量向下，翻转它)
        if up_axis == "Z" and best_normal[2] < 0:
            best_normal = -best_normal
            processing_log.append("法向量已翻转为向上方向(Z轴)")
        elif up_axis == "Y" and best_normal[1] < 0:
            best_normal = -best_normal
            processing_log.append("法向量已翻转为向上方向(Y轴)")
        elif up_axis == "-Y" and best_normal[1] > 0:
            best_normal = -best_normal
            processing_log.append("法向量已翻转为向下方向(-Y轴)")
        
        # 调试信息：输出检测到的法向量
        processing_log.append(f"最终地面法向量: [{best_normal[0]:.4f}, {best_normal[1]:.4f}, {best_normal[2]:.4f}]")
        
        # 检查法向量是否合理（地面法向量的Z分量应该是主要的）
        normal_magnitude = np.linalg.norm(best_normal)
        processing_log.append(f"法向量模长: {normal_magnitude:.4f}")
        
        # 计算法向量与垂直向量的夹角
        if up_axis == "Z":
            vertical_angle = np.arccos(np.clip(best_normal[2], -1.0, 1.0))
            axis_label = "(0,0,1)"
        elif up_axis == "Y":
            vertical_angle = np.arccos(np.clip(best_normal[1], -1.0, 1.0))
            axis_label = "(0,1,0)"
        elif up_axis == "-Y":
            vertical_angle = np.arccos(np.clip(-best_normal[1], -1.0, 1.0))
            axis_label = "(0,-1,0)"
        else:
            vertical_angle = np.arccos(np.clip(best_normal[2], -1.0, 1.0))
            axis_label = "(0,0,1)"
            
        processing_log.append(f"地面法向量与垂直轴{axis_label}的夹角: {np.degrees(vertical_angle):.2f} 度")
        
        # 如果角度太大，可能检测错误
        if np.degrees(vertical_angle) > 60:
            processing_log.append("⚠ 警告：检测到的地面法向量可能不正确（倾斜角度>60度）")
        
        detection_info = {
            "method": method,
            "ground_candidates_count": len(ground_candidates) if method == "height_based" else len(vertices),
            "inliers_count": len(best_inliers) if best_inliers is not None else 0,
            "ransac_stats": ransac_info
        }
        
        return best_normal, best_point, best_inliers, detection_info
    
    def _ransac_plane_fitting(self, points, max_iterations, distance_threshold, min_points, processing_log):
        """使用RANSAC算法拟合平面"""
        
        if len(points) < min_points:
            processing_log.append(f"点数不足进行RANSAC拟合 (需要至少{min_points}个)")
            return None, None, None, {}
        
        # 对于大规模点云，先进行采样以提高性能
        original_point_count = len(points)
        max_ransac_points = 50000  # RANSAC处理的最大点数
        
        if len(points) > max_ransac_points:
            # 采样点云进行RANSAC
            sample_indices = np.random.choice(len(points), max_ransac_points, replace=False)
            ransac_points = points[sample_indices]
            processing_log.append(f"RANSAC采样: 从 {original_point_count:,} 个点中采样 {max_ransac_points:,} 个点进行拟合")
        else:
            ransac_points = points
            sample_indices = None
        
        best_inliers = []
        best_normal = None
        best_point = None
        best_score = 0
        
        processing_log.append(f"开始RANSAC平面拟合: {len(ransac_points):,} 个点, {max_iterations} 次迭代")
        
        np.random.seed(42)  # 确保结果可重现
        
        for iteration in range(max_iterations):
            # 随机选择3个点
            point_sample_indices = np.random.choice(len(ransac_points), 3, replace=False)
            sample_points = ransac_points[point_sample_indices]
            
            # 计算平面法向量
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)
            
            # 标准化法向量
            normal_length = np.linalg.norm(normal)
            if normal_length < 1e-8:
                continue  # 三点共线，跳过
            normal = normal / normal_length
            
            # 平面上的参考点
            plane_point = sample_points[0]
            
            # 计算所有RANSAC点到平面的距离
            distances = np.abs(np.dot(ransac_points - plane_point, normal))
            
            # 统计内点
            inliers = distances <= distance_threshold
            inlier_count = np.sum(inliers)
            
            # 更新最佳结果
            if inlier_count > best_score:
                best_score = inlier_count
                best_normal = normal.copy()
                best_point = plane_point.copy()
                best_inliers = np.where(inliers)[0]
        
        if best_score < min_points:
            processing_log.append(f"RANSAC拟合失败: 最佳结果仅有 {best_score} 个内点")
            return None, None, None, {}
        
        # 使用内点重新拟合平面（提高精度）
        if len(best_inliers) >= 3:
            # 如果使用了采样，需要映射回原始点集的内点
            if sample_indices is not None:
                # 将采样点的内点索引映射回原始点集
                ransac_inlier_points = ransac_points[best_inliers]
                
                # 在原始点集中找到对应的内点 
                # 重新在所有原始点上应用最佳平面进行筛选
                all_distances = np.abs(np.dot(points - best_point, best_normal))
                all_inliers_mask = all_distances <= distance_threshold
                inlier_points = points[all_inliers_mask]
                
                processing_log.append(f"重新筛选原始点集: {np.sum(all_inliers_mask):,} 个内点")
            else:
                inlier_points = ransac_points[best_inliers]
            centroid = np.mean(inlier_points, axis=0)
            
            # 对于大规模点云，限制SVD使用的点数以避免内存问题
            max_svd_points = 10000  # 限制SVD使用的最大点数
            if len(inlier_points) > max_svd_points:
                # 随机采样代表性点
                sample_indices = np.random.choice(len(inlier_points), max_svd_points, replace=False)
                sampled_points = inlier_points[sample_indices]
                processing_log.append(f"SVD拟合采样: 从 {len(inlier_points):,} 个内点中采样 {max_svd_points:,} 个点")
            else:
                sampled_points = inlier_points
            
            try:
                # 使用SVD进行更精确的平面拟合
                centered_points = sampled_points - centroid
                U, S, Vt = np.linalg.svd(centered_points)
                refined_normal = Vt[-1]  # 最小奇异值对应的向量
                
                best_normal = refined_normal
                best_point = centroid
                processing_log.append(f"SVD精细拟合完成，使用 {len(sampled_points):,} 个点")
            except np.core._exceptions._ArrayMemoryError as e:
                processing_log.append(f"SVD内存不足，使用原始RANSAC结果: {str(e)}")
                # 保持原始RANSAC结果
                pass
            except Exception as e:
                processing_log.append(f"SVD拟合失败，使用原始RANSAC结果: {str(e)}")
                # 保持原始RANSAC结果
                pass
        
        # 计算最终的内点统计
        final_inlier_count = len(inlier_points) if len(best_inliers) >= 3 else best_score
        
        ransac_info = {
            "iterations": max_iterations,
            "best_inliers_count": final_inlier_count,
            "inlier_ratio": final_inlier_count / original_point_count,
            "distance_threshold": distance_threshold,
            "used_sampling": sample_indices is not None,
            "ransac_points_count": len(ransac_points),
            "original_points_count": original_point_count
        }
        
        processing_log.append(f"RANSAC拟合完成: {final_inlier_count:,} 个内点 (比例: {ransac_info['inlier_ratio']:.1%})")
        
        return best_normal, best_point, best_inliers, ransac_info
    
    def _calculate_rotation_transform(self, ground_normal, ground_point, up_axis, processing_log):
        """计算将地面法向量对齐到指定向上轴的旋转变换"""
        
        # 根据up_axis设置目标法向量
        if up_axis == "Z":
            target_normal = np.array([0.0, 0.0, 1.0])  # Z轴向上
            height_axis = 2
        elif up_axis == "Y":
            target_normal = np.array([0.0, 1.0, 0.0])  # Y轴向上
            height_axis = 1
        elif up_axis == "-Y":
            target_normal = np.array([0.0, -1.0, 0.0])  # Y轴向下
            height_axis = 1
        else:
            target_normal = np.array([0.0, 0.0, 1.0])  # 默认Z轴向上
            height_axis = 2
        
        # 标准化输入法向量
        ground_normal = ground_normal / np.linalg.norm(ground_normal)
        
        processing_log.append(f"原始地面法向量: [{ground_normal[0]:.4f}, {ground_normal[1]:.4f}, {ground_normal[2]:.4f}]")
        processing_log.append(f"目标法向量: [{target_normal[0]:.4f}, {target_normal[1]:.4f}, {target_normal[2]:.4f}]")
        
        # 计算旋转轴和角度
        dot_product = np.dot(ground_normal, target_normal)
        
        # 处理特殊情况
        if np.abs(dot_product - 1.0) < 1e-6:
            # 已经对齐，无需旋转
            rotation_matrix = np.eye(3)
            rotation_angles = np.array([0.0, 0.0, 0.0])
            processing_log.append("地面已经水平，无需旋转")
            
        elif np.abs(dot_product + 1.0) < 1e-6:
            # 完全相反，绕X轴旋转180度
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            rotation_angles = np.array([np.pi, 0.0, 0.0])
            processing_log.append("地面完全颠倒，应用180度旋转")
            
        else:
            # 一般情况：计算旋转使地面水平
            
            # 调试：输出地面法向量的详细信息
            processing_log.append(f"地面法向量详情: X={ground_normal[0]:.4f}, Y={ground_normal[1]:.4f}, Z={ground_normal[2]:.4f}")
            processing_log.append(f"法向量长度: {np.linalg.norm(ground_normal):.4f}")
            
            # 方法2：分解为两个简单旋转
            # 首先绕Y轴旋转消除X分量，然后绕X轴旋转消除Y分量
            
            # 重新计算：我们需要找到一个旋转，使得ground_normal变成[0,0,1]
            # 使用最小旋转原理
            
            # 使用标准的旋转公式
            # 计算旋转轴（ground_normal × target_normal）
            rotation_axis = np.cross(ground_normal, target_normal)
            rotation_axis_length = np.linalg.norm(rotation_axis)
            
            if rotation_axis_length < 1e-8:
                # 向量平行或反平行
                if dot_product > 0:
                    # 已经对齐
                    rotation_matrix = np.eye(3)
                else:
                    # 180度旋转
                    # 找一个垂直于ground_normal的轴
                    if abs(ground_normal[0]) < 0.9:
                        perp = np.array([1, 0, 0])
                    else:
                        perp = np.array([0, 1, 0])
                    rotation_axis = np.cross(ground_normal, perp)
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    
                    # 180度旋转
                    rotation_matrix = 2 * np.outer(rotation_axis, rotation_axis) - np.eye(3)
            else:
                # 一般情况：使用罗德里格公式
                rotation_axis = rotation_axis / rotation_axis_length
                rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                
                # 罗德里格公式
                K = np.array([
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]
                ])
                
                rotation_matrix = (np.eye(3) + 
                                 np.sin(rotation_angle) * K + 
                                 (1 - np.cos(rotation_angle)) * np.dot(K, K))
                
                processing_log.append(f"使用罗德里格公式: 旋转轴=[{rotation_axis[0]:.4f}, {rotation_axis[1]:.4f}, {rotation_axis[2]:.4f}], 角度={np.degrees(rotation_angle):.2f}度")
            
            # 从旋转矩阵提取欧拉角
            rotation_angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
            
            processing_log.append(f"旋转角度 (弧度): [{rotation_angles[0]:.4f}, {rotation_angles[1]:.4f}, {rotation_angles[2]:.4f}]")
        
        # 验证旋转结果
        rotated_normal = np.dot(rotation_matrix, ground_normal)
        processing_log.append(f"理论旋转后法向量: [{rotated_normal[0]:.4f}, {rotated_normal[1]:.4f}, {rotated_normal[2]:.4f}]")
        
        # 检查旋转是否正确
        angle_after = np.arccos(np.clip(rotated_normal[2], -1.0, 1.0))
        processing_log.append(f"旋转后与垂直轴夹角: {np.degrees(angle_after):.2f} 度")
        
        if np.degrees(angle_after) > 5.0:
            processing_log.append("⚠ 警告：旋转结果与预期不符，地面未能完全水平")
        
        # 构建4x4变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        
        # 计算平移：旋转后将地面点移到高度轴=0的平面
        rotated_ground_point = np.dot(rotation_matrix, ground_point)
        transform_matrix[height_axis, 3] = -rotated_ground_point[height_axis]  # 将地面移动到高度轴=0
        
        processing_log.append(f"旋转前地面点: [{ground_point[0]:.4f}, {ground_point[1]:.4f}, {ground_point[2]:.4f}]")
        processing_log.append(f"旋转后地面点: [{rotated_ground_point[0]:.4f}, {rotated_ground_point[1]:.4f}, {rotated_ground_point[2]:.4f}]")
        processing_log.append(f"{['X', 'Y', 'Z'][height_axis]}轴平移: {-rotated_ground_point[height_axis]:.4f} (地面移至{['X', 'Y', 'Z'][height_axis]}=0)")
        
        return rotation_matrix, rotation_angles, transform_matrix
    
    def _rotation_matrix_to_euler_angles(self, R):
        """将旋转矩阵转换为欧拉角 (XYZ顺序)"""
        
        # 提取欧拉角
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def _apply_rotation_transform(self, scene, geometries_info, transform_matrix, processing_log):
        """应用旋转变换到场景中的所有几何体"""
        
        corrected_scene = trimesh.Scene()
        
        for name, geometry, original_transform in geometries_info:
            # 复制几何体
            new_geometry = geometry.copy()
            
            # 先重置任何现有的变换，然后应用校正变换
            if hasattr(new_geometry, 'vertices') and new_geometry.vertices is not None:
                # 获取原始顶点
                original_vertices = geometry.vertices.copy()
                
                # 如果几何体之前有变换，先应用原始变换
                if original_transform is not None and not np.allclose(original_transform, np.eye(4)):
                    # 将3x4变换扩展为4x4
                    if original_transform.shape == (3, 4):
                        full_original_transform = np.eye(4)
                        full_original_transform[:3, :] = original_transform
                    else:
                        full_original_transform = original_transform
                    
                    # 应用原始变换
                    world_vertices = trimesh.transformations.transform_points(original_vertices, full_original_transform)
                    processing_log.append(f"几何体 {name}: 应用原始变换 -> 世界坐标")
                else:
                    world_vertices = original_vertices
                    processing_log.append(f"几何体 {name}: 无原始变换，直接使用顶点")
                
                # 应用校正变换到世界坐标
                corrected_vertices = trimesh.transformations.transform_points(world_vertices, transform_matrix)
                
                # 更新几何体顶点
                new_geometry.vertices = corrected_vertices
                
                # 验证变换结果
                before_bounds = np.array([np.min(world_vertices, axis=0), np.max(world_vertices, axis=0)])
                after_bounds = np.array([np.min(corrected_vertices, axis=0), np.max(corrected_vertices, axis=0)])
                
                processing_log.append(f"几何体 {name} 变换前包围盒: Z={before_bounds[0][2]:.3f}~{before_bounds[1][2]:.3f}")
                processing_log.append(f"几何体 {name} 变换后包围盒: Z={after_bounds[0][2]:.3f}~{after_bounds[1][2]:.3f}")
            else:
                # 如果没有顶点，直接应用变换
                new_geometry.apply_transform(transform_matrix)
                processing_log.append(f"几何体 {name}: 应用变换矩阵")
            
            # 添加到新场景（不带额外变换）
            corrected_scene.add_geometry(new_geometry, node_name=name)
            
            processing_log.append(f"已校正几何体: {name}")
        
        processing_log.append(f"所有几何体已应用旋转校正")
        
        return corrected_scene
    
    def _verify_correction_result(self, corrected_scene, processing_log):
        """验证校正结果是否正确"""
        
        # 收集校正后的所有顶点
        all_corrected_vertices = []
        
        if isinstance(corrected_scene, trimesh.Scene):
            for name, geometry in corrected_scene.geometry.items():
                if hasattr(geometry, 'vertices') and geometry.vertices is not None:
                    all_corrected_vertices.append(geometry.vertices)
        else:
            if hasattr(corrected_scene, 'vertices') and corrected_scene.vertices is not None:
                all_corrected_vertices.append(corrected_scene.vertices)
        
        if all_corrected_vertices:
            combined_corrected = np.vstack(all_corrected_vertices)
            
            # 分析校正后的点云分布
            min_coords = np.min(combined_corrected, axis=0)
            max_coords = np.max(combined_corrected, axis=0)
            ranges = max_coords - min_coords
            
            processing_log.append(f"校正后包围盒: X=[{min_coords[0]:.3f}, {max_coords[0]:.3f}], Y=[{min_coords[1]:.3f}, {max_coords[1]:.3f}], Z=[{min_coords[2]:.3f}, {max_coords[2]:.3f}]")
            processing_log.append(f"校正后各轴范围: X={ranges[0]:.3f}, Y={ranges[1]:.3f}, Z={ranges[2]:.3f}")
            
            # 检查地面是否水平（使用最低点检测）
            z_values = combined_corrected[:, 2]
            z_min = np.min(z_values)
            height_tolerance = ranges[2] * 0.05
            ground_mask = z_values <= (z_min + height_tolerance)
            ground_points = combined_corrected[ground_mask]
            
            if len(ground_points) >= 10:
                # 对校正后的地面点拟合平面
                try:
                    centroid = np.mean(ground_points, axis=0)
                    centered_points = ground_points - centroid
                    
                    # 使用少量点进行SVD避免内存问题
                    if len(centered_points) > 1000:
                        sample_indices = np.random.choice(len(centered_points), 1000, replace=False)
                        centered_points = centered_points[sample_indices]
                    
                    U, S, Vt = np.linalg.svd(centered_points)
                    corrected_normal = Vt[-1]
                    
                    # 确保法向量向上
                    if corrected_normal[2] < 0:
                        corrected_normal = -corrected_normal
                    
                    # 计算与垂直方向的夹角
                    vertical_angle = np.arccos(np.clip(corrected_normal[2], -1.0, 1.0))
                    
                    processing_log.append(f"校正后地面法向量: [{corrected_normal[0]:.4f}, {corrected_normal[1]:.4f}, {corrected_normal[2]:.4f}]")
                    processing_log.append(f"校正后地面与水平面夹角: {np.degrees(vertical_angle):.2f} 度")
                    
                    if np.degrees(vertical_angle) < 5.0:
                        processing_log.append("✓ 校正成功：地面已接近水平")
                    else:
                        processing_log.append(f"⚠ 校正可能不完全：地面仍有 {np.degrees(vertical_angle):.1f} 度倾斜")
                    
                except Exception as e:
                    processing_log.append(f"校正验证失败: {str(e)}")
            else:
                processing_log.append("校正验证：地面点不足，无法验证")
        else:
            processing_log.append("校正验证：没有找到顶点数据")
    
    def _add_reference_planes(self, scene, original_vertices, ground_normal, ground_point, 
                            rotation_matrix, plane_size, processing_log):
        """添加参考平面可视化"""
        
        # 计算平面尺寸
        bbox_min = np.min(original_vertices, axis=0)
        bbox_max = np.max(original_vertices, axis=0)
        bbox_size = bbox_max - bbox_min
        plane_extent = np.max(bbox_size[:2]) * plane_size / 2  # 只考虑XY尺寸
        
        # 创建原始倾斜平面的网格点 (红色) - 在原始空间中
        plane_points = []
        plane_colors = []
        
        # 在地面平面上生成网格点
        u_axis = np.array([1, 0, 0]) if abs(ground_normal[0]) < 0.9 else np.array([0, 1, 0])
        u_axis = u_axis - np.dot(u_axis, ground_normal) * ground_normal
        u_axis = u_axis / np.linalg.norm(u_axis)
        v_axis = np.cross(ground_normal, u_axis)
        
        grid_density = 20  # 每个方向的点数
        
        for i in range(grid_density):
            for j in range(grid_density):
                u = (i - grid_density/2) * plane_extent / (grid_density/2)
                v = (j - grid_density/2) * plane_extent / (grid_density/2)
                
                # 在原始地面上的点
                original_point = ground_point + u * u_axis + v * v_axis
                
                # 应用旋转变换到这个点（因为场景已经被旋转了）
                rotated_point = np.dot(rotation_matrix, original_point)
                
                plane_points.append(rotated_point)
                plane_colors.append([255, 0, 0, 128])  # 半透明红色
        
        # 添加原始倾斜平面（旋转后的位置）
        if plane_points:
            original_plane_pc = trimesh.PointCloud(vertices=np.array(plane_points), colors=np.array(plane_colors))
            scene.add_geometry(original_plane_pc, node_name="original_ground_plane")
            processing_log.append(f"添加原始倾斜平面可视化（旋转后）: {len(plane_points)} 个点")
        
        # 创建校正后的水平平面 (绿色)
        horizontal_points = []
        horizontal_colors = []
        
        # 获取旋转后地面的中心位置作为水平平面的中心
        rotated_ground_center = np.dot(rotation_matrix, ground_point)
        z_level = rotated_ground_center[2]  # 使用旋转后地面的Z高度
        
        processing_log.append(f"水平参考平面Z高度: {z_level:.4f}")
        
        # 创建真正的水平平面（与XY平面平行）
        grid_density = 20
        for i in range(grid_density):
            for j in range(grid_density):
                x = rotated_ground_center[0] + (i - grid_density/2) * plane_extent / (grid_density/2)
                y = rotated_ground_center[1] + (j - grid_density/2) * plane_extent / (grid_density/2)
                z = z_level  # 保持在同一Z高度
                
                horizontal_point = np.array([x, y, z])
                horizontal_points.append(horizontal_point)
                horizontal_colors.append([0, 255, 0, 128])  # 半透明绿色
        
        # 验证水平平面的法向量
        if len(horizontal_points) >= 3:
            # 取三个点计算法向量
            p1 = horizontal_points[0]
            p2 = horizontal_points[1]
            p3 = horizontal_points[grid_density]
            v1 = p2 - p1
            v2 = p3 - p1
            test_normal = np.cross(v1, v2)
            test_normal = test_normal / np.linalg.norm(test_normal)
            processing_log.append(f"绿色平面法向量验证: [{test_normal[0]:.4f}, {test_normal[1]:.4f}, {test_normal[2]:.4f}]")
        
        # 添加水平参考平面
        if horizontal_points:
            horizontal_plane_pc = trimesh.PointCloud(vertices=np.array(horizontal_points), colors=np.array(horizontal_colors))
            scene.add_geometry(horizontal_plane_pc, node_name="horizontal_reference_plane")
            processing_log.append(f"添加水平参考平面可视化: {len(horizontal_points)} 个点 (Z={z_level:.4f}水平平面)")
    
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
