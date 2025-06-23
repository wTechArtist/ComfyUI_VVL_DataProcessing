from .common import *


class GLBPointCloudTransformApplier:
    """GLB点云变换应用器 - 根据提供的 4×4 齐次矩阵对 GLB 点云进行变换，并输出新的 GLB 文件"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rotation_info": ("STRING", {
                    "default": "",
                    "tooltip": "从 GLBPointCloudRotationCorrector 节点输出的 rotation_info(JSON 字符串)"
                }),
                "glb_file_path": ("STRING", {
                    "default": "",
                    "tooltip": "需要应用矩阵变换的 GLB 点云文件路径"
                }),
            },
            "optional": {
                "output_filename": ("STRING", {
                    "default": "transformed_pointcloud",
                    "tooltip": "输出 GLB 文件名(无需扩展名)"
                })
            }
        }

    RETURN_TYPES = (
        "STRING",  # 变换后的 GLB 文件路径
    )
    RETURN_NAMES = (
        "transformed_glb_path",
    )
    OUTPUT_TOOLTIPS = [
        "应用矩阵后的 GLB 文件完整路径",
    ]
    OUTPUT_NODE = True
    FUNCTION = "apply_transform"
    CATEGORY = "💃VVL/Point Cloud Transform"

    # ------------------------------------------------------------------
    # 核心功能实现
    # ------------------------------------------------------------------

    def apply_transform(self,
                        rotation_info: str,
                        glb_file_path: str,
                        output_filename: str = "transformed_pointcloud"):
        """根据 rotation_info 中的 4×4 transform_matrix 对 GLB 文件进行变换"""

        processing_log = []
        processing_log.append("开始应用 4×4 变换矩阵到 GLB ...")

        # 检查 trimesh 是否可用
        if not TRIMESH_AVAILABLE:
            logger.error("trimesh 库不可用，无法处理 GLB 文件")
            return ("",)

        # ------------------------------------------------------------------
        # 解析 rotation_info，提取 4×4 矩阵
        # ------------------------------------------------------------------
        try:
            info = json.loads(rotation_info)
            transform_matrix = info.get("transform_matrix")
            if transform_matrix is None:
                raise ValueError("rotation_info 中不包含 transform_matrix")
            transform_matrix = np.asarray(transform_matrix, dtype=np.float64)
            if transform_matrix.shape != (4, 4):
                raise ValueError(f"transform_matrix 形状错误: {transform_matrix.shape}")
        except Exception as e:
            logger.error(f"解析 rotation_info 失败: {e}")
            return ("",)

        processing_log.append("成功解析 transform_matrix")

        # ------------------------------------------------------------------
        # 解析 GLB 路径
        # ------------------------------------------------------------------
        if not glb_file_path or not glb_file_path.strip():
            logger.error("GLB 文件路径为空")
            return ("",)

        input_path = self._resolve_file_path(glb_file_path.strip())
        if not os.path.exists(input_path):
            logger.error(f"GLB 文件不存在: {input_path}")
            return ("",)

        processing_log.append(f"输入 GLB: {input_path}")

        # ------------------------------------------------------------------
        # 加载 GLB
        # ------------------------------------------------------------------
        try:
            scene = trimesh.load(input_path)
        except Exception as e:
            logger.error(f"加载 GLB 失败: {e}")
            return ("",)

        # ------------------------------------------------------------------
        # 对场景中所有几何体应用变换
        # ------------------------------------------------------------------
        transformed_scene = trimesh.Scene()

        if isinstance(scene, trimesh.Scene):
            for node_name in scene.graph.nodes:
                if node_name not in scene.geometry:
                    continue

                geometry = scene.geometry[node_name]
                original_transform = scene.graph[node_name][0]

                new_geometry = geometry.copy()

                if hasattr(new_geometry, 'vertices') and new_geometry.vertices is not None:
                    vertices = geometry.vertices.copy()

                    # 应用原始节点变换
                    if original_transform is not None and not np.allclose(original_transform, np.eye(4)):
                        if original_transform.shape == (3, 4):
                            full_transform = np.eye(4)
                            full_transform[:3, :] = original_transform
                            original_transform = full_transform
                        vertices_world = trimesh.transformations.transform_points(vertices, original_transform)
                    else:
                        vertices_world = vertices

                    # 应用新的 4×4 变换
                    vertices_transformed = trimesh.transformations.transform_points(vertices_world, transform_matrix)
                    new_geometry.vertices = vertices_transformed
                else:
                    # 对于不含显式顶点的几何体，直接在矩阵级别应用变换
                    new_geometry.apply_transform(transform_matrix)

                transformed_scene.add_geometry(new_geometry, node_name=node_name)
        else:
            # 单一 Mesh
            transformed_scene = scene.copy()
            transformed_scene.apply_transform(transform_matrix)

        # ------------------------------------------------------------------
        # 导出结果
        # ------------------------------------------------------------------
        output_path = self._generate_output_path(output_filename)
        try:
            transformed_scene.export(output_path)
        except Exception as e:
            logger.error(f"保存 GLB 失败: {e}")
            return ("",)

        if not os.path.exists(output_path):
            logger.error("输出文件保存失败")
            return ("",)

        file_size = os.path.getsize(output_path)
        processing_log.append(f"GLB 保存成功: {output_path} (大小: {file_size} bytes)")

        return (output_path,)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _resolve_file_path(self, file_path: str) -> str:
        """解析文件路径，支持绝对/相对路径以及 ComfyUI 输出目录"""
        if os.path.isabs(file_path):
            return file_path

        # 尝试 ComfyUI 输出目录
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

        return os.path.join(output_dir, filename) 