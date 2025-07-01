import json
import logging
from typing import Any, Dict, Union

# 配置日志
logger = logging.getLogger('json_merger')

class JSONMerger:
    """JSON合并器 - 用于合并GLBPointCloudBounds和GLBPointCloudOriginAdjuster的输出数据"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data_1": ("STRING", {
                    "tooltip": "第一个JSON数据：来自GLBPointCloudBounds或GLBPointCloudOriginAdjuster的输出。支持包含name、scale、position等字段的JSON格式数据"
                }),
                "json_data_2": ("STRING", {
                    "tooltip": "第二个JSON数据：来自GLBPointCloudBounds或GLBPointCloudOriginAdjuster的输出。支持包含name、scale、position等字段的JSON格式数据"
                }),
            },
            "optional": {
                "json_data_3": ("STRING", {
                    "tooltip": "第三个JSON数据（可选）：额外的JSON数据，用于三路合并。留空则只合并前两个数据"
                }),
                "merge_mode": (["smart_merge", "array_concat"], {
                    "default": "smart_merge",
                    "tooltip": "合并模式：smart_merge=智能合并（相同name合并字段，不同name用逗号连接）；array_concat=数组连接（所有数据放入数组）"
                }),
            }
        }

    RETURN_TYPES = (
        "STRING",    # 合并后的JSON字符串
    )
    RETURN_NAMES = (
        "merged_json",
    )
    OUTPUT_TOOLTIPS = [
        "合并后的JSON数据：根据name字段智能合并或连接多个JSON对象的结果",
    ]
    OUTPUT_NODE = True
    FUNCTION = "merge_json_data"
    CATEGORY = "💃VVL/JSON Processing"

    def merge_json_data(self,
                       json_data_1: str,
                       json_data_2: str,
                       json_data_3: str = "",
                       merge_mode: str = "smart_merge"):
        """
        合并多个JSON数据
        """
        
        processing_log = []
        processing_log.append("开始JSON数据合并...")
        
        try:
            # 解析输入的JSON数据
            json_objects = []
            
            # 解析第一个JSON
            if json_data_1.strip():
                try:
                    obj1 = json.loads(json_data_1.strip())
                    json_objects.append(obj1)
                    processing_log.append(f"解析JSON数据1成功: {self._get_object_summary(obj1)}")
                except json.JSONDecodeError as e:
                    error_msg = f"JSON数据1解析失败: {str(e)}"
                    processing_log.append(f"错误: {error_msg}")
                    return (json.dumps({"error": error_msg}, indent=2),)
            
            # 解析第二个JSON
            if json_data_2.strip():
                try:
                    obj2 = json.loads(json_data_2.strip())
                    json_objects.append(obj2)
                    processing_log.append(f"解析JSON数据2成功: {self._get_object_summary(obj2)}")
                except json.JSONDecodeError as e:
                    error_msg = f"JSON数据2解析失败: {str(e)}"
                    processing_log.append(f"错误: {error_msg}")
                    return (json.dumps({"error": error_msg}, indent=2),)
            
            # 解析第三个JSON（可选）
            if json_data_3.strip():
                try:
                    obj3 = json.loads(json_data_3.strip())
                    json_objects.append(obj3)
                    processing_log.append(f"解析JSON数据3成功: {self._get_object_summary(obj3)}")
                except json.JSONDecodeError as e:
                    error_msg = f"JSON数据3解析失败: {str(e)}"
                    processing_log.append(f"错误: {error_msg}")
                    return (json.dumps({"error": error_msg}, indent=2),)
            
            if not json_objects:
                error_msg = "没有有效的JSON数据输入"
                processing_log.append(f"错误: {error_msg}")
                return (json.dumps({"error": error_msg}, indent=2),)
            
            if len(json_objects) == 1:
                processing_log.append("只有一个有效JSON对象，直接返回")
                result = json_objects[0]
            else:
                # 根据合并模式处理
                if merge_mode == "smart_merge":
                    result = self._smart_merge(json_objects, processing_log)
                else:  # array_concat
                    result = self._array_concat(json_objects, processing_log)
            
            # 生成最终的JSON字符串
            merged_json = json.dumps(result, indent=2, ensure_ascii=False)
            
            processing_log.append("JSON数据合并完成!")
            processing_log.append(f"最终结果: {self._get_object_summary(result)}")
            
            return (merged_json,)
                
        except Exception as e:
            error_msg = f"合并JSON数据时发生错误: {str(e)}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return (json.dumps({"error": error_msg}, indent=2),)
    
    def _smart_merge(self, json_objects: list, processing_log: list) -> Union[Dict, list]:
        """
        智能合并：相同name的对象合并字段，不同name的对象用数组形式返回
        """
        processing_log.append("使用智能合并模式...")
        
        # 按name字段分组
        name_groups = {}
        unnamed_objects = []
        
        for i, obj in enumerate(json_objects):
            if isinstance(obj, dict):
                name = obj.get("name")
                if name:
                    if name not in name_groups:
                        name_groups[name] = []
                    name_groups[name].append(obj)
                    processing_log.append(f"对象{i+1}归入组 '{name}'")
                else:
                    unnamed_objects.append(obj)
                    processing_log.append(f"对象{i+1}没有name字段，归入未命名组")
            else:
                unnamed_objects.append(obj)
                processing_log.append(f"对象{i+1}不是字典类型，归入未命名组")
        
        # 合并结果
        merged_results = []
        
        # 处理有name的组
        for name, group_objects in name_groups.items():
            if len(group_objects) == 1:
                # 只有一个对象，直接使用
                merged_results.append(group_objects[0])
                processing_log.append(f"组 '{name}' 只有一个对象，直接使用")
            else:
                # 多个对象，合并字段
                merged_obj = {"name": name}
                for obj in group_objects:
                    for key, value in obj.items():
                        if key != "name":  # 跳过name字段
                            if key not in merged_obj:
                                merged_obj[key] = value
                            else:
                                # 如果字段已存在，优先使用非空值
                                if not merged_obj[key] and value:
                                    merged_obj[key] = value
                
                merged_results.append(merged_obj)
                processing_log.append(f"组 '{name}' 合并了 {len(group_objects)} 个对象")
        
        # 添加未命名对象
        merged_results.extend(unnamed_objects)
        if unnamed_objects:
            processing_log.append(f"添加了 {len(unnamed_objects)} 个未命名对象")
        
        # 返回结果
        if len(merged_results) == 1:
            processing_log.append("合并结果为单个对象")
            return merged_results[0]
        else:
            processing_log.append(f"合并结果为 {len(merged_results)} 个对象的数组")
            return merged_results
    
    def _array_concat(self, json_objects: list, processing_log: list) -> list:
        """
        数组连接：将所有对象放入一个数组
        """
        processing_log.append("使用数组连接模式...")
        processing_log.append(f"将 {len(json_objects)} 个对象连接为数组")
        return json_objects
    
    def _get_object_summary(self, obj: Any) -> str:
        """
        获取对象的简要描述
        """
        if isinstance(obj, dict):
            keys = list(obj.keys())
            name = obj.get("name", "未命名")
            return f"字典对象 name='{name}', 字段=[{', '.join(keys)}]"
        elif isinstance(obj, list):
            return f"数组对象，包含 {len(obj)} 个元素"
        else:
            return f"其他类型: {type(obj).__name__}"

class UESceneGenerator:
    """UE场景生成器 - 将JSONMerger的输出转换为UE场景配置格式"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merged_json": ("*", {
                    "tooltip": "合并后的JSON数据：来自JSONMerger的输出，包含name、position、scale等字段的JSON数据，支持STRING/LIST/DICT类型"
                }),
            },
            "optional": {
                "camera_position": ("STRING", {
                    "default": "[-1000, 0, 100]",
                    "tooltip": "相机位置：UE场景中相机的世界坐标位置 [X, Y, Z]，默认为[-1000, 0, 100]"
                }),
                "camera_rotation": ("STRING", {
                    "default": "[-12, 0, 0]",
                    "tooltip": "相机旋转：UE场景中相机的旋转角度 [Pitch, Yaw, Roll]，默认为[-12, 0, 0]"
                }),
                "camera_fov": ("FLOAT", {
                    "default": 58.53, "min": 1.0, "max": 179.0, "step": 0.1,
                    "tooltip": "相机视野角度：UE场景中相机的视野角度(FOV)，默认为58.53度。注意：这是一个数值参数，不要连接数组输入！"
                }),
                "default_object_rotation": ("STRING", {
                    "default": "[0, 0, 0]",
                    "tooltip": "默认物体旋转：当输入数据中没有rotation字段时使用的默认旋转值 [Pitch, Yaw, Roll]"
                }),
                "scale_axis": (["XYZ", "X", "Y", "Z", "XY", "XZ", "YZ"], {
                    "default": "XYZ",
                    "tooltip": "缩放轴向：选择对物体scale进行缩放的轴向。XYZ=全轴，X/Y/Z=单轴，XY/XZ/YZ=双轴"
                }),
                "scale_multiplier": ("FLOAT", {
                    "default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.1,
                    "tooltip": "缩放倍数：对选定轴向的scale值进行缩放的倍数，用于调整物体大小到合适的UE单位。注意：这是一个数值参数，不要连接数组输入！"
                }),
                "position_axis": (["XYZ", "X", "Y", "Z", "XY", "XZ", "YZ"], {
                    "default": "Z",
                    "tooltip": "位置轴向：选择对物体position进行缩放的轴向。XYZ=全轴，X/Y/Z=单轴，XY/XZ/YZ=双轴"
                }),
                "position_multiplier": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1,
                    "tooltip": "位置倍数：对选定轴向的position值进行缩放的倍数，用于调整物体位置到合适的UE单位。注意：这是一个数值参数，不要连接数组输入！"
                }),
                "recenter_scene": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否自动将所有物体整体平移，使场景中心(质心)位于原点(0,0,0)。开启后，不影响物体间相对位置和大小。"
                }),
                "flip_y_axis": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否翻转Y轴坐标（Y值取反）以适配UE坐标体系。开启后，所有position的Y值会取反，其他轴不变。"
                }),
            }
        }

    RETURN_TYPES = (
        "STRING",    # UE场景配置JSON
    )
    RETURN_NAMES = (
        "ue_scene_json",
    )
    OUTPUT_TOOLTIPS = [
        "UE场景配置JSON：符合UE导入格式的完整场景配置，包含相机设置和物体列表",
    ]
    OUTPUT_NODE = True
    FUNCTION = "generate_ue_scene"
    CATEGORY = "💃VVL/UE Integration"

    def generate_ue_scene(self,
                         merged_json,
                         camera_position: str = "[-1000, 0, 100]",
                         camera_rotation: str = "[-12, 0, 0]",
                         camera_fov = 58.53,
                         default_object_rotation: str = "[0, 0, 0]",
                         scale_axis: str = "XYZ",
                         scale_multiplier = 1.0,
                         position_axis: str = "XYZ",
                         position_multiplier = 0.0,
                         recenter_scene: bool = False,
                         flip_y_axis: bool = True):
        """
        将合并的JSON数据转换为UE场景配置格式
        """
        
        processing_log = []
        processing_log.append("开始生成UE场景配置...")
        
        # 参数验证和修复
        try:
            # 修复可能错误连接的参数
            if isinstance(camera_fov, str):
                processing_log.append(f"警告: camera_fov参数类型错误(收到字符串'{camera_fov}')，使用默认值58.53")
                camera_fov = 58.53
            elif not isinstance(camera_fov, (int, float)):
                processing_log.append(f"警告: camera_fov参数类型错误(收到{type(camera_fov).__name__})，使用默认值58.53")
                camera_fov = 58.53
            
            if isinstance(scale_multiplier, str):
                processing_log.append(f"警告: scale_multiplier参数类型错误(收到字符串'{scale_multiplier}')，使用默认值1.0")
                scale_multiplier = 1.0
            elif not isinstance(scale_multiplier, (int, float)):
                processing_log.append(f"警告: scale_multiplier参数类型错误(收到{type(scale_multiplier).__name__})，使用默认值1.0")
                scale_multiplier = 1.0
                
            if isinstance(position_multiplier, str):
                processing_log.append(f"警告: position_multiplier参数类型错误(收到字符串'{position_multiplier}')，使用默认值0.0")
                position_multiplier = 0.0
            elif not isinstance(position_multiplier, (int, float)):
                processing_log.append(f"警告: position_multiplier参数类型错误(收到{type(position_multiplier).__name__})，使用默认值0.0")
                position_multiplier = 0.0
            
            if not isinstance(recenter_scene, bool):
                processing_log.append(f"警告: recenter_scene参数类型错误(收到{type(recenter_scene).__name__})，使用默认值False")
                recenter_scene = False
            
            if not isinstance(flip_y_axis, bool):
                processing_log.append(f"警告: flip_y_axis参数类型错误(收到{type(flip_y_axis).__name__})，使用默认值True")
                flip_y_axis = True
            
            # 确保参数在合理范围内
            camera_fov = max(1.0, min(179.0, float(camera_fov)))
            scale_multiplier = max(0.001, min(1000.0, float(scale_multiplier)))
            position_multiplier = max(0.0, min(1000.0, float(position_multiplier)))
            
            # 验证轴向参数
            valid_axes = ["XYZ", "X", "Y", "Z", "XY", "XZ", "YZ"]
            if scale_axis not in valid_axes:
                processing_log.append(f"警告: scale_axis参数无效('{scale_axis}')，使用默认值'XYZ'")
                scale_axis = "XYZ"
            if position_axis not in valid_axes:
                processing_log.append(f"警告: position_axis参数无效('{position_axis}')，使用默认值'XYZ'")
                position_axis = "XYZ"
            
            processing_log.append(f"参数验证完成: camera_fov={camera_fov}")
            processing_log.append(f"缩放设置: {scale_axis}轴, 倍数={scale_multiplier}")
            processing_log.append(f"位置设置: {position_axis}轴, 倍数={position_multiplier}")
            processing_log.append(f"坐标系设置: Y轴翻转={'开启' if flip_y_axis else '关闭'}, 场景居中={'开启' if recenter_scene else '关闭'}")
            
        except Exception as e:
            processing_log.append(f"参数验证失败: {str(e)}，使用默认值")
            camera_fov = 58.53
            scale_multiplier = 1.0
            position_multiplier = 0.0
            scale_axis = "XYZ"
            position_axis = "XYZ"
            recenter_scene = False
        
        try:
            # 处理输入的合并JSON数据（可能是字符串或直接的数据）
            if isinstance(merged_json, str):
                # 字符串输入，需要解析
                if not merged_json.strip():
                    error_msg = "合并JSON数据为空"
                    processing_log.append(f"错误: {error_msg}")
                    return (json.dumps({"error": error_msg}, indent=2),)
                
                try:
                    merged_data = json.loads(merged_json.strip())
                    processing_log.append("字符串JSON数据解析成功")
                except json.JSONDecodeError as e:
                    error_msg = f"JSON数据解析失败: {str(e)}"
                    processing_log.append(f"错误: {error_msg}")
                    return (json.dumps({"error": error_msg}, indent=2),)
            elif isinstance(merged_json, (list, dict)):
                # 直接的数据结构输入
                merged_data = merged_json
                processing_log.append(f"直接数据输入: {type(merged_json).__name__}")
                
                # 详细分析输入结构
                if isinstance(merged_json, list):
                    processing_log.append(f"列表包含 {len(merged_json)} 个元素:")
                    for i, item in enumerate(merged_json):
                        if isinstance(item, dict):
                            keys = list(item.keys())
                            processing_log.append(f"  元素{i}: dict with keys {keys}")
                        elif isinstance(item, str):
                            preview = item[:100] + "..." if len(item) > 100 else item
                            processing_log.append(f"  元素{i}: string '{preview}'")
                        else:
                            processing_log.append(f"  元素{i}: {type(item).__name__}")
                            
            else:
                error_msg = f"不支持的输入数据类型: {type(merged_json).__name__}"
                processing_log.append(f"错误: {error_msg}")
                return (json.dumps({"error": error_msg}, indent=2),)
            
            # 解析相机参数
            try:
                cam_pos = json.loads(camera_position)
                cam_rot = json.loads(camera_rotation)
                default_rot = json.loads(default_object_rotation)
                
                if not isinstance(cam_pos, list) or len(cam_pos) != 3:
                    raise ValueError("相机位置必须是包含3个数值的数组")
                if not isinstance(cam_rot, list) or len(cam_rot) != 3:
                    raise ValueError("相机旋转必须是包含3个数值的数组")
                if not isinstance(default_rot, list) or len(default_rot) != 3:
                    raise ValueError("默认旋转必须是包含3个数值的数组")
                    
                processing_log.append(f"相机参数解析成功: 位置{cam_pos}, 旋转{cam_rot}, FOV{camera_fov}")
                
            except (json.JSONDecodeError, ValueError) as e:
                error_msg = f"相机参数解析失败: {str(e)}"
                processing_log.append(f"错误: {error_msg}")
                return (json.dumps({"error": error_msg}, indent=2),)
            
            # 创建UE场景配置基础结构
            ue_scene = {
                "camera": {
                    "position": cam_pos,
                    "rotation": cam_rot,
                    "fov": camera_fov
                },
                "objects": []
            }
            
            # 处理物体数据
            objects_processed = 0
            
            def process_data_item(data_item, item_name=""):
                """递归处理数据项，支持嵌套结构"""
                nonlocal objects_processed
                
                if isinstance(data_item, dict):
                    # 单个物体字典
                    ue_object = self._convert_to_ue_object(
                        data_item, default_rot, scale_axis, scale_multiplier, 
                        position_axis, position_multiplier, flip_y_axis, processing_log
                    )
                    if ue_object:
                        ue_scene["objects"].append(ue_object)
                        objects_processed += 1
                        
                elif isinstance(data_item, list):
                    # 数组，递归处理每个元素
                    for i, sub_item in enumerate(data_item):
                        sub_name = f"{item_name}[{i}]" if item_name else f"item_{i}"
                        process_data_item(sub_item, sub_name)
                        
                elif isinstance(data_item, str):
                    # 字符串，尝试解析为JSON
                    try:
                        parsed_item = json.loads(data_item)
                        processing_log.append(f"解析嵌套JSON字符串: {item_name}")
                        process_data_item(parsed_item, f"{item_name}_parsed")
                    except json.JSONDecodeError:
                        processing_log.append(f"跳过无法解析的字符串: {item_name} = '{data_item[:50]}...'")
                        
                else:
                    processing_log.append(f"跳过不支持的数据类型 {item_name}: {type(data_item).__name__}")
            
            # 开始处理数据
            if isinstance(merged_data, (dict, list)):
                processing_log.append(f"开始处理输入数据类型: {type(merged_data).__name__}")
                process_data_item(merged_data, "root")
            else:
                error_msg = f"不支持的合并数据类型: {type(merged_data).__name__}"
                processing_log.append(f"错误: {error_msg}")
                return (json.dumps({"error": error_msg}, indent=2),)
            
            processing_log.append(f"成功处理 {objects_processed} 个物体")
            
            # -------------------------------------------------
            # 场景整体居中: 将所有物体质心移动到原点
            # -------------------------------------------------
            if recenter_scene and objects_processed > 0:
                # 计算质心
                sum_x = sum(o["position"][0] for o in ue_scene["objects"])
                sum_y = sum(o["position"][1] for o in ue_scene["objects"])
                sum_z = sum(o["position"][2] for o in ue_scene["objects"])
                centroid = [sum_x / objects_processed, sum_y / objects_processed, sum_z / objects_processed]
                processing_log.append(f"场景质心: {centroid}")
                
                # 检查物体位置是否有明显差异（避免所有物体都移动到原点的问题）
                positions = [o["position"] for o in ue_scene["objects"]]
                min_x = min(pos[0] for pos in positions)
                max_x = max(pos[0] for pos in positions)
                min_y = min(pos[1] for pos in positions)
                max_y = max(pos[1] for pos in positions)
                min_z = min(pos[2] for pos in positions)
                max_z = max(pos[2] for pos in positions)
                
                # 计算各轴的范围
                range_x = max_x - min_x
                range_y = max_y - min_y
                range_z = max_z - min_z
                max_range = max(range_x, range_y, range_z)
                
                processing_log.append(f"物体位置范围: X=[{min_x:.3f}, {max_x:.3f}] ({range_x:.3f}), Y=[{min_y:.3f}, {max_y:.3f}] ({range_y:.3f}), Z=[{min_z:.3f}, {max_z:.3f}] ({range_z:.3f})")
                
                # 只有当物体位置有明显差异时才进行居中（阈值设为0.01）
                if max_range > 0.01:
                    # 将物体位置减去质心
                    for o in ue_scene["objects"]:
                        original_pos = o["position"].copy()
                        o["position"] = [original_pos[0] - centroid[0],
                                           original_pos[1] - centroid[1],
                                           original_pos[2] - centroid[2]]
                    processing_log.append("已将所有物体整体平移，使质心位于原点")
                else:
                    processing_log.append("物体位置差异很小，跳过居中操作以避免所有物体重叠在原点")
            
            # 生成最终的UE场景JSON
            ue_scene_json = json.dumps(ue_scene, indent=2, ensure_ascii=False)
            
            processing_log.append("UE场景配置生成完成!")
            processing_log.append(f"场景包含: 1个相机 + {len(ue_scene['objects'])} 个物体")
            
            return (ue_scene_json,)
                
        except Exception as e:
            error_msg = f"生成UE场景配置时发生错误: {str(e)}"
            logger.error(error_msg)
            processing_log.append(f"错误: {error_msg}")
            import traceback
            traceback.print_exc()
            return (json.dumps({"error": error_msg}, indent=2),)
    
    def _convert_to_ue_object(self, obj_data: dict, default_rotation: list, 
                             scale_axis: str, scale_multiplier: float,
                             position_axis: str, position_multiplier: float,
                             flip_y_axis: bool, processing_log: list) -> dict:
        """
        将单个对象数据转换为UE物体格式
        """
        try:
            # 获取物体名称
            name = obj_data.get("name", "未命名物体")
            
            # 获取位置信息
            position = obj_data.get("position", [0, 0, 0])
            if not isinstance(position, list) or len(position) != 3:
                processing_log.append(f"物体 '{name}' 位置格式错误，使用默认值 [0, 0, 0]")
                position = [0, 0, 0]
            
            # 获取旋转信息
            rotation = obj_data.get("rotation", default_rotation)
            if not isinstance(rotation, list) or len(rotation) != 3:
                processing_log.append(f"物体 '{name}' 旋转格式错误，使用默认值 {default_rotation}")
                rotation = default_rotation.copy()
            
            # 获取缩放信息
            scale = obj_data.get("scale", [1, 1, 1])
            if not isinstance(scale, list) or len(scale) != 3:
                processing_log.append(f"物体 '{name}' 缩放格式错误，使用默认值 [1, 1, 1]")
                scale = [1, 1, 1]
            
            # 应用位置倍数（按轴向）
            if position_multiplier != 1.0:
                original_pos = position.copy()
                position = self._apply_axis_multiplier(position, position_axis, position_multiplier)
                processing_log.append(f"物体 '{name}' 位置{position_axis}轴应用倍数{position_multiplier}: {original_pos} -> {position}")
            
            # 应用缩放倍数（按轴向）
            if scale_multiplier != 1.0:
                original_scale = scale.copy()
                scale = self._apply_axis_multiplier(scale, scale_axis, scale_multiplier)
                processing_log.append(f"物体 '{name}' 缩放{scale_axis}轴应用倍数{scale_multiplier}: {original_scale} -> {scale}")
            
            # 翻转Y轴以适配UE坐标体系
            if flip_y_axis:
                original_y = position[1]
                position[1] = -position[1]
                processing_log.append(f"物体 '{name}' Y轴翻转: {original_y} -> {position[1]}")
            
            # 创建UE物体
            ue_object = {
                "name": name,
                "position": [float(position[0]), float(position[1]), float(position[2])],
                "rotation": [float(rotation[0]), float(rotation[1]), float(rotation[2])],
                "scale": [float(scale[0]), float(scale[1]), float(scale[2])]
            }
            
            processing_log.append(f"转换物体 '{name}': 位置{position}, 旋转{rotation}, 缩放{scale}")
            
            return ue_object
            
        except Exception as e:
            processing_log.append(f"转换物体失败: {str(e)}")
            return None
    
    def _apply_axis_multiplier(self, values: list, axis: str, multiplier: float) -> list:
        """
        根据轴向选择对指定轴应用倍数
        
        Args:
            values: 三维数值列表 [X, Y, Z]
            axis: 轴向选择 ("XYZ", "X", "Y", "Z", "XY", "XZ", "YZ")
            multiplier: 倍数
            
        Returns:
            应用倍数后的数值列表
        """
        result = values.copy()
        
        if axis == "XYZ":
            # 全轴缩放
            result = [v * multiplier for v in result]
        elif axis == "X":
            # 只缩放X轴
            result[0] *= multiplier
        elif axis == "Y":
            # 只缩放Y轴
            result[1] *= multiplier
        elif axis == "Z":
            # 只缩放Z轴
            result[2] *= multiplier
        elif axis == "XY":
            # 缩放X和Y轴
            result[0] *= multiplier
            result[1] *= multiplier
        elif axis == "XZ":
            # 缩放X和Z轴
            result[0] *= multiplier
            result[2] *= multiplier
        elif axis == "YZ":
            # 缩放Y和Z轴
            result[1] *= multiplier
            result[2] *= multiplier
        
        return result


# -----------------------------------------------------------------------------
# 节点注册
# -----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "JSONMerger": JSONMerger,
    "UESceneGenerator": UESceneGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JSONMerger": "VVL JSON Merger",
    "UESceneGenerator": "VVL UE Scene Generator",
}

# 添加节点信息，帮助ComfyUI更好地识别
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"] 