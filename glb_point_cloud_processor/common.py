import os
import json
import tempfile
import logging
import struct
import time
from typing import List, Any, Dict, Tuple

import numpy as np
import torch

# 导入ComfyUI的路径管理
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    folder_paths = None
    FOLDER_PATHS_AVAILABLE = False

# 导入trimesh用于GLB文件处理
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None

# 配置日志
logger = logging.getLogger('glb_point_cloud_processor') 