from __future__ import annotations

import os
import folder_paths
from typing import Literal
from comfy.comfy_types import IO, FileLocator, ComfyNodeABC
from comfy_api.input import VideoInput
from comfy_api.util import VideoContainer, VideoCodec
from comfy.cli_args import args


class SaveVideo(ComfyNodeABC):
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type: Literal["output"] = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "The video to save."}),
                "filename_prefix": ("STRING", {"default": "video/ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "format": (VideoContainer.as_input(), {"default": "auto", "tooltip": "The format to save the video as."}),
                "codec": (VideoCodec.as_input(), {"default": "auto", "tooltip": "The codec to use for the video."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save_video"

    OUTPUT_NODE = True

    CATEGORY = "VVL/video"
    DESCRIPTION = "Saves the input video to your ComfyUI output directory and returns the file path."

    def save_video(self, video: VideoInput, filename_prefix, format, codec, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            width,
            height
        )
        results: list[FileLocator] = list()
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if extra_pnginfo is not None:
                metadata.update(extra_pnginfo)
            if prompt is not None:
                metadata["prompt"] = prompt
            if len(metadata) > 0:
                saved_metadata = metadata
        file = f"{filename}_{counter:05}_.{VideoContainer.get_extension(format)}"
        full_video_path = os.path.join(full_output_folder, file)
        
        video.save_to(
            full_video_path,
            format=format,
            codec=codec,
            metadata=saved_metadata
        )

        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })
        counter += 1

        return (full_video_path, { "ui": { "images": results, "animated": (True,) } })