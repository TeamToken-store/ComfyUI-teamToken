"""ComfyUI-teamToken — Veo, Seedance, Kling and nano-banana through one API key.

ComfyUI imports this package from ``custom_nodes/`` and reads the three module
globals below. The node classes live in ``teamtoken/`` so the HTTP/wire layer
stays importable without a ComfyUI runtime (see teamtoken/client.py).
"""

from .teamtoken.nodes_image import TeamTokenImage, TeamTokenImageEdit
from .teamtoken.nodes_video import TeamTokenVideo, TeamTokenVideoExtend

NODE_CLASS_MAPPINGS = {
    "TeamTokenImage": TeamTokenImage,
    "TeamTokenImageEdit": TeamTokenImageEdit,
    "TeamTokenVideo": TeamTokenVideo,
    "TeamTokenVideoExtend": TeamTokenVideoExtend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TeamTokenImage": "teamToken Image",
    "TeamTokenImageEdit": "teamToken Image Edit",
    "TeamTokenVideo": "teamToken Video",
    "TeamTokenVideoExtend": "teamToken Video Extend",
}

# The JS extension that adds the API-key / server-URL settings pane.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
