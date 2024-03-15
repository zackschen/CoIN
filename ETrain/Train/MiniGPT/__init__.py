import os
import sys

from omegaconf import OmegaConf

from ETrain.utils.MiniGPT.common.registry import registry

from ETrain.Dataset.MiniGPT.builders import *
from ETrain.Models.MiniGPT import *
from ETrain.utils.MiniGPT.processors import *


root_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = '/'.join(root_dir.split('/')[:-2])
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/MiniGPT/default.yaml"))

registry.register_path("library_root", root_dir)
repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])