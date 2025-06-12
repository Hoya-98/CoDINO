from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    model_name: str = "dinov2_vits14_reg"
    resize_dim: int = 840
    map_keys: List[str] = ["vit_out"]
    device: str = "cuda"

@dataclass
class ProcessingConfig:
    divide_et_impera: bool = True
    divide_et_impera_twice: bool = True
    filter_background: bool = True
    ellipse_normalization: bool = True
    ellipse_kernel_cleaning: bool = True
    threshold: float = 0.5

@dataclass
class DataConfig:
    img_dir: str = "data/screws/images"
    annotation_file: str = "data/screws/annotations/annotation.json"
    splits_file: str = "data/screws/annotations/test_split.json"

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    processing: ProcessingConfig = ProcessingConfig()
    data: DataConfig = DataConfig()
    
    def __post_init__(self):
        if self.model.resize_dim % 14 != 0:
            raise ValueError(f"resize_dim must be multiple of 14, got {self.model.resize_dim}") 