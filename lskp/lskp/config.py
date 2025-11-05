# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Any
import torch
from dataclasses import dataclass
from vllm.logger import init_logger
logger = init_logger(__name__)

@dataclass
class LSKPEngineConfig:
    chunk_size: int
    local_device: Optional[str]
    max_local_cache_size: int
    save_decode_cache: bool  # whether to store decode kv cache
    enable_async_loading: bool = False

    @staticmethod
    def from_defaults(
        chunk_size: int = 256,
        local_device: str = "cuda",
        max_local_cache_size: int = 5,
        save_decode_cache: bool = False,
        enable_async_loading: bool = False,
    ) -> "LSKPEngineConfig":
        return LSKPEngineConfig(
            chunk_size,
            local_device,
            max_local_cache_size,
            save_decode_cache,
            enable_async_loading
        )

    def validate_and_set_config_value(self, config_key: str, value: Any) -> bool:
        """Validate and set configuration value"""
        if not hasattr(self, config_key):
            logger.warning(f"Config key '{config_key}' does not exist in configuration")
            return False

        try:
            setattr(self, config_key, value)
            return True
        except Exception as e:
            logger.error(
                f"Failed to set config item '{config_key}' with value {value}: {e}"
            )
            return False

@dataclass
class LSKPEngineMetadata:
    """name of the LLM model"""
    model_name: str
    """ world size when running under a distributed setting """
    world_size: int
    """ worker id when running under a distributed setting """
    worker_id: int
    """ the format of kv tensors """
    fmt: str
    """ the data type of kv tensors """
    kv_dtype: torch.dtype
    """ the shape of kv tensors """
    """ (num_layer, 2, chunk_size, num_kv_head, head_size) """
    kv_shape: tuple[int, int, int, int, int]
    """ the role of the current instance (e.g., 'scheduler', 'worker') """
    role: Optional[str] = None
    """ the first rank of the distributed setting """
    # TODO(baoloongmao): first_rank should be configurable
    first_rank = 0

    def is_first_rank(self) -> bool:
        """Check if the current worker is the first rank"""
        return self.worker_id == self.first_rank