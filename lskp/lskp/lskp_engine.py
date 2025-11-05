from typing import Optional, Any, Union
import torch
import time
from vllm.logger import init_logger
from vllm.config import VllmConfig
try:
    # Third Party
    from vllm.utils.torch_utils import get_kv_cache_torch_dtype
except ImportError:
    # Third Party
    from vllm.utils import get_kv_cache_torch_dtype

from lskp.config import LSKPEngineConfig, LSKPEngineMetadata
from lskp.gpu_connector import GPUConnectorInterface
from lskp.utils import KVCache

import lskp

logger = init_logger(__name__)

class LSKPEngine:
    """The main class for the cache engine.

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends.
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        config: LSKPEngineConfig,
        metadata: LSKPEngineMetadata,
        gpu_connector: Optional[GPUConnectorInterface] = None,
    ):
        logger.info("Initializing LSKP Cache Engine")
        self.config = config
        self.metadata = metadata
        self.chunk_size = config.chunk_size
        self.save_decode_cache = config.save_decode_cache
        self.gpu_connector = gpu_connector
    
    def retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve operation"
        )
        
        tot_kv_size = 0
        t = time.perf_counter()

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)

        # TODO: 定义 KVCache 在 Storage 中的存储结构

        # TODO: transfer KVCache from GPU to DRAM
        self.gpu_connector.batched_to_gpu()

    @torch.inference_mode()
    def store(
        self,
        tokens: torch.Tensor,
        kv_tensors_raw: KVCache,
        kv_tensors_mask: Optional[torch.Tensor] = None,
        skip_existing: bool = True,
        blocking: bool = True,
    ) -> None :
        """
        Store the KV cache of the tokens into the cache engine.
        Format: either 'huggingface' or 'vllm'

                For huggingface,
                it should have the shape of
                [num_heads, num_tokens, head_size]

                For vllm,
                it should have the shape of
                [num_tokens, num_heads, head_size]

        :param tokens: the input tokens, with shape [seq_len]
        :param kv_tensors_raw: the kv cache of the tokens, in
            the format of nested tuples. The number of tokens
            in the kv_tensors_raw should be the same as trues in
            kv_tensors_mask if mask is not None. Otherwise,
            it should be the same as the input tokens.
        :param kv_tensors_mask: a boolean mask of tokens indicating
            which tokens' KV Cache should be stored. Only support
            suffix mask. None is taken as trues for all tokens.
            len(kv_tensors_mask) should be the same as len(tokens)
            number of true should be the same as kv_tensors_raw token
            number.

        :param skip_existing: whether to skip the existing chunks
        :param blocking: whether to wait for the store operation to finish
        :return: None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        logger.info("Storing KV Cache into LSKP Engine")

        start_time = time.perf_counter()
        fmt = self.metadata.fmt
        if kv_tensors_mask is None:
            kv_tensors_mask = torch.ones_like(tokens, dtype=torch.bool)
        assert len(tokens.shape) == 1, f"Invalid shape of tokens: {tokens.shape}"
        assert len(kv_tensors_mask.shape) == 1, (
            f"Invalid shape of mask: {kv_tensors_mask.shape}"
        )
        assert len(tokens) == len(kv_tensors_mask), (
            "token length does not match mask length"
        )

        num_skip_tok = len(kv_tensors_mask) - torch.sum(kv_tensors_mask)
        assert num_skip_tok == 0 or skip_existing, (
            "When skip_existing is False, the mask must cover all tokens"
        )

        num_skip_chunk = num_skip_tok // self.chunk_size
        assert num_skip_tok == num_skip_chunk * self.chunk_size, (
            "Store KV mask should align to chunk size"
        )
        assert (
            len(tokens) == self._num_tokens_in_kv(kv_tensors_raw, fmt) + num_skip_tok
        ), "Number of tokens in the kv cache does not match the input tokens"

        # TODO: storage backend，transfer KVCache from GPU to DRAM
        pass


def init_lskp_engine(
    lskp_config: LSKPEngineConfig,
    vllm_config: VllmConfig,
    role: str
):
    """Initialize the LSKP Cache Engine  by the given model config and parallel
    config.

    Args:
        lskp_config (LSKPEngineConfig): The configuration for the LSKP Cache Engine.
        vllm_config (VllmConfig): The configuration for the vLLM serving engine.
        role (str): The role of this engine, either "server" or "client".

    Returns:
        LSKPEngine: The initialized LSKP Cache Engine.
    """
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config

    kv_dtype = get_kv_cache_torch_dtype(
        cache_config.cache_dtype,
        model_config.dtype
    )
    
    # construct KV Shape (for mem pool)
    num_layer = model_config.get_num_layers(parallel_config)
    chunk_size = lskp_config.chunk_size
    num_kv_head = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    # do not consider MLA here now
    kv_shape = (num_layer, 2, chunk_size, num_kv_head, head_size)
    logger.info(f"KV Cache Dtype: {kv_dtype}, KV Shape: {kv_shape}")

    # change current device
    num_gpus = torch.cuda.device_count()
    local_rank = parallel_config.local_rank % num_gpus
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    metadata = LSKPEngineMetadata(
        model_name=model_config.model,
        world_size=parallel_config.world_size,
        worker_id=parallel_config.rank,
        fmt="vllm",
        kv_dtype=kv_dtype,
        kv_shape=kv_shape,
        role=role,
    )

    vllm_gpu_connector: Optional[GPUConnectorInterface]
    hidden_dim_size = num_kv_head * head_size
    if role == "scheduler":
        vllm_gpu_connector = None
    else:
        vllm_gpu_connector = GPUConnectorInterface  # TODO: replace with actual implementation
    
    engine = LSKPEngine(
                    lskp_config,
                    metadata,
                    vllm_gpu_connector,
                )
    return engine




