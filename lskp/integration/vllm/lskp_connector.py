# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, Optional, Union, Generator
from dataclasses import dataclass, field
import torch
import vllm
from vllm.config import VllmConfig
from lskp.utils import cdiv
from lskp.config import LSKPEngineConfig
from lskp.lskp_engine import LSKPEngine, init_lskp_engine
from vllm.sampling_params import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

logger = init_logger(__name__)

@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LSKP
    lskp_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool

@dataclass
class DisaggSpec:
    req_id: str
    receiver_id: str
    receiver_host: str
    receiver_init_port: int
    receiver_alloc_port: int
    is_last_prefill: bool = False
    num_transferred_tokens: int = 0


tmp_disagg_tracker: dict[str, DisaggSpec] = {}


def extract_request_configs(sampling_params: SamplingParams) -> Optional[dict]:
    request_configs = None
    if sampling_params.extra_args is not None:
        if kv_transfer_params := sampling_params.extra_args.get("kv_transfer_params"):
            for k, v in kv_transfer_params.items():
                if k.startswith("lskp."):
                    if request_configs is None:
                        request_configs = {}
                    request_configs[k] = v
    return request_configs


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # Total prompt token length
    prompt_len: int

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    allocated_block_ids: list[int]

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: Optional[DisaggSpec] = None

    # The configs of the request, includes tags and other configs
    request_configs: Optional[dict] = None

    # Whether the request is in decode phase
    is_decode_phase = False

    # Whether the request cache should be saved
    skip_save: bool = False

    @staticmethod
    def from_new_request(
        LSKP_config: LSKPEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lskp_cached_tokens: int,
        skip_save: bool,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            LSKP_config (LSKPEngineConfig): the LSKP engine config.
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
            lskp_cached_tokens (int): the number of tokens that are
                cached in LSKP.
            request_priority (int): the priority of the request
            skip_save (bool): whether the request cache should be saved
        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # list[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            unfolded_block_ids = new_request.block_ids[0].copy()

        # NOTE: Initialized in `update_state_after_alloc`
        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)

        request_configs = extract_request_configs(new_request.sampling_params)

        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=lskp_cached_tokens,
            disagg_spec=disagg_spec,
            skip_save=skip_save,
            request_configs=request_configs,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[Optional[tuple[list[int], ...]], list[int]],
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again
        """

        self.token_ids.extend(new_token_ids)

        if new_block_ids is None:
            # https://github.com/vllm-project/vllm/commit/
            # b029de9902aa3ac58806c8c17776c7074175b6db#
            # diff-cafd89ce8a698a56acb24ada62831cbc7a980782f78a52d1742ba238031f296cL94
            new_block_ids = []
        elif len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            pass
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")
        self.allocated_block_ids.extend(new_block_ids)

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]  # torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor

    # Whether is last prefill or not
    is_last_prefill: bool = False

    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None
    # disagg spec
    disagg_spec: Optional[DisaggSpec] = None
    # the configs of the request
    request_configs: Optional[dict] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lskp_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lskp_chunk_size (int): the chunk size for LSKP.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            discard_partial_chunks (bool): whether to discard partial chunks.
            save_decode_cache (bool): whether to save the cache in decode phase.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        is_last_prefill = False
        if input_token_len == tracker.prompt_len:
            is_last_prefill = True

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        # 3. if save_decode_cache is False and it is in decode phase

        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, lskp_chunk_size) * lskp_chunk_size
        )

        # NOTE(vladnosiv): for disagg, you cannot skip saving, as saving is a transfer
        # Check if request_configs has lskp.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("lskp.skip_save", False)

        skip_save = tracker.disagg_spec is None and (
            tracker.skip_save
            or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)
            or (tracker.is_decode_phase and not save_decode_cache)
            or request_skip
        )

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        num_tokens_to_save = (
            (input_token_len // lskp_chunk_size * lskp_chunk_size)
            if not is_last_prefill or discard_partial_chunks
            else input_token_len
        )

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        token_ids = input_token_ids[:num_tokens_to_save]

        num_blocks = len(tracker.allocated_block_ids)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks."
                "Something might be wrong in scheduling logic!"
            )
            logger.error(
                "Num tokens: %d, num blocks: %d, block size: %d",
                len(token_ids),
                num_blocks,
                block_size,
            )

        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids.reshape((num_blocks, 1)) * block_size
        )

        slot_mapping = slot_mapping.flatten()[: len(token_ids)]
        assert slot_mapping.dtype == torch.long

        # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.lskp_cached_tokens,
                tracker.req_id,
            )
        else:
            # Do not load if not in `can_load` state
            load_spec = None

        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            is_last_prefill=is_last_prefill,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
        )

@dataclass
class LSKPConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)

class LSKPConnector(KVConnectorBase_V1):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._vllm_config = vllm_config
        self.kv_role = role
        self.worker_count = vllm_config.distributed_config.tensor_parallel_size
        config = LSKPEngineConfig.from_defaults()
        # vllm extra_config to the config
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        if kv_connector_extra_config:
            for key, value in kv_connector_extra_config.items():
                if key.startswith("LSKP."):
                    config_key = key[4:]  # Remove "LSKP." prefix
                    if self.validate_and_set_config_value(config_key, value):
                        logger.info(
                            f"Updated config {config_key} from vLLM "
                            f"extra config: {value}"
                        )
        self.config = config
        self.async_loading = config.enable_async_loading
        self.layerwise_retrievers: list[
            Generator[Optional[torch.Tensor], None, None]
        ] = []

        if role == KVConnectorRole.SCHEDULER:
            self.lskp_engine = init_lskp_engine(
                lskp_config=config,
                vllm_config=vllm_config,
                role="scheduler",
            )
        else:
            self.lskp_engine = init_lskp_engine(
                lskp_config=config,
                vllm_config=vllm_config,
                role="worker",
            )
        
        self.kv_caches: dict[str, torch.Tensor] = {}
        self._block_size = vllm_config.cache_config.block_size
        # request_id -> (vllm cached_tokens, lskp_cached_tokens)
        self.load_specs: dict[str, LoadSpec] = {}
        self._lskp_chunk_size = config.chunk_size
        self._save_decode_cache = config.save_decode_cache

        self.num_layer = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.current_layer = 0

    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue
            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]    

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be 
            the same.
            
        """
        logger.debug("LSKPConnector: start_load_kv called")
        self.current_layer = 0
        
        if len(self.kv_caches) == 0:
            self._init_kv_caches_from_forward_context(forward_context)
        
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LSKPConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        assert self.lskp_engine is not None
        self.layerwise_retrievers = []
        
        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None:
                continue
            last_idx = idx
        
        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None:
                continue
            tokens = request.token_ids
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = request.slot_mapping.cuda()
            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens // self._lskp_chunk_size * self._lskp_chunk_size
            )
            token_mask[:masked_token_count] = False

            lskp_cached_tokens = request.load_spec.lskp_cached_tokens

            if self.use_layerwise:
                # TODO: implement layerwise loading
                raise NotImplementedError("Layerwise loading is not implemented yet.")
            else:
                ret_token_mask = self.lskp_engine.retrieve(
                    tokens=tokens[:lskp_cached_tokens],
                    token_mask=token_mask[:lskp_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lskp_cached_tokens],
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                    skip_contains_check=True,
                )
            
            # Check the result
            num_retrieved_tokens = ret_token_mask.sum().item()
            num_expected_tokens = (
                lskp_cached_tokens - request.load_spec.vllm_cached_tokens
            )
            if num_retrieved_tokens < num_expected_tokens:
                logger.error(
                    "The number of retrieved tokens is less than expected!"
                )
                logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )
    
    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.
        
        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        # self._lskp_engine.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata",
                      **kwargs: Any) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer 
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current 
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        # self._lskp_engine.save_kv_layer(layer_name, kv_layer, attn_metadata,
        #                                   **kwargs)

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, LSKPConnectorMetadata)

        if self.kv_role == "kv_consumer":
            # Do not save if the role is kv_consumer
            return
        
        # TODO: implement the layer-wise saving logic
        if self.use_layerwise:
        #     for layerwise_storer in self.layerwise_storers:
        #         next(layerwise_storer)
            raise NotImplementedError("Layerwise saving is not implemented yet.")
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lskp_engine is not None

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if (
                save_spec is None
                or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            token_ids = request.token_ids
            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.cuda()

            skip_leading_tokens = save_spec.skip_leading_tokens
            if self.kv_caches == "kv_producer":
                skip_leading_tokens = min(
                    skip_leading_tokens, request.disagg_spec.num_transferred_tokens
                )
            
            if skip_leading_tokens == len(token_ids):
                continue

            # Align to lskp 
            skip_leading_tokens = (
                skip_leading_tokens // self._lskp_chunk_size * self._lskp_chunk_size
            )

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.info(
                "Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            token_len = len(token_ids)
            aligned_token_len = (
                token_len // self._lskp_chunk_size * self._lskp_chunk_size
            )
            token_ids = token_ids[:aligned_token_len]
            store_mask = store_mask[:aligned_token_len]
            slot_mapping = slot_mapping[:aligned_token_len]
        
        self.lskp_engine.store(
            tokens=token_ids,
            mask=store_mask,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            offset=skip_leading_tokens,
            transfer_spec=request.disagg_spec,
            request_configs=request.request_configs,
        )


    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        # return self._lskp_engine.get_finished(finished_req_ids)

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.
        
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the 
            external KV cache beyond what is already computed.
        """
        # return self._lskp_engine.get_num_new_matched_tokens(
            # request, num_computed_tokens), False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.
        """
        # self._lskp_engine.update_state_after_alloc(request,
        #                                             blocks, num_external_tokens)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        # return self._lskp_engine.build_connector_meta(scheduler_output)


    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        # return self._lskp_engine.request_finished(request, block_ids)
