
# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, Tuple, Union
import abc

import torch

class GPUConnectorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_gpu(self):
        raise NotImplementedError

    @abc.abstractmethod
    def from_gpu(self):
        raise NotImplementedError

    @abc.abstractmethod
    def batched_from_gpu(self):
        raise NotImplementedError

    @abc.abstractmethod
    def batched_to_gpu(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self) -> torch.Size:
        raise NotImplementedError

    def initialize_kvcaches_ptr(self, **kwargs):
        """Initialize the kvcaches pointers if not already initialized."""
        if "kvcaches" in kwargs:
            self.kvcaches = kwargs["kvcaches"]