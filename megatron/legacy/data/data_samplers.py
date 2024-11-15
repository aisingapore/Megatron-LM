# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""


import random
import torch
import numpy as np
from torch.utils.data import Dataset
from megatron.training import get_args
from megatron.core import mpu
from typing import List, Tuple, Any
from megatron.core.datasets.blended_megatron_dataset_builder import StratifiedDataset

def build_pretraining_data_loader(dataset, consumed_samples):
    """Build dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args()

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding)
    elif args.dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        # return dataset
        batch_sampler = MegatronPretrainingStratifiedSampler(
            dataset_with_weight=dataset,
            global_batch_size=args.global_batch_size,
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
        
        # Merge the datasets for DataLoader, i.e.
        # from: {prefix: {'weight': float, 'total_samples': int, 'dataset': MegatronDataset}} 
        # to: StratifiedDataset
        dataset = StratifiedDataset(dataset)
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       persistent_workers=True if args.num_workers > 0 else False,
                                       )

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class RandomSeedDataset(Dataset):

    def __init__(self, dataset):
        args = get_args()
        self.base_seed = args.seed
        self.curr_seed = args.seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, data_sharding):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                           * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = \
                torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

class MegatronPretrainingStratifiedSampler:
    """Sampler that handles multiple datasets with specified weights"""
    
    def __init__(
        self,
        dataset_with_weight: dict,  # Dict of {prefix: {'weight': float, 'dataset': MegatronDataset}}
        global_batch_size: int,
        micro_batch_size: int,
        consumed_samples: int,
        data_parallel_rank: int,
        data_parallel_size: int
    ):
        # Validate dataset_with_weight data type
        assert isinstance(dataset_with_weight, dict), "dataset_with_weight must be a dictionary"
        for key, value in dataset_with_weight.items():
            assert isinstance(key, str), "Each key in dataset_with_weight must be a string"
            assert isinstance(value, dict), "Each value in dataset_with_weight must be a dictionary"
            assert 'weight' in value, "Each dictionary in dataset_with_weight must contain 'weight' key"
            assert isinstance(value['weight'], float), "'weight' must be a float"
            assert 'dataset' in value, "Each dictionary in dataset_with_weight must contain 'dataset' key"

        self.dataset_with_weight = dataset_with_weight
        self.global_batch_size = global_batch_size
        self.micro_batch_size = micro_batch_size
        self.consumed_samples = consumed_samples
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        
        # Validate weights sum to 1
        total_weight = sum(info['weight'] for info in dataset_with_weight.values())
        assert abs(total_weight - 1.0) < 1e-6, f"weights must sum to 1, got {total_weight}"
        
        # Assert that the global_batch_size is divisible by (micro_batch_size * data_parallel_size)
        assert global_batch_size % (micro_batch_size * data_parallel_size) == 0, (
            f"global_batch_size ({global_batch_size}) must be divisible by "
            f"micro_batch_size ({micro_batch_size}) * data_parallel_size ({data_parallel_size})"
        )
        
        # Calculate samples per dataset in the global batch using round
        self.dataset_num_samples = {
            prefix: round(global_batch_size * info['weight'])
            for prefix, info in dataset_with_weight.items()
        }
        
        # Adjust rounding errors
        total_samples = sum(self.dataset_num_samples.values())
        if total_samples != global_batch_size:
            # Calculate the difference
            difference = global_batch_size - total_samples
            
            # Adjust the dataset with the largest weight
            max_weight = 0
            max_prop_dataset = None
            for dataset, info in dataset_with_weight.items():
                if info['weight'] > max_weight:
                    max_weight = info['weight']
                    max_prop_dataset = dataset
            
            # Adjust the number of samples for the largest weight dataset
            self.dataset_num_samples[max_prop_dataset] += difference

    def __len__(self):
        # Sum the lengths of all datasets
        return sum(len(info['dataset']) for info in self.dataset_with_weight.values())

    def __collate_global_batch__(self):
        """Collates samples from all datasets into a global batch.
        
        Returns:
            list: List of (prefix, idx) tuples representing the global batch
        """
        global_batch_indices = []
        
        # For each dataset, generate its portion of samples for the global batch
        for prefix, info in self.dataset_with_weight.items():
            num_samples = self.dataset_num_samples[prefix]
            total_samples_local = len(info['dataset'])  # Get the length of the dataset
            weight_local = info['weight']
            
            # Calculate local consumed samples and epoch
            consumed_samples_local = int(self.consumed_samples * weight_local)
            epoch_local = consumed_samples_local // total_samples_local
            bucket_offset_local = consumed_samples_local % total_samples_local
            
            # Generate random permutation for this dataset
            g_local = torch.Generator()
            g_local.manual_seed(epoch_local)
            
            # Generate indices for this dataset's portion
            indices = torch.randperm(total_samples_local, generator=g_local).tolist()[bucket_offset_local:]
            
            # Add (dataset, index) tuples to global batch
            global_batch_indices.extend([(prefix, idx) for idx in indices[:num_samples]])

        # Permute the global_batch_indices according to the epoch_global
        effective_length = (self.__len__() // self.global_batch_size) * self.global_batch_size
        epoch_global = self.consumed_samples // effective_length
        g_global = torch.Generator()
        g_global.manual_seed(epoch_global)
        permuted_indices = torch.randperm(len(global_batch_indices), generator=g_global).tolist()
        global_batch_indices = [global_batch_indices[i] for i in permuted_indices]
            
        return global_batch_indices

    def __iter__(self):
        while True:
            # Get collated global batch
            global_batch_indices = self.__collate_global_batch__()
            # print(f"{self.data_parallel_rank}: global_batch_indices regenerated")
            
            # Calculate the number of micro-batches in the global batch
            num_micro_batches = self.global_batch_size // self.micro_batch_size
            num_micro_batches_per_rank = num_micro_batches // self.data_parallel_size
            # print(f"{self.data_parallel_rank}: num_micro_batches = {self.global_batch_size} // {self.micro_batch_size} = {num_micro_batches}")
            
            # Interleave indices for each rank
            for i in range(num_micro_batches_per_rank):
                # Calculate the starting index for this rank
                start_idx = self.data_parallel_rank + i * (self.data_parallel_size * self.micro_batch_size)
                # Collect indices for this rank's global-batch
                # print(f"Slicing of rank_indices: {list(range(start_idx, self.global_batch_size, self.data_parallel_size))}")
                rank_indices = global_batch_indices[start_idx:self.global_batch_size:self.data_parallel_size]
                # Collect indices for this rank's micro-batch
                micro_batch_indices=rank_indices[:self.micro_batch_size]
                # print(f"{self.data_parallel_rank}: micro_batch_indices = {micro_batch_indices}")
                
                yield micro_batch_indices
            
            # Update consumed samples
            self.consumed_samples += self.global_batch_size
