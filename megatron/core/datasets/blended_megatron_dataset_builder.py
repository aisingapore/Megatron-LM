# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union, Tuple

import numpy
import torch

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split, normalize
from megatron.core.parallel_state import get_virtual_pipeline_model_parallel_rank
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

MidLevelDataset = MegatronDataset

TopLevelDataset = Union[BlendedDataset, MidLevelDataset]

DistributedDataset = Union[
    TopLevelDataset, MidLevelDataset, LowLevelDataset, torch.utils.data.Dataset
]

class StratifiedDataset:
    def __init__(self, datasets: dict):
        """
        Initialize the StratifiedDataset with a dictionary of datasets.

        Args:
            datasets (dict): A dictionary where each key is a prefix and each value is a dictionary
                             containing 'dataset' (an instance of MegatronDataset) and 'weight' (a float).
        """
        self.datasets = datasets

    def __len__(self):
        """
        Return the total number of samples in all datasets.

        Returns:
            int: The sum of the number of samples in all datasets.
        """
        total_samples = 0
        for dataset_info in self.datasets.values():
            dataset = dataset_info['dataset']
            total_samples += len(dataset)
        return total_samples

    def __getitems__(self, indices: List[Tuple[str, int]]) -> List[Any]:
        """
        Fetch samples from the datasets based on the provided indices.

        Args:
            indices (List[Tuple[str, int]]): A list of tuples where each tuple contains a dataset prefix
                                             and a sample index within that dataset.

        Returns:
            List[Any]: A list of samples fetched from the datasets.
        """
        samples = []
        for prefix, sample_index in indices:
            dataset_info = self.datasets.get(prefix)
            if dataset_info is not None:
                dataset = dataset_info['dataset']
                sample = dataset[sample_index]
                samples.append(sample)
            else:
                raise KeyError(f"Dataset with prefix '{prefix}' not found.")
        return samples

class BlendedMegatronDatasetBuilder(object):
    """Builder class for the BlendedDataset and MegatronDataset classes

    Args:
        cls (Type[MegatronDataset]): The class to instantiate, must inherit from MegatronDataset

        sizes (List[Optional[int]]): The minimum total number of samples to draw, or None, per split

        is_built_on_rank (Callable): A callable which returns True if the dataset should be built on the current rank and False otherwise. It should be Megatron Core parallelism aware i.e. global rank, local group rank, and virtual rank may inform its return value.

        config (BlendedMegatronDatasetConfig): The config object which informs dataset creation
    """

    def __init__(
        self,
        cls: Type[MidLevelDataset],
        sizes: List[int],
        is_built_on_rank: Callable,
        config: BlendedMegatronDatasetConfig,
    ):
        self.cls = cls
        self.sizes = sizes
        self.is_built_on_rank = is_built_on_rank
        self.config = config

        log_single_rank(
            logger,
            logging.INFO,
            f"Building dataset splits with cls={cls.__name__}, sizes={self.sizes}, and config={self.config}",
        )

        if not self.config.mock:
            for split in Split:
                size_is_none = self.sizes[split.value] is None
                if self.config.blend_per_split is None:
                    weights_are_none = self.config.blend[1] is None
                else:
                    if self.config.blend_per_split[split.value] is None:
                        continue
                    weights_are_none = self.config.blend_per_split[split.value][1] is None
                if size_is_none:
                    assert (
                        weights_are_none
                    ), f"size_is_none => weights_are_none fails for {split.name} split"

        if torch.distributed.is_initialized():
            gb_rank = torch.distributed.get_rank()
            vp_rank = get_virtual_pipeline_model_parallel_rank()
            if gb_rank == 0 and (vp_rank == 0 or vp_rank is None):
                assert (
                    self.is_built_on_rank()
                ), "is_built_on_rank must return True when global rank = 0 and vp rank = 0"

    def build(self) -> List[Optional[TopLevelDataset]]:
        """Build all dataset splits according to the provided blend(s)

        This method is distributed-aware and must be called on all ranks.

        The dataset splits returned can vary according to the config. Supply config.blend and
        config.split to build BlendedDataset and/or MegatronDataset splits from the same
        distribution. Supply config.blend_per_split to build BlendedDataset and/or MegatronDataset
        splits from separate distributions. In either case, for each split, handle the following
        cases:

        (1) The split is None
            - do nothing

        (2) The split has one contributing dataset, and...

            (a) 'size' is not None
                - Build a mid-level dataset with low-level dataset sampling in proportion to the size

            (b) 'size' is None
                - Build mid-level datasets with no excess low-level dataset sampling

        (3) The split has multiple contributing datasets, and...

            (a) 'weights' is not None and 'size' is not None
                - Build mid-level datasets with low-level dataset sampling in proportion to their weights and the size
                - Build a top-level dataset of length marginally greater than 'size' with mid-level dataset sampling in proportion to their weights and the size

            (b) 'weights' is not None and 'size' is None
                - Error

            (c) 'weights' is None and 'size' is not None
                - Build mid-level datasets with no excess low-level dataset sampling
                - Build a top-level dataset of length 'size' with mid-level dataset sampling in proportion to their lengths and the size

                  - The 'size' of the top-level dataset is capped at the sum of the mid-level dataset lengths

            (d) 'weights' is None and 'size' is None
                - Build mid-level datasets with no excess low-level dataset sampling
                - Build a top-level dataset with no excess mid-level dataset sampling

        Returns:
            Union[List[Optional[TopLevelDataset]], List[Dict[str, Dict[str, Union[MegatronDataset, float]]]]]: A list containing a dataset instance (or None) per split
        """
        datasets = self._build_blended_dataset_splits()

        if self.config.stratified:
            return datasets

        for dataset in datasets:
            if dataset is not None and len(dataset) > 0:
                if isinstance(dataset, BlendedDataset):
                    if dataset.built_anew_on_cache_miss or any(
                        x.built_anew_on_cache_miss for x in dataset.datasets
                    ):
                        log_single_rank(
                            logger,
                            logging.INFO,
                            f"Verifying NumPy indices for {type(dataset).__name__} {dataset.split.name} split",
                        )
                    else:
                        log_single_rank(
                            logger,
                            logging.INFO,
                            f"NumPy indices for {type(dataset).__name__} {dataset.split.name} split are fully cached, skipping verification",
                        )
                        continue
                    # Check blend size
                    assert dataset.size is None or dataset.size == dataset.dataset_index.shape[0]
                    # Check blend access of mid-level datasets
                    _, sizes = numpy.unique(dataset.dataset_index, return_counts=True)
                    for i, dataset_and_size in enumerate(zip(dataset.datasets, sizes)):
                        if len(dataset_and_size[0]) < dataset_and_size[1]:
                            raise IndexError(
                                f"The {dataset.split.name} blend oversamples (N = {dataset_and_size[1]}) {type(dataset_and_size[0]).__name__} {i} (len = {len(dataset_and_size[0])}). "
                                f"Set renormalize_blend_weights to True and re-run. File an issue if the problem is not resolved."
                            )

        return datasets

    def _build_blended_dataset_splits(
        self
    ) -> Union[
        List[Optional[TopLevelDataset]], 
        List[Dict[str, Dict[str, Union[LowLevelDataset, float]]]]
    ]:
        """Build all dataset splits according to the provided blend(s)

        See the BlendedMegatronDatasetBuilder.build alias for more information.

        Returns:
            Union[List[Optional[TopLevelDataset]], List[Dict[str, Dict[str, Union[MegatronDataset, float]]]]]: 
                - If not using stratified batching:
                    A list containing a dataset instance (or None) per split
                - If using stratified batching:
                    A list of dictionaries, where each dictionary contains:
                    {
                        prefix: {
                            'dataset': MegatronDataset,
                            'weight': float
                        }
                    }
                    for each enabled split
        """
        ##
        # Return fake "mock" datasets
        ##
        if self.config.mock:
            split = self.config.split_matrix
            try:
                return self._build_megatron_dataset_splits(None, split, self.sizes)
            except Exception as error:
                raise Exception(
                    f"{self.cls.__name__} failed to build as a mock data generator"
                ) from error

        ##
        # All splits come from the same distribution
        ##
        elif self.config.blend:
            prefixes, weights = self.config.blend
            if weights is not None:
                weights = normalize(weights)

            split = self.config.split_matrix

            # If we only have one dataset prefix and no weights specified,
            # we can directly build a single dataset without blending
            if len(prefixes) == 1 and weights is None:
                return self._build_megatron_dataset_splits(prefixes[0], split, self.sizes)

            # For multiple datasets, we need to determine the size of each dataset per split
            # If no weights are provided, set all sizes to None to use full dataset sizes
            if weights is None:
                # Create a 2D list of None values with dimensions:
                # [num_prefixes][num_splits]
                sizes_per_dataset = [[None for split in Split] for prefix in prefixes]
            else:
                # Calculate the size of each dataset per split based on the provided weights
                # This ensures datasets are sampled proportionally according to weights
                sizes_per_dataset = _get_size_per_split_per_dataset(weights, self.sizes)

            # Build all the individual datasets in parallel for efficiency
            # Each dataset is built according to its prefix, split configuration,
            # and target size per split
            megatron_datasets = self._build_megatron_datasets_parallel(
                prefixes, split, sizes_per_dataset
            )

            # If using stratified batching, return datasets with their proportions
            if self.config.stratified:
                stratified_datasets = []
                # Only iterate through enabled splits based on split matrix
                for i in range(len(Split)):
                    if split[i] is not None:
                        split_dict = {}
                        for prefix, dataset, weight in zip(prefixes, megatron_datasets[i], weights):
                            split_dict[prefix] = {'dataset': dataset, 'weight': weight}
                        stratified_datasets.append(split_dict)
                return stratified_datasets

            # Build the top-level datasets by blending multiple datasets together
            blended_datasets = [None] * len(Split)  # Initialize list to store blended datasets for each split
            for i in range(len(Split)):  # Iterate through each split (train/val/test)
                if split[i] is not None:  # Only process if this split is enabled
                    weights_i = weights  # Get weights for blending datasets
                    
                    # Case 1: We have predefined weights and sizes
                    if weights_i is not None and self.sizes[i] is not None:
                        # Get size for each dataset in this split
                        size_per_dataset = list(zip(*sizes_per_dataset))[i]
                        size_i = sum(size_per_dataset)  # Total size for this split
                        # Optionally renormalize weights based on actual dataset sizes
                        if self.config.renormalize_blend_weights:
                            weights_i = list(map(lambda _size: _size / size_i, size_per_dataset))
                    
                    # Case 2: No predefined weights - use dataset lengths as weights
                    elif weights_i is None:
                        try:
                            # Try to get lengths of each dataset to use as weights
                            weights_i = [
                                len(megatron_dataset) for megatron_dataset in megatron_datasets[i]
                            ]
                        except TypeError:
                            # If lengths not available, use equal weights of 0
                            weights_i = [0 for _ in prefixes]
                            
                        # Set size to either specified size or sum of dataset lengths
                        if self.sizes[i] is not None:
                            size_i = min(self.sizes[i], sum(weights_i))
                        else:
                            size_i = None  # Will default to sum of all dataset lengths
                    
                    # Case 3: Invalid state - weights without sizes
                    else:
                        raise RuntimeError
                        
                    # Create the blended dataset by combining individual datasets
                    blended_datasets[i] = self.build_generic_dataset(
                        BlendedDataset,  # Class to instantiate
                        self.is_built_on_rank,  # Whether to build on this rank
                        True,  # synchronize_ranks: build on rank-0 first
                        megatron_datasets[i],  # List of datasets to blend
                        weights_i,  # Weights for blending
                        size_i,  # Total size of blended dataset
                        self.config,  # Configuration object
                    )

            return blended_datasets

        ##
        # Each split comes from a separate distribution
        ##
        else:
            blended_datasets = [None] * len(Split)
            for i in range(len(Split)):
                split_spoof = [None] * len(Split)
                split_spoof[i] = (0.0, 1.0)
                sizes_spoof = [0] * len(Split)
                sizes_spoof[i] = self.sizes[i]

                # Blend is provided for the split
                blend = self.config.blend_per_split[i]
                if blend is not None:
                    prefixes, weights = blend
                    if weights is not None:
                        weights = normalize(weights)

                    # Blend consists of a sigle prefix
                    if len(prefixes) == 1:
                        blended_datasets[i] = self._build_megatron_dataset_splits(
                            prefixes[0], split_spoof, sizes_spoof
                        )[i]
                        continue

                    # Build mid-level datasets
                    if weights is None:
                        sizes_per_dataset = [[None for split in Split] for prefix in prefixes]
                    else:
                        sizes_per_dataset = _get_size_per_split_per_dataset(weights, sizes_spoof)

                    # build each dataset in parallel
                    megatron_datasets = self._build_megatron_datasets_parallel(
                        prefixes, split_spoof, sizes_per_dataset
                    )[i]

                    # Build top-level dataset
                    if weights is not None and self.sizes[i] is not None:
                        size_per_dataset = list(zip(*sizes_per_dataset))[i]
                        size = sum(size_per_dataset)
                        if self.config.renormalize_blend_weights:
                            weights = list(map(lambda _size: _size / size, size_per_dataset))
                    elif weights is None:
                        try:
                            weights = [
                                len(megatron_dataset) for megatron_dataset in megatron_datasets
                            ]
                        except TypeError:
                            weights = [0 for _ in prefixes]
                        if self.sizes[i] is not None:
                            size = min(self.sizes[i], sum(weights))
                        else:
                            size = None  # => the size will be sum(weights)
                    else:
                        raise RuntimeError
                    blended_datasets[i] = self.build_generic_dataset(
                        BlendedDataset,
                        self.is_built_on_rank,
                        True,  # synchronize_ranks, default behavior to build on rank-0 first
                        megatron_datasets,
                        weights,
                        size,
                        self.config,
                    )

            return blended_datasets

    def _build_megatron_datasets_parallel(
        self, prefixes: List[str], split: List[float], sizes_per_dataset: List[List[int]]
    ) -> List[List[Optional[MegatronDataset]]]:
        """Build the megatron datasets for a list of prefixes in parallel

        Args:
            prefixes (List[str]): The list of prefix strings

            split (List[float]): The dataset split ratios (must sum to 1.00)

            sizes_per_dataset (List[List[int]]): The number of samples to request
            per MegatronDataset per spilt

        Returns:
            List[List[Optional[MegatronDataset]]]: For each split, have a list of
            MegatronDataset per prefix
        """

        # Helper function to wrap the threading logic
        def _threading_helper(
            megatron_datasets: List[List[Optional[MegatronDataset]]],
            num_workers: int,
            prefixes: List[str],
            split: List[float],
            sizes_per_dataset: List[List[int]],
        ) -> None:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                all_futures = []
                for i in range(len(prefixes)):
                    all_futures.append(
                        executor.submit(
                            self._build_megatron_dataset_splits,
                            prefixes[i],
                            split,
                            sizes_per_dataset[i],
                            False,  # synchronize_ranks, barrier is called in this function
                        )
                    )
                for future in all_futures:
                    try:
                        megatron_datasets_split = future.result()
                        for j in range(len(megatron_datasets_split)):
                            megatron_datasets[j].append(megatron_datasets_split[j])
                    except Exception as err:
                        raise err

        megatron_datasets = [[] for _ in range(len(Split))]
        num_dataset_builder_threads = self.config.num_dataset_builder_threads

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            # First, build on rank 0
            if rank == 0:
                num_workers = num_dataset_builder_threads
                if num_workers > 1:
                    # since only rank 0 is running, scale up the thread count
                    # but not too much to avoid overloading storage on miss path.
                    # if user set num_dataset_builder_threads to 1,
                    # i.e. meant for serial build, do not scale up.
                    num_workers *= min(2, max(1, torch.cuda.device_count()))
                _threading_helper(
                    megatron_datasets, num_workers, prefixes, split, sizes_per_dataset
                )

            torch.distributed.barrier()

            # Then, build on other ranks; guaranteed to be data_cache hit
            if rank != 0:
                _threading_helper(
                    megatron_datasets,
                    num_dataset_builder_threads,
                    prefixes,
                    split,
                    sizes_per_dataset,
                )
        else:
            _threading_helper(
                megatron_datasets, num_dataset_builder_threads, prefixes, split, sizes_per_dataset
            )

        return megatron_datasets

    def _build_megatron_dataset_splits(
        self,
        dataset_path: Optional[str],
        split: List[float],
        sizes: List[int],
        synchronize_ranks: bool = True,
    ) -> List[Optional[MidLevelDataset]]:
        """Build each MidLevelDataset split from a single LowLevelDataset

        Args:
            dataset_path (Optional[str]): The path on disk which defines the underlying LowLevelDataset, or None for mock dataset classes

            split (List[Tuple[float, float]]): The dataset split matrix

            sizes (List[int]): The number of total samples to draw from each split

            synchronize_ranks (bool): Whether to call barrier for rank-0 / barrier / other-ranks behavior. Set to False when we enforce this behavior at higher level.

        Returns:
            List[Optional[MidLevelDataset]]: The MidLevelDataset (or None) per split
        """
        # short-cut if we are not building on this rank
        if torch.distributed.is_initialized() and not self.is_built_on_rank():
            for i in range(len(Split)):
                if split[i] is not None and synchronize_ranks:
                    torch.distributed.barrier()
            return [None] * len(Split)

        # Build the low level dataset
        low_level_dataset = self.cls.build_low_level_dataset(dataset_path, self.config)

        # Build the split indices for the low level dataset
        num_elements = self.cls.numel_low_level_dataset(low_level_dataset)
        split_indices = []
        for i, _ in enumerate(Split):
            if split[i] is not None:
                beg = int(round(split[i][0] * float(num_elements)))
                end = int(round(split[i][1] * float(num_elements)))
                split_indices.append(numpy.arange(start=beg, stop=end, step=1, dtype=numpy.int32))
            else:
                split_indices.append(None)

        # Build the mid level dataset
        mid_level_datasets = []
        for i, _split in enumerate(Split):
            if split[i] is None:
                mid_level_datasets.append(None)
            else:
                mid_level_datasets.append(
                    self.build_generic_dataset(
                        self.cls,
                        self.is_built_on_rank,
                        synchronize_ranks,
                        low_level_dataset,
                        dataset_path,
                        split_indices[i],
                        sizes[i],
                        _split,
                        self.config,
                    )
                )

        return mid_level_datasets

    @staticmethod
    def build_generic_dataset(
        cls: Union[Type[DistributedDataset], Callable],
        is_built_on_rank: Callable,
        synchronize_ranks: bool,
        *args: Any,
    ) -> Optional[Union[DistributedDataset, Iterable]]:
        """Build the DistributedDataset

        Return None if and only if the underlying dataset class is not built on the current rank
        and torch.distributed is initialized.

        Args:
            cls (Union[Type[DistributedDataset], Callable]): The DistributedDataset class to be built. In special cases, e.g. when we are building the low level dataset for a RawMegatronDataset instance, we can accept a Callable which returns an Iterable.

            synchronize_ranks (bool): Whether to call barrier for rank-0 / barrier / other-ranks behavior. Set to False when we enforce this behavior at higher level.

            args (Tuple[Any]): The positional arguments used to build the provided DistributedDataset class

        Raises:
            Exception: When the dataset constructor raises an OSError

        Returns:
            Optional[Union[DistributedDataset, Iterable]]: The DistributedDataset instantion, the Iterable instantiation, or None
        """
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

            dataset = None

            # First, build on rank 0
            if rank == 0 and is_built_on_rank():
                try:
                    dataset = cls(*args)
                except OSError as err:
                    log = (
                        f"Failed to write dataset materials to the data cache directory. "
                        + f"Please supply a directory to which you have write access via "
                        + f"the path_to_cache attribute in BlendedMegatronDatasetConfig and "
                        + f"retry. Refer to the preserved traceback above for more information."
                    )
                    raise Exception(log) from err

            if synchronize_ranks:
                torch.distributed.barrier()

            # After, build on other ranks
            if rank != 0 and is_built_on_rank():
                dataset = cls(*args)

            return dataset

        return cls(*args)


def _get_size_per_split_per_dataset(
    normalized_weights: List[float], target_size_per_split: List[int]
) -> List[List[int]]:
    """Determine the contribution of the MegatronDataset splits to the BlendedDataset splits

    Args:
        normalized_weights (List[float]): e.g. [0.3, 0.7]

        target_size_per_split (List[int]): The number of samples to target for each BlendedDataset split

    Returns:
        List[List[int]]: The number of samples to request per MegatronDataset per split
    """
    assert numpy.isclose(sum(normalized_weights), 1.0)

    # Use 0.5% target margin to ensure we satiate the request
    sizes_per_dataset = [
        [int(math.ceil(target_size * weight * 1.005)) for target_size in target_size_per_split]
        for weight in normalized_weights
    ]

    return sizes_per_dataset
