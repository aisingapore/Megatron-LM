##
# Compile megatron.core.datasets.helpers dependencies before BlendedDataset import
##

import os
import tempfile
from collections import defaultdict
from typing import Dict, Optional

import numpy
import pytest
import torch
import math
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split, compile_helpers, get_blend_from_list
from tests.unit_tests.test_utilities import Utils
from megatron.training.global_vars import _GLOBAL_ARGS, get_args, set_global_variables
from argparse import Namespace
from megatron.core.parallel_state import initialize_model_parallel

_NUM_DATASETS = 10

_SEQUENCE_LENGTH = 10

_SIZES = {}
for split in Split:
    _SIZES[split] = []
    for i in range(_NUM_DATASETS):
        _SIZES[split].append({Split.train: 1000, Split.valid: 100, Split.test: 10}[split] * (i + 1))

"""
train: [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
valid: [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
test: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
"""
_MARGIN = 0.005

# Define the class here to avoid pytest warnings
class TestDataset(MegatronDataset):
    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: BlendedMegatronDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

        if self.num_samples is None:
            self.num_samples = len(self.indices)

        self.sample_index = numpy.random.choice(self.indices, size=self.num_samples)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str, config: BlendedMegatronDatasetConfig
    ) -> LowLevelDataset:
        return numpy.load(dataset_path)

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Dict[str, numpy.ndarray]:
        return {"text": self.dataset[self.sample_index[idx]]}

class TestDataset2(MegatronDataset):
    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: BlendedMegatronDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

        if self.num_samples is None:
            self.num_samples = len(self.indices)

        self.sample_index = numpy.random.choice(self.indices, size=self.num_samples)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str, config: BlendedMegatronDatasetConfig
    ) -> LowLevelDataset:
        return numpy.load(dataset_path)

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Dict[str, numpy.ndarray]:
        return self.dataset[self.sample_index[idx]]

def do_setup(odir):
    paths = defaultdict(list)

    for i in range(_NUM_DATASETS):
        path_to_data = os.path.join(odir, str(i))
        os.mkdir(path_to_data)

        for split in _SIZES:
            data = numpy.zeros((_SIZES[split][i], _SEQUENCE_LENGTH))
            path = os.path.join(path_to_data, f"{split.name}.npy")
            numpy.save(path, data)
            paths[split].append(path)

    return paths

def do_setup2(odir):
    paths = defaultdict(list)

    for i in range(_NUM_DATASETS):
        path_to_data = os.path.join(odir, str(i))
        os.mkdir(path_to_data)

        for split in _SIZES:
            data = numpy.full((_SIZES[split][i], _SEQUENCE_LENGTH), i + 1)
            path = os.path.join(path_to_data, f"{split.name}.npy")
            numpy.save(path, data)
            paths[split].append(path)

    return paths

"""
train split paths:
Dataset 0: /tmp/tmptxzol9u5/0/train.npy, 
containing numpy array of zeros with shape (_SIZES[Split.train][i], _SEQUENCE_LENGTH),
i.e. (1000, 10)
...
Dataset 9: /tmp/tmptxzol9u5/9/train.npy
containing numpy array of zeros with shape (_SIZES[Split.train][i], _SEQUENCE_LENGTH),
i.e. (10000, 10)


valid split paths:
...

test split paths:
Dataset 0: /tmp/tmptxzol9u5/0/test.npy
...
Dataset 9: /tmp/tmptxzol9u5/9/test.npy
"""

@pytest.fixture(autouse=True)
def cleanup_distributed():
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

def test_builder():
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()

        # Add cache path configuration
        cache_path = "/shared/aisingapore/.tmp_yuli/megatron_cache"  # or another appropriate path
        os.makedirs(cache_path, exist_ok=True)
        
        # When creating your BlendedDataset, pass the cache path
        dataset_config = {
            "path_to_cache": cache_path,
        }
        torch.distributed.barrier()
    else:
        compile_helpers()

    with tempfile.TemporaryDirectory() as temp_dir:

        paths = do_setup2(temp_dir)

        blends = {
            split: get_blend_from_list(
                [
                    weight_or_path
                    for pair in zip(list(range(1, len(paths[split]) + 1, 1)), paths[split])
                    for weight_or_path in pair
                ]
            )
            for split in Split
        }
        # blends is a dictionary where each key is a Split enum (train, valid, test)
        # and each value is a tuple. The first element of the tuple is a list of dataset
        # prefixes (strings), and the second element is either None or a list of weights (floats).

        blends_unweighted = {split: (blends[split][0], None) for split in blends}

# ####################################################################################################### Existing Tests
#         config = BlendedMegatronDatasetConfig(
#             random_seed=1234,
#             sequence_length=_SEQUENCE_LENGTH,
#             blend_per_split=[blends[Split.train], None, None],
#         )
#         try:
#             datasets = BlendedMegatronDatasetBuilder(
#                 TestDataset, [None, None, None], lambda: True, config
#             ).build()
#             raise RuntimeError
#         except AssertionError:
#             pass

#         config = BlendedMegatronDatasetConfig(
#             random_seed=1234,
#             sequence_length=_SEQUENCE_LENGTH,
#             blend_per_split=[get_blend_from_list([paths[Split.train][0]]), None, None],
#         )
#         datasets = BlendedMegatronDatasetBuilder(
#             TestDataset, [1000, None, None], lambda: True, config
#         ).build()
#         assert len(datasets[0]) == 1000 and isinstance(datasets[0], TestDataset)
#         assert datasets[1] is None
#         assert datasets[2] is None

#         config = BlendedMegatronDatasetConfig(
#             random_seed=1234,
#             sequence_length=_SEQUENCE_LENGTH,
#             blend_per_split=[
#                 blends_unweighted[Split.train],
#                 blends_unweighted[Split.valid],
#                 blends_unweighted[Split.test],
#             ],
#         )
#         datasets = BlendedMegatronDatasetBuilder(
#             TestDataset, [1000, 1000, 1000], lambda: True, config
#         ).build()
#         assert len(datasets[0]) == 1000
#         assert len(datasets[1]) == 1000
#         assert len(datasets[2]) == sum(_SIZES[Split.test])

#         config = BlendedMegatronDatasetConfig(
#             random_seed=1234,
#             sequence_length=_SEQUENCE_LENGTH,
#             blend_per_split=[
#                 blends_unweighted[Split.train],
#                 blends_unweighted[Split.valid],
#                 blends_unweighted[Split.test],
#             ],
#         )
#         datasets = BlendedMegatronDatasetBuilder(
#             TestDataset, [None, None, None], lambda: True, config
#         ).build()
#         assert len(datasets[0]) == sum(_SIZES[Split.train])
#         assert numpy.all(
#             numpy.array(datasets[0].weights)
#             == numpy.unique(datasets[0].dataset_index, return_counts=True)[1]
#         )
#         assert len(datasets[1]) == sum(_SIZES[Split.valid])
#         assert numpy.all(
#             numpy.array(datasets[1].weights)
#             == numpy.unique(datasets[1].dataset_index, return_counts=True)[1]
#         )
#         assert len(datasets[2]) == sum(_SIZES[Split.test])
#         assert numpy.all(
#             numpy.array(datasets[2].weights)
#             == numpy.unique(datasets[2].dataset_index, return_counts=True)[1]
#         )

#         config = BlendedMegatronDatasetConfig(
#             random_seed=1234,
#             sequence_length=_SEQUENCE_LENGTH,
#             blend_per_split=[blends_unweighted[Split.train], None, None],
#         )
#         datasets = BlendedMegatronDatasetBuilder(
#             TestDataset, [1000, None, None], lambda: True, config
#         ).build()
#         assert len(datasets[0]) == 1000
#         for i in range(_NUM_DATASETS):
#             assert len(datasets[0].datasets[i]) == _SIZES[Split.train][i]
#         assert datasets[1] is None
#         assert datasets[2] is None

#         config = BlendedMegatronDatasetConfig(
#             random_seed=1234,
#             sequence_length=_SEQUENCE_LENGTH,
#             blend_per_split=[blends[Split.train], None, None],
#         )
#         try:
#             datasets = BlendedMegatronDatasetBuilder(
#                 TestDataset, [1000, None, None], lambda: True, config
#             ).build()
#             raise RuntimeError
#         except IndexError:
#             ##
#             #
#             # The size per dataset is a function of the requested size, the weight per dataset,
#             # and a constant coefficient. The sizes, and consequently the total size to request,
#             # are modified such that the weights may or may not be sufficiently representative.
#             # To fix this, the weights should be reset according to the new sizes:
#             #
#             # S := size
#             # W := weights
#             #
#             # S = func(S, W)
#             #
#             # W = S / sum(S)
#             #
#             ##
#             config = BlendedMegatronDatasetConfig(
#                 random_seed=1234,
#                 sequence_length=_SEQUENCE_LENGTH,
#                 blend_per_split=[blends[Split.train], None, None],
#                 renormalize_blend_weights=True,
#             )
#             datasets = BlendedMegatronDatasetBuilder(
#                 TestDataset, [1000, None, None], lambda: True, config
#             ).build()
#             assert (
#                 len(datasets[0]) >= 1000
#                 and len(datasets[0]) <= 1000 * (1 + _MARGIN) + _NUM_DATASETS
#             )

#             config = BlendedMegatronDatasetConfig(
#                 random_seed=1234,
#                 sequence_length=_SEQUENCE_LENGTH,
#                 blend_per_split=[blends[Split.train], blends[Split.valid], blends[Split.test]],
#             )
#             datasets = BlendedMegatronDatasetBuilder(
#                 TestDataset, [100, 100, 100], lambda: True, config
#             ).build()
#             assert (
#                 len(datasets[0]) >= 100 and len(datasets[0]) <= 100 * (1 + _MARGIN) + _NUM_DATASETS
#             )
#             assert (
#                 len(datasets[1]) >= 100 and len(datasets[1]) <= 100 * (1 + _MARGIN) + _NUM_DATASETS
#             )
#             assert (
#                 len(datasets[2]) >= 100 and len(datasets[2]) <= 100 * (1 + _MARGIN) + _NUM_DATASETS
#             )

#         config = BlendedMegatronDatasetConfig(
#             random_seed=1234,
#             sequence_length=_SEQUENCE_LENGTH,
#             blend=blends_unweighted[Split.train],
#             split="100,0,0",
#         )
#         datasets = BlendedMegatronDatasetBuilder(
#             TestDataset, [None, None, None], lambda: True, config
#         ).build()
#         assert len(datasets[0]) == sum(_SIZES[Split.train])
#         assert numpy.all(
#             numpy.array(datasets[0].weights)
#             == numpy.unique(datasets[0].dataset_index, return_counts=True)[1]
#         )
#         assert datasets[1] is None
#         assert datasets[2] is None

#         if torch.distributed.is_initialized():
#             config = BlendedMegatronDatasetConfig(
#                 random_seed=1234,
#                 sequence_length=_SEQUENCE_LENGTH,
#                 blend=blends_unweighted[Split.train],
#                 split="100,0,0",
#             )
#             datasets = BlendedMegatronDatasetBuilder(
#                 TestDataset,
#                 [None, None, None],
#                 lambda: torch.distributed.get_rank() % 2 == 0,
#                 config,
#             ).build()
#             if torch.distributed.get_rank() % 2 == 0:
#                 assert len(datasets[0]) == sum(_SIZES[Split.train])
#                 assert numpy.all(
#                     numpy.array(datasets[0].weights)
#                     == numpy.unique(datasets[0].dataset_index, return_counts=True)[1]
#                 )
#             else:
#                 assert datasets[0] is None
#             assert datasets[1] is None
#             assert datasets[2] is None

#         config = BlendedMegatronDatasetConfig(
#             random_seed=1234,
#             sequence_length=_SEQUENCE_LENGTH,
#             blend=blends_unweighted[Split.train],
#             split="50,50,0",
#         )
#         datasets = BlendedMegatronDatasetBuilder(
#             TestDataset, [1000, 0, None], lambda: True, config
#         ).build()
#         assert len(datasets[0]) == 1000
#         assert sum(map(len, datasets[0].datasets)) == sum(_SIZES[Split.train]) / 2
#         assert sum(map(len, datasets[1].datasets)) == sum(_SIZES[Split.train]) / 2
#         assert datasets[1] is not None and len(datasets[1]) == 0
#         assert datasets[2] is None

#         config = BlendedMegatronDatasetConfig(
#             random_seed=1234,
#             sequence_length=_SEQUENCE_LENGTH,
#             blend=blends_unweighted[Split.train],
#             split="50,50,0",
#         )
#         datasets = BlendedMegatronDatasetBuilder(
#             TestDataset,
#             [int(sum(_SIZES[Split.train]) / 4), int(sum(_SIZES[Split.train])), None],
#             lambda: True,
#             config,
#         ).build()
#         assert len(datasets[0]) == sum(_SIZES[Split.train]) / 4
#         assert len(datasets[1]) == sum(_SIZES[Split.train]) / 2
#         assert datasets[2] is None

#         # 990 9 1
#         # 100000 1000 1
#         # []
#         config = BlendedMegatronDatasetConfig(
#             random_seed=1234,
#             sequence_length=_SEQUENCE_LENGTH,
#             blend=blends[Split.train],
#             split="990,9,1",
#         )
#         try:
#             # All three of 100000, 1000, and 1 result in error, yet 10000 and 100 do not
#             datasets = BlendedMegatronDatasetBuilder(
#                 TestDataset, [100000, 1000, 1], lambda: True, config
#             ).build()
#         except IndexError:
#             ##
#             #
#             # The size per dataset is a function of the requested size, the weight per dataset,
#             # and a constant coefficient. The sizes, and consequently the total size to request,
#             # are modified such that the weights may or may not be sufficiently representative.
#             # To fix this, the weights should be reset according to the new sizes:
#             #
#             # S := size
#             # W := weights
#             #
#             # S = func(S, W)
#             #
#             # W = S / sum(S)
#             #
#             ##
#             config = BlendedMegatronDatasetConfig(
#                 random_seed=1234,
#                 sequence_length=_SEQUENCE_LENGTH,
#                 blend=blends[Split.train],
#                 split="990,9,1",
#                 renormalize_blend_weights=True,
#             )
#             datasets = BlendedMegatronDatasetBuilder(
#                 TestDataset, [100000, 1000, 1], lambda: True, config
#             ).build()
#             assert (
#                 len(datasets[0]) >= 100000
#                 and len(datasets[0]) <= 100000 * (1 + _MARGIN) + _NUM_DATASETS
#             )
#             assert (
#                 len(datasets[1]) >= 1000
#                 and len(datasets[1]) <= 1000 * (1 + _MARGIN) + _NUM_DATASETS
#             )
#             assert len(datasets[2]) >= 1 and len(datasets[2]) <= 1 * (1 + _MARGIN) + _NUM_DATASETS

#             config = BlendedMegatronDatasetConfig(
#                 random_seed=1234,
#                 sequence_length=_SEQUENCE_LENGTH,
#                 blend=blends[Split.train],
#                 split="990,9,1",
#             )
#             datasets = BlendedMegatronDatasetBuilder(
#                 TestDataset, [10000, 100, 0], lambda: True, config
#             ).build()
#             assert (
#                 len(datasets[0]) >= 10000
#                 and len(datasets[0]) <= 10000 * (1 + _MARGIN) + _NUM_DATASETS
#             )
#             assert (
#                 len(datasets[1]) >= 100 and len(datasets[1]) <= 100 * (1 + _MARGIN) + _NUM_DATASETS
#             )
#             assert len(datasets[2]) == 0

########################################################################################################
        # Test the new feature of BlendedMegatronDatasetBuilder to generate the dict of multiple datasets with weights, 
        # after applying train/val/test split on each dataset
        # the returned value is a tuple with 3 values train_datasets, valid_datasets, test_datasets 
        # each is a "dataset_with_weight" dictionary with:
        # - keys as dataset prefices 
        # - values as dictionaries with 'dataset' and 'weight' keys
        blend = (
            [paths[Split.train][0], paths[Split.train][1], paths[Split.train][2], paths[Split.train][3], paths[Split.train][4]],  # First five dataset paths
            [0.4, 0.2, 0.15, 0.15, 0.1]  # Their weights
        )

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend=blend,  # Don't set blend directly
            split="990,9,1",
            stratified=True,
            renormalize_blend_weights=True,
        )
        
        # Sizes list should match the splits configuration
        sizes = [990,9,1]  # Only train split has size
        
        train_datasets, _, _ = BlendedMegatronDatasetBuilder(
            TestDataset, sizes, lambda: True, config
        ).build()
        
        # Verify dictionary structure for train split
        for prefix in train_datasets:
            assert 'dataset' in train_datasets[prefix]
            assert 'weight' in train_datasets[prefix]
            assert isinstance(train_datasets[prefix]['dataset'], TestDataset)
            assert isinstance(train_datasets[prefix]['weight'], (int, float))
            
        # Verify total size across stratified datasets matches expected
        train_total_size = sum(len(d['dataset']) for d in train_datasets.values())
        assert train_total_size >= 990 and train_total_size <= 990 * (1 + _MARGIN) + _NUM_DATASETS

        # Verify structure for stratified datasets
        assert isinstance(train_datasets, dict)
        assert len(train_datasets) == 5  # Should have 5 datasets
        
        # Get the dataset paths (will be like '0/train.npy' and '1/train.npy')
        dataset_paths = [paths[Split.train][0], paths[Split.train][1]]
        
        # Verify each dataset in the stratified dict
        for path, weight in zip(dataset_paths, [0.4, 0.2, 0.15, 0.15, 0.1]):
            assert path in train_datasets
            assert 'dataset' in train_datasets[path]
            assert 'weight' in train_datasets[path]
            assert isinstance(train_datasets[path]['dataset'], TestDataset)
            assert math.isclose(train_datasets[path]['weight'], weight)

        # Test StratifiedDataset, which takes the dict of dataset_with_weight as input, and returns a unified map type dataset 
        # that are indexed by (prefix, index) tuples
        from megatron.core.datasets.blended_megatron_dataset_builder import StratifiedDataset
        
        stratified_dataset = StratifiedDataset(train_datasets)
        
        # Test length
        total_length = sum(len(d['dataset']) for d in train_datasets.values())
        assert len(stratified_dataset) == total_length

        # Test __getitems__ functionality
        prefixes = list(train_datasets.keys())  # Get all dataset prefixes
        
        # Create a list of (prefix, index) tuples to test batch retrieval
        test_batch_indices = [
            (prefixes[0], 0),  # First item from first dataset
            (prefixes[1], 1),  # Second item from second dataset
            (prefixes[0], 2),  # Third item from first dataset
        ]
        
        # Get multiple items at once using __getitems__
        batch_items = stratified_dataset.__getitems__(test_batch_indices)
        
        # Verify the results
        assert len(batch_items) == len(test_batch_indices)
        for item in batch_items:
            # print(f"Item: {item}")

            assert isinstance(item, dict)
            assert 'text' in item
            assert isinstance(item['text'], numpy.ndarray)
            
        # Test error handling for invalid prefix
        invalid_indices = [(prefixes[0], 0), ('invalid_prefix', 0)]
        with pytest.raises(KeyError, match="Dataset with prefix 'invalid_prefix' not found"):
            stratified_dataset.__getitems__(invalid_indices)

        # Test MegatronPretrainingStratifiedSampler, which takes the "dataset_with_weight" as input, and returns a sampler
        # which yields micro batches of sample indices that are (prefix, index) tuples
        # the samples within each global batch also follows the exact weights defined in the "dataset_with_weight"
        from megatron.legacy.data.data_samplers import MegatronPretrainingStratifiedSampler

        # Initialize and create sampler
        sampler = MegatronPretrainingStratifiedSampler(
            dataset_with_weight=train_datasets,
            global_batch_size=32,
            micro_batch_size=8,
            consumed_samples=0,
            data_parallel_rank=torch.distributed.get_rank(),
            data_parallel_size=torch.distributed.get_world_size()
        )
       
        # Check samples per dataset
        total_samples_in_global_batch = sum(sampler.dataset_num_samples.values())

        # Assert statements to check if all global batch size equals total samples in global batch
        assert total_samples_in_global_batch == sampler.global_batch_size, (
            f"Total samples in global batch ({total_samples_in_global_batch}) "
            f"does not equal global batch size ({sampler.global_batch_size})"
        )

        for prefix, num_samples in sampler.dataset_num_samples.items():
            expected_num_samples = round(sampler.global_batch_size * train_datasets[prefix]['weight'])
            assert abs(num_samples - expected_num_samples) < 2, (
                f"Number of samples in {prefix} ({num_samples}) does not closely match "
                f"global batch size ({sampler.global_batch_size}) * weight ({train_datasets[prefix]['weight']})"
            )
            # print(f"{prefix}: {num_samples} (expected: {expected_num_samples})")


        # Test iteration
        iterator = iter(sampler)

        # Global batch to display
        global_batch_number_to_test = 2
        
        # Iterate over all micro batches inside one global batch
        for micro_batch_number, micro_batch_indices in enumerate(iterator):
            print(f"\nRank: {sampler.data_parallel_rank}, Micro Batch: {micro_batch_number + 1}:")
            assert len(micro_batch_indices) == sampler.micro_batch_size, (
                f"Micro Batch size: {len(micro_batch_indices)} (should be {sampler.micro_batch_size})"
            )
            for sample_id_in_micro_batch, (prefix, idx) in enumerate(micro_batch_indices):
                # print(f"  {sample_id_in_micro_batch}: Dataset: {prefix}, Index: {idx}")
                # Verify we can actually get this item
                item = sampler.dataset_with_weight[prefix]['dataset'][idx]
                # print(f"  Retrieved item type: {type(item)}")
            # only show the first global batch
            global_batch_number = (micro_batch_number + 1) * sampler.micro_batch_size * sampler.data_parallel_size // sampler.global_batch_size
            if global_batch_number >= global_batch_number_to_test:
                break

def test_build_stratified_dataloader():
    """Test the build_pretraining_data_loader function with external dataloader type"""
    from megatron.training.global_vars import _GLOBAL_ARGS, get_args, set_global_variables
    from megatron.core.parallel_state import (
        initialize_model_parallel,
        get_data_parallel_world_size,
        get_data_parallel_rank,
    )
    from argparse import Namespace

    # Initialize dummy args properly using set_global_variables
    args = Namespace(
        dataloader_type='external',
        num_workers=0,
        micro_batch_size=8,
        data_sharding=False,
        # Add required arguments for init_num_microbatches_calculator
        rank=0,
        rampup_batch_size=None,
        global_batch_size=32,
        data_parallel_size=2,
        decrease_batch_size_if_needed=False,
        # Add arguments for other initializations
        tensorboard_dir=None,
        tensorboard_queue_size=1,
        wandb_project=None,
        enable_one_logger=False,
        adlr_autoresume=False,
        timing_log_level=0,
        timing_log_option='minmax',
        exit_signal_handler=False,
    )
    
    # Skip tokenizer building since we don't need it for dataloader test
    set_global_variables(args, build_tokenizer=False)

    if torch.distributed.is_available():
        Utils.initialize_distributed()
        
        # Add this minimal initialization
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        
        if torch.distributed.get_rank() == 0:
            compile_helpers()

        # Add cache path configuration
        cache_path = "/shared/aisingapore/.tmp_yuli/megatron_cache"
        os.makedirs(cache_path, exist_ok=True)
        
        dataset_config = {
            "path_to_cache": cache_path,
        }
        torch.distributed.barrier()
    else:
        compile_helpers()

    # Now get_args() should work
    args = get_args()
    assert args.dataloader_type == 'external'
    
    with tempfile.TemporaryDirectory() as temp_dir:

        paths = do_setup2(temp_dir)

        blend = (
            [paths[Split.train][0], paths[Split.train][1], paths[Split.train][2], paths[Split.train][3], paths[Split.train][4]],  # First five dataset paths
            [0.4, 0.2, 0.15, 0.15, 0.1]  # Their weights
        )

        config = BlendedMegatronDatasetConfig(
            random_seed=1234,
            sequence_length=_SEQUENCE_LENGTH,
            blend=blend,  # Don't set blend directly
            split="990,9,1",
            stratified=True,
            renormalize_blend_weights=True,
        )
        
        # Sizes list should match the splits configuration
        sizes = [990,9,1]  # Only train split has size
        
        train_datasets, _, _ = BlendedMegatronDatasetBuilder(
            TestDataset2, sizes, lambda: True, config
        ).build()

        # first_item_key = next(iter(train_datasets))
        # print(f"First item of train_datasets: {first_item_key}: {train_datasets[first_item_key]}")

        from megatron.legacy.data.data_samplers import build_pretraining_data_loader
        
        # Set dataloader type to external
        args.dataloader_type = 'external'
        args.num_workers = 0  # For testing purposes
        
        # Build the dataloader
        dataloader = build_pretraining_data_loader(
            dataset=train_datasets,
            consumed_samples=64
        )
        
        print("\nTesting build_pretraining_data_loader:")
        print(f"World Size: {get_data_parallel_world_size()}")
        print(f"Rank: {get_data_parallel_rank()}")
        
        # Get one batch of data
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nRank {get_data_parallel_rank()} received batch {batch_idx + 1}:")
            print(f"Number of samples in batch: {len(batch)}")
            
            # Print the first few samples
            for i, sample in enumerate(batch):
                print(f"  Sample {i + 1}: dataset={sample[0]}, index={sample[:]}")
                
            # Only print first batch
            break

if __name__ == "__main__":
    # test_builder()
    test_build_stratified_dataloader()