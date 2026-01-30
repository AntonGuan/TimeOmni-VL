

from .vlm_dataset import SftJSONLIterableDataset

from .interleave_datasets.think_dataset import (
    ThinkingGenerationIterableDataset,
)


DATASET_REGISTRY = {
    'reasoning': SftJSONLIterableDataset,
    'understanding': SftJSONLIterableDataset,
    'forecasting': ThinkingGenerationIterableDataset,
    'imputation': ThinkingGenerationIterableDataset,
}


DATASET_INFO = {

    'reasoning': {
        'reasoning_thinking': {
            'data_dir': 'data_pipeline/data_demo',
            'jsonl_path': 'data_pipeline/data_demo/sft_dataset.jsonl',
            'num_total_samples': num_of_samples
        },
    },    

    'understanding': {
        'understanding_thinking': {
            'data_dir': 'data_pipeline/data_demo',
            'jsonl_path': 'data_pipeline/data_demo/all_qa_pairs.jsonl',
            'num_total_samples': num_of_samples
        },
    },    


    'forecasting': {
        'forecasting_thinking': {
            'data_dir': 'data_pipeline/data_demo',
            'jsonl_path': 'data_pipeline/data_demo/forecast_samples_thinking_gen.jsonl',
            'num_total_samples': num_of_samples
        },
    },
    'imputation': {
        'imputation_thinking': {
            'data_dir': 'data_pipeline/data_demo',
            'jsonl_path': 'data_pipeline/data_demo/imputation_samples_thinking_gen.jsonl',
            'num_total_samples': num_of_samples
        },
    },
}