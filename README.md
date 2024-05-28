# PruneGPT

## Overview
This framework provides tools for pruning layers in Large Language Models (LLMs). Currently, quantization via bitsandbytes is supported for loading the model and computing the pruning criterion. Any pruning job can be defined by a valid [config.json](configs/config.json) and started by calling [main.py](src/main.py).

## Installation

1. Install dependencies using `pip install -r requirements.txt`.
2. Run the pruning using `python src/main.py --config_path "configs/config.json"`.

## Configuration Arguments:
A sample config is provided here. Below is a breakdown of each parameter within the file:

### Model Settings
- **`model_name`**: Specifies the full identifier of the model to prune.  
  Example: `"microsoft/Phi-3-mini-4k-instruct"`

- **`quantize`**: Enables or disables the quantization of model weights via bitsandbytes.  
  Boolean value: `true` or `false`

### Bit Packing and Quantization (BnB Config)
- **`bnb_config`**:
  - **`load_in_4bit`**: Loads model weights in 4-bit precision to reduce memory footprint.  
    Boolean value: `true` or `false`
  - **`bnb_4bit_quant_type`**: Type of 4-bit quantization applied.  
    Example: `"nf4"`
  - **`bnb_4bit_use_double_quant`**: Utilizes double quantization for increased precision.  
    Boolean value: `true` or `false`
  - **`bnb_4bit_compute_dtype`**: Data type used during the quantization compute process.  
    Example: `"float16"`

### Dataset Configuration
- **`dataset`**: Defines the dataset used for evaluating the pruning effectiveness.
  - **`name`**: Dataset name.  
    Example: `"wikitext"`
  - **`subset`**: Specific subset of the dataset.  
    Example: `"wikitext-2-raw-v1"`
  - **`split`**: Which split of the dataset to use.  
    Example: `"test"`
  - **`batch_size`**: Number of samples per batch during evaluation.  
    Integer value: `8`

### Pruning Parameters
- **`num_blocks_to_prune`**: Total number of model blocks to prune.  
  Integer value: `5`
- **`pruning_method`**: Methodology used for pruning. We currently support [angular_distance](https://arxiv.org/abs/2403.17887) and [cosine_similarity](https://arxiv.org/abs/2403.03853).
- **`pruning_token`**: Tokens used for computing pruning metrics.
  Example: `"all" or "last"`

### Output Configuration
- **`save_directory`**: Path where the pruned model will be saved.  
  Example: `"models/phi3_pruned_model"`
- **`log_file`**: File to store logging information during pruning.  
  Example: `"pruning_phi3.log"`
- **`calculate_ppl`**: Calculates and logs the perplexity of the pruned model to evaluate performance.  
  Boolean value: `true` or `false`

## Licsence
This codebase is open-source and free to use for non-commercial purposes.

## Support
For issues and support, please file an issue in the repository issue tracker.