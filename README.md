# PruneGPT

## Overview
This minimilistic framework provides tools for pruning layers in Large Language Models (LLMs). Currently, quantization via bitsandbytes is supported for loading the model and computing the pruning criterion. Any pruning job can be defined by a valid [config.json](configs/config.json) and started by calling [main.py](src/main.py).

## ToDo

- [ ] Add support for width pruning [FLAP](https://arxiv.org/abs/2312.11983)
- [ ] Add support for Taylor and PPL metric [Shortened LLaMA](https://arxiv.org/pdf/2402.02834)
- [ ] Benchmark training-free pruning of larger models 
  - [x] Meta-Llama-3-70B
  - [ ] mixtral-8x22B-v0.3
  - [ ] Phi-3-medium-4k-instruct

## Pruning Results

The following table summarizes downstream task performance of pruning on larger models -

|  | MBZUAI K2-65B | Meta-Llama2-70B | Meta-Llama3-70B | Nyun-Llama3-62B | Nyun-Llama3-60B |
| --- | --- | --- | --- | --- | --- | 
| MMLU (5-shot) | 67.9 | 69.7 | 79.5 | 78.9 | 78.6 | 
| Winogrande (5-shot) | 77.0 | 81.8 | 83.1 | 83.3 | 83.4 | 
| BoolQ (0-shot) | 83.0 | 73.1 | 79.0 | 85.3 | 85.2 | 
| Hellaswag (10-shot) | 85.5 | 86.9 | 88.0 | 85.8 | 85.7 |  
| Arc Challenge (25-shot) | 64.8 | 67.2 | 68.8 | 65.9 | 64.4 |  
| GSM8K (5-shot) | 50.2 | 52.6 | 76.9 | 70.9 | 68.7 |  
| Average | 71.4 |  71.9 | 79.2 | 78.4 | 77.7 |  

The following table summarizes the wikitext2 perplexity of pruning on smaller models -

| # Blocks | Meta-Llama-3-8B                  |                               |                  | Phi-3-mini-8k-instruct                      |                               |                  | Mistral-7B-Instruct-v0.3                  |                               |                  |
|----------|-------------------------|-------------------------------|------------------|---------------------------|-------------------------------|------------------|---------------------------|-------------------------------|------------------|
|          | #Param                  | Angular                       | Cosine           | #Param                    | Angular                       | Cosine           | #Param                    | Angular                       | Cosine           |
| Baseline | 8.0                        |  6.14                             |  -                |  3.8                         |  6.35                             |  -                |    7.2                       | 5.31                              |  -                |
| 2        |  7.6                       |  7.92                             |  7.92                |      3.6                     |  9.27                             |  7.60                |  6.8                         |  6.21                             |  7.68                |
| 4        |  7.2                       |   10.70                            |  10.70                |   3.4                        |  15.82                             |  22.83                |   6.4                        |  7.94                             |   13.67               |
| 6        |  6.7                       |   17.82                            |  17.82                |    3.1                       |  27.31                             |  45.86                |  5.9                         |  10.97                             |  16.17                |
| 8        |  6.3                       |   83.23                            | 83.23                 |   2.9                        |  45.86                             |  90.52                |   5.5                        |  23.93                             |  20.46                |

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

### Quantization Configuration (BnB Config)
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
- **`skip_blocks`**: Block indexed to skip while pruning.
  List: [0,1,2,3,-1,-2]
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

## License
This codebase is open-source and free to use for non-commercial purposes.

## Support
For issues and support, please file an issue in the repository issue tracker.

## Citation

If you use PruneGPT in your research, please cite the original research works!

```bibtext
@misc{gromov2024unreasonable,
      title={The Unreasonable Ineffectiveness of the Deeper Layers}, 
      author={Andrey Gromov and Kushal Tirumala and Hassan Shapourian and Paolo Glorioso and Daniel A. Roberts},
      year={2024},
      eprint={2403.17887},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
```bibtext
@misc{men2024shortgpt,
      title={ShortGPT: Layers in Large Language Models are More Redundant Than You Expect}, 
      author={Xin Men and Mingyu Xu and Qingyu Zhang and Bingning Wang and Hongyu Lin and Yaojie Lu and Xianpei Han and Weipeng Chen},
      year={2024},
      eprint={2403.03853},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```