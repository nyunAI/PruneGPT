import torch
from datasets import load_dataset
from model_manager import ModelManager
from pruner import ModelPruner
from utils import LoggingManager, ConfigManager
from eval_utils import PPLMetric
from transformers import BitsAndBytesConfig
import argparse

def main(path):
    """
    Main function to perform model pruning.

    Parameters:
    path (str): Path to the configuration file.

    Returns:
    None
    """

    # Load configuration settings
    config_manager = ConfigManager(path)
    model_name = config_manager.get("model_name")
    quantize = config_manager.get("quantize")
    dataset_config = config_manager.get("dataset")
    num_blocks_to_prune = config_manager.get("num_blocks_to_prune")
    skip_blocks = config_manager.get("skip_blocks")
    save_directory = config_manager.get("save_directory")
    calculate_ppl = config_manager.get("calculate_ppl")
    log_file = config_manager.get("log_file")
    pruning_method = config_manager.get("pruning_method")
    pruning_token = config_manager.get("pruning_token")
    bnb_config = BitsAndBytesConfig(**config_manager.get('bnb_config'))

    # Setup logging
    LoggingManager.setup_logging(log_file)
    LoggingManager.log_info("Started model pruning process.")

    # Initialize model manager and load model and tokenizer
    model_manager = ModelManager(model_name=model_name)
    model_manager.load_model_and_tokenizer(quantize=quantize, bnb_config=bnb_config)

    # Load dataset and create dataloader
    dataset = load_dataset(dataset_config["name"], dataset_config["subset"], split=dataset_config["split"]).select(list(range(100))) # 100 samples for pruning metric computation
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset_config["batch_size"], shuffle=True)

    # Initialize model pruner and calculate block importance scores
    pruner = ModelPruner(model=model_manager.get_model())
    bi_scores = pruner.calculate_bi(dataloader, model_manager.get_tokenizer(), pruning_method, pruning_token)
    LoggingManager.log_info(f"Pruning {num_blocks_to_prune} blocks using {pruning_method} and {pruning_token} tokens.")

    # Prune model based on block importance scores
    model_manager.model = pruner.prune_model_blocks(bi_scores, num_blocks_to_prune, skip_blocks)
    num_params = model_manager.model.num_parameters()
    LoggingManager.log_info(f"Number of parameters post pruning: {num_params}")

    # Calculate perplexity on Wikitext2 dataset
    if calculate_ppl:
        ppl = PPLMetric(model_manager.model, model_manager.get_tokenizer(), ['wikitext2'])['wikitext2']
        LoggingManager.log_info(f"Wikitext2 perplexity:{ppl}")

    # Save the pruned model
    if save_directory is not None:
        LoggingManager.ensure_directory_exists(save_directory)
        LoggingManager.log_info(f"Saving pruned model to {save_directory}.")
        model_manager.export_model(save_directory)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_path", type=str, default="configs/config.json")
    args = argparser.parse_args()
    main(args.config_path)