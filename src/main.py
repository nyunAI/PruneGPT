import torch
from datasets import load_dataset
from model_manager import ModelManager
from pruner import ModelPruner
from utils import Utils, ConfigManager
from transformers import BitsAndBytesConfig

def main(path):

    config_manager = ConfigManager(path)
    model_name = config_manager.get("model_name")
    quantize = config_manager.get("quantize")
    dataset_config = config_manager.get("dataset")
    num_blocks_to_prune = config_manager.get("num_blocks_to_prune")
    save_directory = config_manager.get("save_directory")
    log_file = config_manager.get("log_file")
    pruning_method = config_manager.get("pruning_method")
    pruning_token = config_manager.get("token")
    bnb_config = BitsAndBytesConfig(**config_manager.get('bnb_config'))

    Utils.setup_logging(log_file)
    Utils.log_info("Started model pruning process.")

    model_manager = ModelManager(model_name=model_name)
    model_manager.load_model_and_tokenizer(quantize=quantize, bnb_config=bnb_config)

    dataset = load_dataset(dataset_config["name"], dataset_config["subset"], split=dataset_config["split"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset_config["batch_size"], shuffle=True)

    pruner = ModelPruner(model=model_manager.get_model())
    bi_scores = pruner.calculate_bi(dataloader, model_manager.get_tokenizer(), pruning_method, pruning_token)
    Utils.log_info(f"Pruning {num_blocks_to_prune} blocks from the model.")
    model_manager.model = pruner.prune_model_blocks(bi_scores, num_blocks_to_prune)

    # Save the pruned model
    Utils.ensure_directory_exists(save_directory)
    Utils.log_info(f"Saving pruned model to {save_directory}.")
    model_manager.export_model(save_directory)

if __name__ == "__main__":
    main("../configs/config.json")