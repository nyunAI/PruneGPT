import torch
from transformers import PreTrainedModel
import copy, math
from tqdm import tqdm
import torch.nn.functional as F

class ModelPruner:
    def __init__(self, model: PreTrainedModel):
        self.model = model

    def prune_model_blocks(self, importance_scores: list, num_blocks_to_prune: int, skip_blocks: list = None) -> PreTrainedModel:
        """
        Prunes blocks from the transformer model based on the importance scores.

        Parameters:
        - importance_scores (list): List of importance scores for each block.
        - num_blocks_to_prune (int): Number of blocks to prune from the model.
        - skip_blocks (list, optional): List of block indices to skip. Defaults to None.

        Returns:
        - PreTrainedModel: The pruned transformer model.
        """

        # Assign max score to skip blocks
        if skip_blocks:
            for block in skip_blocks:
                importance_scores[block] = max(importance_scores)

        # Sort blocks by importance score
        sorted_blocks = sorted(range(len(importance_scores)), key=lambda i: importance_scores[i])

        # Identify blocks to prune
        blocks_to_prune = sorted_blocks[:num_blocks_to_prune]

        # Create a new model without the pruned blocks
        pruned_model = copy.deepcopy(self.model)
        # pruned_model.load_state_dict(self.model.state_dict())

        # Prune the blocks
        layers = []
        for i, layer in enumerate(self.model.model.layers):
            if i in blocks_to_prune:
                continue
            layer = self.model.model.layers[i]
            layer.self_attn.layer_idx = len(layers)
            layers.append(layer)
        
        pruned_model.model.layers = torch.nn.ModuleList(layers)
        pruned_model.config.num_hidden_layers = len(self.model.model.layers) - len(blocks_to_prune)

        return pruned_model

    def calculate_bi(self, dataloader, tokenizer, pruning_method="angular_distance", pruning_token='last'):
        """
        Calculate Block Influence (BI) scores for each layer.

        Parameters:
        - dataloader (DataLoader): DataLoader for the dataset.
        - tokenizer (Tokenizer): Tokenizer for the model inputs.
        - pruning_method (str, optional): Pruning method to use. One of "angular_distance", "cosine_similarity.
        - pruning_token (str, optional): Pruning token to use. One of "all", "last".

        Returns:
        - list: List of BI scores for each block.
        """
        scores = []
        num_batches = 0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.model.device)
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                if not scores:
                    scores = [0] * (len(hidden_states) - 1)

                for i in range(1, len(hidden_states)):
                    input_hidden_state = hidden_states[i-1]
                    output_hidden_state = hidden_states[i]
                    if pruning_token == 'last':
                        input_hidden_state = input_hidden_state[:,-1,:]
                        output_hidden_state = output_hidden_state[:,-1,:]
                    sim = F.cosine_similarity(input_hidden_state, output_hidden_state)
                    if pruning_method == 'angular_distance':
                        sim = torch.clamp(sim, -1.0, 1.0)
                        sim = (1 / math.pi) * torch.acos(sim)
                    elif pruning_method == 'cosine_similarity':
                        sim = 1 - sim
                    scores[i-1] += sim.mean().item()

                num_batches += 1

        scores = [score / num_batches for score in scores]  # Average scores over all batches
        return scores