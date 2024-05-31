import os
import json
config_path = '/home/azureuser/arnav/PruneGPT/configs/config.json'
for model in ['meta-llama/Meta-Llama-3-70B']:
    for blocks in [24]:
        for method in ['angular_distance', 'cosine_similarity']:
            with open(config_path, 'r') as file:
                config = json.load(file)
            config['model_name'] = model
            config['pruning_method'] = method
            if method == 'angular_distance':
                token = 'last'
            else:
                token = 'all'
            config['pruning_token'] = token
            config['num_blocks_to_prune'] = blocks
            config['save_directory'] = f'models/llama3_70b_{80-blocks}blocks_{method}'
            with open(config_path, 'w') as f:
                json.dump(config, f)
            os.system("python /home/azureuser/arnav/PruneGPT/src/main.py")
