from model.Transformer        import * 
from model.Lstm               import * 
import torch 


model = get_transformer_model(model_dim = 256, num_layers = 4)
trainable_params = [p for p in model.parameters() if p.requires_grad]

total_params = sum(p.numel() for p in trainable_params)
print(f"Total trainable parameters: {total_params:,}")

