from model.AttentionMotionVae import *

if __name__ == "__main__":
    dummy = torch.randn(1, 14, 26).to('cuda') 
    model = get_attention_vae_model().to('cuda')
    out, mean, log_var = model(dummy, 14)
    print(out) 
