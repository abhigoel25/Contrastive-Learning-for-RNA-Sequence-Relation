import torch

# Load the checkpoint
checkpoint_path = "checkpoints/introns_cl/NTv2/199/best-checkpoint.ckpt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Extract full model state dict
model_state_dict = checkpoint["state_dict"]

# Print all layer names to see what’s inside
print("Layers in checkpoint:", model_state_dict.keys())

# Extract encoder weights (assumes encoder is named `model`)
encoder_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items() if k.startswith("model.")}

# Extract projection head weights (assuming it's named `projection_head`)
projection_head_state_dict = {k.replace("model.projection_head.", ""): v for k, v in model_state_dict.items() if k.startswith("model.projection_head.")}

# Save them separately
torch.save(encoder_state_dict, "contrastive_encoder.pth")
torch.save(projection_head_state_dict, "projection_head.pth")

print("✅ Saved encoder to contrastive_encoder.pth")
print("✅ Saved projection head to projection_head.pth")
