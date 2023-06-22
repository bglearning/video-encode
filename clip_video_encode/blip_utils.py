import torch
from lavis.models import load_model_and_preprocess


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
blip_model, blip_vis_processors, blip_text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device)