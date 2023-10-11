from enum import Enum
from typing import Any, List, Optional

import numpy as np
import open_clip
import torch

from clip import clip

from lavis.models import load_model_and_preprocess
from torchvision.transforms import ToPILImage


class VideoEncoder:
    """Encoder for video frames (and captions)

    Alternative of simplemapper.FrameMapper
    (which is to be deprecated)
    """
    @property
    def preprocessor(self):
        ...

    def __call__(self, frames: torch.tensor, *args: Any, **kwds: Any) -> np.array:
        ...

    def encode_captions(self, captions: List[str]) -> np.array:
        ...

    def generate_captions(self, frames: torch.tensor) -> List[str]:
        raise NotImplementedError("Caption Generation is not supported right now")


class OpenClipEncoder(VideoEncoder):
    """VideoEncoder that uses open_clip
    """
    def __init__(self, model_name="ViT-B-32", device="cpu", pretrained="laion2b_s34b_b79k") -> None:
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.preprocess.transforms = [ToPILImage()] + self.preprocess.transforms[-3:]

        self.model.eval()

    @property
    def preprocessor(self):
        return self.preprocess

    def __call__(self, frames: torch.tensor, *args: Any, **kwds: Any) -> np.array:
        return self.model.encode_image(frames).cpu().detach().numpy()

    def encode_captions(self, captions: List[str]) -> np.array:
        tokens = self.tokenizer(captions).to(self.device)
        return self.model.encode_text(tokens).cpu().detach().numpy()


class OpenAIClipEncoder(VideoEncoder):
    """VideoEncoder that uses OpenAI CLIP
    """
    def __init__(self, model_name="ViT-B/32", device="cpu") -> None:
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.preprocess.transforms = [ToPILImage()] + self.preprocess.transforms
        self.model.eval()

    @property
    def preprocessor(self):
        return self.preprocess

    def __call__(self, frames: torch.tensor, *args: Any, **kwds: Any) -> np.array:
        return self.model.encode_image(frames).cpu().detach().numpy()

    def encode_captions(self, captions: List[str]) -> np.array:
        text = clip.tokenize(captions).to(self.device)
        return self.model.encode_text(text).cpu().detach().numpy()


class BLIP2Encoder(VideoEncoder):
    """VideoEncoder that uses BLIP2
    """

    class OutputType(Enum):
        EMBED = 'embed'
        EMBED_PROJ = 'embed_proj'
        EMBED_BOTH = 'embed_both'

    def __init__(self, model_name="blip2_feature_extractor", device="cpu", model_type="pretrain", output_type: OutputType = OutputType.EMBED) -> None:
        super().__init__()
        model, vis_preprocessors, text_preprocessors = load_model_and_preprocess(model_name,
                                                                                 model_type=model_type,
                                                                                 device=device)
        self.model = model
        self.preprocess = vis_preprocessors['eval']
        self.preprocess.transform.transforms = [ToPILImage()] + self.preprocess.transform.transforms[-3:]
        self.text_preprocess = text_preprocessors['eval']
        self.output_type = output_type

        self.model.eval()
                                                                                
    @property
    def preprocessor(self):
        return self.preprocess

    def __call__(self, frames: torch.tensor, *args: Any, **kwds: Any) -> np.array:
        image_output = self.model.extract_features({'image': frames}, mode="image")
        
        if self.output_type == BLIP2Encoder.OutputType.EMBED:
            # (n_frames, 32, 768)
            return image_output.image_embeds.cpu().detach().numpy()
        elif self.output_type == BLIP2Encoder.OutputType.EMBED_PROJ:
            # (n_frames, 32, 256)
            return image_output.image_embeds_proj.cpu().detach().numpy()
        else:
            feat = image_output.image_embeds.cpu().detach().numpy()
            proj = image_output.image_embeds_proj.cpu().detach().numpy()
            # (n_frames, 32, 768 + 256)
            return np.concatenate([feat, proj], axis=-1)

    def encode_captions(self, captions: List[str]) -> np.array:
        text_inputs = []
        for caption in captions:
            text_inputs.append(self.text_preprocess(caption))
        text_output = self.model.extract_features({'text_input': text_inputs}, mode="text")

        if self.output_type == BLIP2Encoder.OutputType.EMBED:
            return text_output.text_embeds.cpu().detach().numpy()
        else:
            return text_output.text_embeds_proj.cpu().detach().numpy()


def create_encoder(encoder_type: str="open_clip", device: Optional[str]=None) -> VideoEncoder:

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    if encoder_type == 'open_clip':
        return OpenClipEncoder(device=device)
    elif encoder_type == 'openai_clip':
        return OpenAIClipEncoder(device=device)
    elif encoder_type == 'blip2':
        return BLIP2Encoder(device=device)
    elif encoder_type == 'blip2_proj':
        return BLIP2Encoder(device=device, output_type=BLIP2Encoder.OutputType.EMBED_PROJ)
    elif encoder_type == 'blip2_both':
        return BLIP2Encoder(device=device, output_type=BLIP2Encoder.OutputType.EMBED_BOTH)
    else:
        raise ValueError(f'{encoder_type=} not recognized.'
                         'Must be one of `open_clip`, `openai_clip`, `blip2`, `blip2_proj`')