import torch
from diffusers import StableDiffusionImg2ImgPipeline
import os

model_path = "runwayml/stable-diffusion-v1-5"
inpainting_model_path = "runwayml/stable-diffusion-inpainting"


def download_model():
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=HF_AUTH_TOKEN
    )


if __name__ == "__main__":
    download_model()
