from dataclasses import dataclass
from pathlib import Path

import clip
import pyrallis
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from tqdm import tqdm

import pickle

from imagenet_utils import get_embedding_for_prompt, imagenet_templates


@dataclass
class EvalConfig:
    output_path: Path = Path("./outputs/")
    metrics_save_path: Path = Path("./metrics/")

    def __post_init__(self):
        self.metrics_save_path.mkdir(parents=True, exist_ok=True)


@pyrallis.wrap()
def run(config: EvalConfig):
    print("Loading CLIP model...")
    device = torch.device(
        "cuda"
        if (torch.cuda.is_available() and torch.cuda.device_count() > 0)
        else "cpu"
    )
    model, preprocess = clip.load("ViT-B/16", device)
    model.eval()
    print("Done.")

    print("Loading BLIP model...")
    blip_model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device
    )
    print("Done.")

    prompts = [p.name for p in config.output_path.glob("*") if p.is_dir()]
    print(f"Running on {len(prompts)} prompts...")

    results_per_prompt = {}
    for prompt in tqdm(prompts):
        print(f'Running on: "{prompt}"')

        # get all images for the given prompt
        image_paths = [
            p
            for p in (config.output_path / prompt).rglob("*")
            if p.suffix in [".png", ".jpg"]
        ]
        images = [Image.open(p) for p in image_paths]
        image_names = [p.name for p in image_paths]

        with torch.no_grad():
            # extract prompt embeddings
            prompt_features = get_embedding_for_prompt(
                model, prompt, templates=imagenet_templates
            )

            # extract blip captions and embeddings
            blip_input_images = [
                vis_processors["eval"](image).unsqueeze(0).to(device)
                for image in images
            ]
            blip_captions = [
                blip_model.generate({"image": image})[0] for image in blip_input_images
            ]
            texts = [clip.tokenize([text]).cuda() for text in blip_captions]
            caption_embeddings = [model.encode_text(t) for t in texts]
            caption_embeddings = [
                embedding / embedding.norm(dim=-1, keepdim=True)
                for embedding in caption_embeddings
            ]

            text_similarities = [
                (caption_embedding.float() @ prompt_features.T).item()
                for caption_embedding in caption_embeddings
            ]

            results_per_prompt[prompt] = {
                "text_similarities": text_similarities,
                "captions": blip_captions,
                "image_names": image_names,
            }

    with open("text_text_results.pkl", "wb") as f:
        pickle.dump(results_per_prompt, f)


if __name__ == "__main__":
    run()
