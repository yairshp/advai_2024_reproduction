from dataclasses import dataclass
from pathlib import Path

import clip
import pyrallis
import torch
from PIL import Image
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
        queries = [preprocess(image).unsqueeze(0).to(device) for image in images]

        with torch.no_grad():

            # split prompt into first and second halves
            if " and " in prompt:
                prompt_parts = prompt.split(" and ")
            elif " with " in prompt:
                prompt_parts = prompt.split(" with ")
            else:
                print(
                    f"Unable to split prompt: {prompt}. "
                    f"Looking for 'and' or 'with' for splitting! Skipping!"
                )
                continue

            # extract texture features
            full_text_features = get_embedding_for_prompt(
                model, prompt, templates=imagenet_templates
            )
            first_half_features = get_embedding_for_prompt(
                model, prompt_parts[0], templates=imagenet_templates
            )
            second_half_features = get_embedding_for_prompt(
                model, prompt_parts[1], templates=imagenet_templates
            )

            # extract image features
            images_features = [model.encode_image(image) for image in queries]
            images_features = [
                feats / feats.norm(dim=-1, keepdim=True) for feats in images_features
            ]

            # compute similarities
            full_text_similarities = [
                (feat.float() @ full_text_features.T).item() for feat in images_features
            ]
            first_half_similarities = [
                (feat.float() @ first_half_features.T).item()
                for feat in images_features
            ]
            second_half_similarities = [
                (feat.float() @ second_half_features.T).item()
                for feat in images_features
            ]

            results_per_prompt[prompt] = {
                "full_text": full_text_similarities,
                "first_half": first_half_similarities,
                "second_half": second_half_similarities,
                "image_names": image_names,
            }

    with open("image_text_results.pkl", "wb") as f:
        pickle.dump(results_per_prompt, f)


if __name__ == "__main__":
    run()
