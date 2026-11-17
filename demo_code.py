"""Auto-labeling CIFAR-10 with foundation models, checked against CIFAR-10N human labels.

Three zero-shot labelers, all run locally with Hugging Face transformers:

  clip    openai/clip-vit-large-patch14       contrastive image-text matching
  siglip  google/siglip2-so400m-patch14-224   contrastive image-text matching
  qwen    Qwen/Qwen3-VL-4B-Instruct           generative VLM, asked for the category name; the label
                                              is read from the next-token distribution over the 10
                                              label words, so there is no free text to parse

Every labeler produces a score vector over the 10 classes for each image (cached under
<out-dir>/probs/). The script then writes one CSV row per image with the ground truth, the
CIFAR-10N human labels (train split), each model's label and confidence, the soft-vote
ensemble label, and a needs_review flag for images that should go to a human.

  python demo_code.py --limit 200                      # quick demo on a stratified subset
  python demo_code.py --split test --prompt simple     # dev split: prompt selection and calibration
  python demo_code.py --split train                    # all 50k images with human labels
"""

import argparse
import functools
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor

HERE = Path(__file__).resolve().parent
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

MODELS = {  # "prompt" is the default variant: the one with the best dev-split accuracy
    "clip": {"repo": "openai/clip-vit-large-patch14", "batch_size": 256, "prompt": "guided"},
    "siglip": {"repo": "google/siglip2-so400m-patch14-224", "batch_size": 128, "prompt": "guided"},
    "qwen": {"repo": "Qwen/Qwen3-VL-4B-Instruct", "batch_size": 32, "prompt": "simple"},
}

CIFAR10N_URL = "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/data/CIFAR-10_human.pt"
HUMAN_COLUMNS = {  # three independent MTurk labels per image (in random order) and their majority vote
    "random_label1": "human_label1",
    "random_label2": "human_label2",
    "random_label3": "human_label3",
    "aggre_label": "human_majority",
}

# ---------------------------------------------------------------- prompts

CLIP_CIFAR10_TEMPLATES = [  # the 18 CIFAR-10 templates released with CLIP
    "a photo of a {}.", "a blurry photo of a {}.", "a black and white photo of a {}.",
    "a low contrast photo of a {}.", "a high contrast photo of a {}.", "a bad photo of a {}.",
    "a good photo of a {}.", "a photo of a small {}.", "a photo of a big {}.",
    "a photo of the {}.", "a blurry photo of the {}.", "a black and white photo of the {}.",
    "a low contrast photo of the {}.", "a high contrast photo of the {}.", "a bad photo of the {}.",
    "a good photo of the {}.", "a photo of the small {}.", "a photo of the big {}.",
]
LOW_RES_TEMPLATES = ["a low resolution photo of a {}.", "a pixelated photo of a {}.", "a tiny thumbnail of a {}."]
# Follows the official class definitions: automobile = sedans, SUVs and the like; truck = big trucks only.
CLASS_SYNONYMS = {
    "airplane": ["airplane", "jet airliner", "propeller plane", "military jet"],
    "automobile": ["car", "sedan", "SUV", "sports car"],
    "bird": ["bird", "songbird", "ostrich"],
    "cat": ["cat", "kitten"],
    "deer": ["deer", "elk", "fawn"],
    "dog": ["dog", "puppy"],
    "frog": ["frog", "toad"],
    "horse": ["horse", "pony", "foal"],
    "ship": ["ship", "boat", "cargo ship", "sailboat"],
    "truck": ["truck", "lorry", "fire truck", "semi truck", "dump truck"],
}

QWEN_PROMPTS = {
    "simple": "What is the main object in this image? Answer with one word from this list: "
              + ", ".join(CLASSES) + ".",
    "guided": (
        "This is a 32x32 photo from the CIFAR-10 dataset, shown upscaled, so it is blurry.\n"
        "Classify the main object into exactly one of these categories: " + ", ".join(CLASSES) + ".\n"
        "Definitions: automobile = sedans, SUVs and similar cars; truck = big trucks only "
        "(lorries, fire trucks, semis); frog includes toads.\n"
        "Answer with the category name only, in lowercase."
    ),
}


def contrastive_prompts(variant):
    """One list of prompts per class; the class embedding is the mean of their text embeddings."""
    if variant == "simple":
        return [[f"a photo of a {c}."] for c in CLASSES]
    if variant == "templates":
        return [[t.format(c) for t in CLIP_CIFAR10_TEMPLATES] for c in CLASSES]
    if variant == "guided":
        templates = CLIP_CIFAR10_TEMPLATES + LOW_RES_TEMPLATES
        return [[t.format(n) for n in CLASS_SYNONYMS[c] for t in templates] for c in CLASSES]
    raise ValueError(f"unknown prompt variant for contrastive models: {variant}")


# ---------------------------------------------------------------- data


def load_split(data_dir, split):
    """Images (N, 32, 32, 3) uint8, ground-truth labels (N,), and CIFAR-10N human labels (train only)."""
    ds = torchvision.datasets.CIFAR10(data_dir, train=split == "train", download=True)
    images, labels = ds.data, np.asarray(ds.targets)
    human = {}
    if split == "train":
        human = load_cifar10n(Path(data_dir) / "CIFAR-10_human.pt")
        if not np.array_equal(human["clean_label"], labels):
            raise RuntimeError("CIFAR-10N labels are not aligned with the torchvision CIFAR-10 train split")
    return images, labels, human


def load_cifar10n(path):
    if not path.exists():
        torch.hub.download_url_to_file(CIFAR10N_URL, str(path))
    try:
        from numpy._core.multiarray import _reconstruct
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import _reconstruct
    # The file is a pickled dict of numpy arrays; allow-list exactly those types instead of
    # falling back to weights_only=False, which would execute arbitrary pickle code.
    allowed = [(_reconstruct, "numpy.core.multiarray._reconstruct"), np.ndarray, np.dtype, type(np.dtype(np.int64))]
    with torch.serialization.safe_globals(allowed):
        raw = torch.load(path)
    return {k: np.asarray(v) for k, v in raw.items()}


def stratified_subset(labels, limit, seed):
    rng = np.random.default_rng(seed)
    per_class = max(1, limit // len(CLASSES))
    picks = [rng.choice(np.flatnonzero(labels == c), per_class, replace=False) for c in range(len(CLASSES))]
    return np.sort(np.concatenate(picks))


# ---------------------------------------------------------------- labelers


def _pooled(out):
    """get_*_features returns a tensor in transformers 4.x and a ModelOutput in 5.x."""
    return out if torch.is_tensor(out) else out.pooler_output


class ContrastiveLabeler:
    """Zero-shot classification by cosine similarity between image and class-prompt embeddings."""

    def __init__(self, repo, device):
        self.device = device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo, dtype=dtype).to(device).eval()
        self.is_siglip = "siglip" in self.model.config.model_type

    @torch.inference_mode()
    def encode_text(self, prompts):
        if self.is_siglip:
            # SigLIP was trained on lowercased text padded to 64 tokens.
            prompts = [p.lower() for p in prompts]
            kwargs = {"padding": "max_length", "max_length": 64, "truncation": True}
        else:
            kwargs = {"padding": True}
        tokens = self.processor(text=prompts, return_tensors="pt", **kwargs).to(self.device)
        emb = _pooled(self.model.get_text_features(**tokens)).float()
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def __call__(self, images, prompt_variant, batch_size, desc):
        class_emb = torch.stack([self.encode_text(p).mean(0) for p in contrastive_prompts(prompt_variant)])
        class_emb = class_emb / class_emb.norm(dim=-1, keepdim=True)
        scale = self.model.logit_scale.exp().float()
        scores = []
        for i in tqdm(range(0, len(images), batch_size), desc=desc):
            batch = [Image.fromarray(a) for a in images[i:i + batch_size]]
            pixels = self.processor(images=batch, return_tensors="pt")["pixel_values"]
            emb = _pooled(self.model.get_image_features(pixel_values=pixels.to(self.device, self.model.dtype))).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            scores.append((scale * emb @ class_emb.T).cpu())
        return torch.cat(scores).numpy()


class QwenVLLabeler:
    """Generative VLM used as a classifier.

    The prompt asks for the category name. Instead of sampling text and parsing it, we run one
    forward pass and read the log-probability of each label word's first token at the answer
    position (the 10 first tokens are distinct). This is constrained decoding: the output is
    always a valid label, and the probabilities double as a confidence score.
    """

    def __init__(self, repo, device, image_size=224):
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(repo)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(repo, dtype=dtype, device_map=device).eval()
        tok = self.processor.tokenizer
        self.label_token_ids = [tok.encode(c, add_special_tokens=False)[0] for c in CLASSES]
        assert len(set(self.label_token_ids)) == len(CLASSES), "label words must start with distinct tokens"
        self.image_size = image_size

    @torch.inference_mode()
    def __call__(self, images, prompt_variant, batch_size, desc):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QWEN_PROMPTS[prompt_variant]}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Upscale 32x32 -> 224x224 ourselves and pin the processor to exactly that many pixels;
        # by default it would resize to its own minimum (256x256 for Qwen3-VL).
        side = self.image_size
        size = {"shortest_edge": side * side, "longest_edge": side * side}
        scores = []
        for i in tqdm(range(0, len(images), batch_size), desc=desc):
            batch = [Image.fromarray(a).resize((side, side), Image.BICUBIC) for a in images[i:i + batch_size]]
            inputs = self.processor(text=[text] * len(batch), images=batch, padding=True, return_tensors="pt",
                                    images_kwargs={"size": size}).to(self.model.device)
            logits = self.model(**inputs, logits_to_keep=1).logits[:, -1].float()
            scores.append(logits.log_softmax(-1)[:, self.label_token_ids].cpu())
        return torch.cat(scores).numpy()


def run_labeler(name, variant, images, device, batch_size):
    """Load one model, score all images, and return (scores, timing info)."""
    repo = MODELS[name]["repo"]
    labeler = QwenVLLabeler(repo, device) if name == "qwen" else ContrastiveLabeler(repo, device)
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    scores = labeler(images, variant, batch_size, desc=f"{name}/{variant}")
    if cuda:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    del labeler
    if cuda:
        torch.cuda.empty_cache()
    return scores, {
        "model": repo, "prompt": variant, "images": len(images), "batch_size": batch_size,
        "seconds": round(seconds, 1), "images_per_second": round(len(images) / seconds, 1),
        "device": torch.cuda.get_device_name() if cuda else platform.processor(),
        "torch": torch.__version__, "transformers": transformers.__version__,
    }


# ---------------------------------------------------------------- outputs


def softmax(scores):
    """Contrastive scores are scaled cosine similarities and Qwen scores are token log-probs;
    a softmax over the 10 classes turns either into a class distribution."""
    e = np.exp(scores - scores.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(scores, labels):
    """Temperature scaling: the T minimizing the dev-set NLL of softmax(scores / T).

    Raw confidences are not comparable across models (CLIP is under-confident, Qwen is close to
    one-hot), so they are calibrated before averaging them into a vote or thresholding them.
    """
    grid = np.exp(np.linspace(np.log(0.02), np.log(50), 400))
    rows = np.arange(len(labels))
    nll = [-np.log(softmax(scores / t)[rows, labels] + 1e-12).mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


@functools.cache
def dev_labels(data_dir):
    return np.asarray(torchvision.datasets.CIFAR10(data_dir, train=False, download=True).targets)


def dev_temperature(out_dir, data_dir, name, variant):
    """Temperature fit on the cached full dev-split scores of this model/prompt, or None if absent."""
    dev_cache = out_dir / "probs" / f"test_{name}_{variant}.npy"
    if not dev_cache.exists():
        return None
    return fit_temperature(np.load(dev_cache), dev_labels(data_dir))


def build_table(idx, labels, human, probs, review_threshold):
    names = np.array(CLASSES)
    df = pd.DataFrame({"index": idx, "true_label": names[labels[idx]]})
    for src, col in HUMAN_COLUMNS.items():
        if src in human:
            df[col] = names[human[src][idx]]
    for model, p in probs.items():
        df[f"{model}_label"] = names[p.argmax(1)]
        df[f"{model}_conf"] = p.max(1).round(4)
    mean = np.mean(list(probs.values()), axis=0)
    vote = mean.argmax(1)
    df["vote_label"] = names[vote]
    df["vote_conf"] = mean.max(1).round(4)
    df["n_agree"] = np.sum([p.argmax(1) == vote for p in probs.values()], axis=0)
    df["needs_review"] = (df["n_agree"] < len(probs)) | (df["vote_conf"] < review_threshold)
    return df


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--split", choices=["train", "test"], default="train",
                   help="train = 50k images with CIFAR-10N human labels; test = 10k dev images")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--prompt", nargs="+", default=[],
                   help="one variant for every model (simple | templates | guided), or model=variant pairs; "
                        "'templates' only exists for clip/siglip (default: best dev variant per model)")
    p.add_argument("--limit", type=int, help="label a stratified random subset of this many images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, help="override the per-model default batch size")
    p.add_argument("--review-threshold", type=float, default=0.9,
                   help="route an image to human review if the models disagree or the vote confidence is below this")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-dir", default=str(HERE / "data"))
    p.add_argument("--out-dir", default=str(HERE / "outputs"))
    p.add_argument("--overwrite", action="store_true", help="recompute model scores even if cached")
    args = p.parse_args()

    prompts = {m: MODELS[m]["prompt"] for m in args.models}
    for item in args.prompt:
        model, _, variant = item.rpartition("=")
        if model and model not in prompts:
            p.error(f"--prompt {item}: '{model}' is not one of the selected models")
        for m in [model] if model else prompts:
            prompts[m] = variant
    for m, variant in prompts.items():
        if variant not in (QWEN_PROMPTS if m == "qwen" else ("simple", "templates", "guided")):
            p.error(f"prompt variant '{variant}' is not defined for {m}")
    args.prompts = prompts
    return args


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "probs").mkdir(parents=True, exist_ok=True)

    images, labels, human = load_split(args.data_dir, args.split)
    idx = np.arange(len(labels)) if args.limit is None else stratified_subset(labels, args.limit, args.seed)
    tag = args.split if args.limit is None else f"{args.split}_n{len(idx)}"
    print(f"{tag}: {len(idx)} images, models={args.prompts}, device={args.device}")

    info_path = out_dir / "run_info.json"
    probs, temperatures = {}, {}
    for name in args.models:
        variant = args.prompts[name]
        cache = out_dir / "probs" / f"{tag}_{name}_{variant}.npy"
        if cache.exists() and not args.overwrite:
            scores = np.load(cache)
        else:
            batch_size = args.batch_size or MODELS[name]["batch_size"]
            scores, info = run_labeler(name, variant, images[idx], args.device, batch_size)
            np.save(cache, scores)
            run_info = json.loads(info_path.read_text()) if info_path.exists() else {}
            run_info[cache.stem] = info
            info_path.write_text(json.dumps(run_info, indent=2))

        temperatures[name] = dev_temperature(out_dir, args.data_dir, name, variant)
        probs[name] = softmax(scores / (temperatures[name] or 1.0))
        acc = (probs[name].argmax(1) == labels[idx]).mean()
        calib = f"T={temperatures[name]:.3f}" if temperatures[name] else "uncalibrated (run --split test first)"
        print(f"  {name:<7} prompt={variant:<9} accuracy vs CIFAR-10 ground truth = {acc:.4f}  [{calib}]")

    df = build_table(idx, labels, human, probs, args.review_threshold)
    csv_path = out_dir / f"labels_{tag}.csv"
    df.to_csv(csv_path, index=False)
    config = {"split": args.split, "images": len(idx), "prompts": args.prompts,
              "models": {m: MODELS[m]["repo"] for m in args.models},
              "temperatures": temperatures, "review_threshold": args.review_threshold}
    csv_path.with_suffix(".json").write_text(json.dumps(config, indent=2))
    print(f"  vote    accuracy = {(df.vote_label == df.true_label).mean():.4f}, "
          f"needs_review = {df.needs_review.mean():.1%}")
    if "human_majority" in df:
        for col in ["human_label1", "human_majority"]:
            print(f"  {col:<14} accuracy = {(df[col] == df.true_label).mean():.4f}")
    print(f"wrote {csv_path}")
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
