# Auto-labeling CIFAR-10 with foundation models (vs. CIFAR-10N human labels)

Zero-shot labeling of the 50,000 CIFAR-10 training images with three locally run foundation
models, compared against three independent crowd-worker labels per image from
[CIFAR-10N](https://github.com/UCSC-REAL/cifar-10-100n). The write-up is in `report.pdf`
(source: `report.md`).

| Model | Type | How the label is produced | Prompt used |
|---|---|---|---|
| `openai/clip-vit-large-patch14` | contrastive | cosine similarity to class-prompt embeddings | `guided` |
| `google/siglip2-so400m-patch14-224` | contrastive | cosine similarity to class-prompt embeddings | `guided` |
| `Qwen/Qwen3-VL-4B-Instruct` | generative VLM | asked for the category name; probabilities of the 10 label words at the first answer token | `simple` |

Results on the 50k training images (accuracy against the CIFAR-10 labels):

| Labeler | Accuracy |
|---|---|
| One crowd label / majority of 3 crowd labels | 82.3% / 91.0% |
| CLIP / SigLIP2 / Qwen3-VL | 96.5% / 98.1% / 97.1% |
| Soft vote of the 3 models | 98.6% |
| Vote on the 90.1% of images not flagged `needs_review` | 99.89% |

## Files

```
demo_code.py        read data -> call models -> write per-image labels (the labeling pipeline)
analyze.py          compare against ground truth and human labels, write metrics and figures
report.md / .pdf    2-page report
outputs/
  labels_train.csv         one row per training image: ground truth, human_label1..3 and human_majority
                           (CIFAR-10N), each model's label + calibrated confidence, vote_label/vote_conf,
                           n_agree (models agreeing with the vote) and needs_review
  labels_train.json        models, prompts, temperatures and review threshold used for that CSV
  labels_test.csv / .json  same for the 10k dev split (no human labels)
  examples.csv             the 12 samples shown in figures/examples.png
  suspected_label_errors.csv  29 images where the 3 models and the crowd majority agree (vote
                           confidence >= 0.9) on a label that differs from CIFAR-10
  metrics.json             every number in the report
  run_info.json            runtime and throughput of each model run
  probs/*.npy              raw per-class scores of every model / prompt / split, so analyze.py
                           can be rerun without a GPU
  figures/                 examples, coverage (error rate vs. share auto-labeled), confusions,
                           suspected_label_errors
```

## Run

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Quick demo: 200 stratified training images, all three models (under a minute on an RTX 4060 Ti
# once the ~14 GB of model weights are downloaded; on CPU use --models clip).
python demo_code.py --limit 200

# Full pipeline used for the report.
# 1) Dev split (CIFAR-10 test): prompt ablation and temperature calibration.
python demo_code.py --split test --models clip siglip --prompt simple
python demo_code.py --split test --models clip siglip --prompt templates
python demo_code.py --split test --prompt guided
python demo_code.py --split test        # default prompts = best dev variant per model
# 2) Train split: all 50k images, with the CIFAR-10N human labels.
python demo_code.py --split train
python analyze.py
```

The data (CIFAR-10 from the official site via torchvision; CIFAR-10N labels and side information
from GitHub) is downloaded into `data/` on first use, and models come from the Hugging Face Hub.
Model scores are cached in `outputs/probs/`, so rerunning a command only recomputes what is missing
(`--overwrite` forces a rerun). On an RTX 4060 Ti 16 GB the train split takes 5.4 min (CLIP),
6.3 min (SigLIP2) and 28.6 min (Qwen3-VL).

### Main options of `demo_code.py`

| Option | Meaning |
|---|---|
| `--split train\|test` | `train`: 50k images with human labels; `test`: 10k images used as the dev split |
| `--models clip siglip qwen` | which labelers to run (the CSV gets a label/confidence column pair for each, plus the vote) |
| `--prompt` | one variant for all models (`simple`, `templates`, `guided`) or per model, e.g. `clip=templates qwen=guided`; default `clip=guided siglip=guided qwen=simple` |
| `--limit N` | label a stratified random subset of N images |
| `--review-threshold 0.9` | `needs_review` is set when the models disagree or the vote confidence is below this |

Confidences are calibrated with a per-model temperature fit on the dev split (the cached
`outputs/probs/test_<model>_<prompt>.npy`). Run `--split test` with the same prompt before
labeling the train split; otherwise the raw, uncalibrated confidences are used and a note is printed.
