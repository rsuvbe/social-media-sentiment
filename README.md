# Entity-Level Twitter Sentiment — CAIR SDU Selection Competition

**6th place on the private leaderboard** (accuracy 0.76064), up from 12th on the public one.

The task: given a tweet and an entity (a game, brand or company), predict the tweet's sentiment *towards that entity*:
`Positive`, `Negative`, `Neutral` or `Irrelevant`. Metric: accuracy. The competition was the selection round for the
CAIR lab at SDU, run on Kaggle with strict rules: competition data only, no zero-shot LLMs or APIs, no AutoML, and the
notebook must reproduce the submission exactly.

> The dataset is private and is **not** in this repository. Notebook outputs that showed tweets or test IDs were removed;
> metrics, tables and plots are kept. See [`data/README.md`](data/README.md).

## Highlights

- **Found a validation leak and built the whole pipeline around it.** Every tweet appears 6 times in the training set
  (the original + 5 augmented paraphrases with the same label), and test tweets are never in train. With a random K-fold,
  near-copies land on both sides of the split: a 1-nearest-neighbour model scores **96.5%**. With `GroupKFold` by tweet
  ID it scores **40%**, which matches the real train → test situation. All model selection uses grouped CV, scored on
  original tweets only (the test set has no augmented rows).
- **Trusted CV over the public leaderboard.** The public LB had only ~320 rows (±2.6 pt noise). Ensembles were chosen on
  out-of-fold accuracy; on the private LB the ranking of my submissions matched the CV ranking, and the place went from
  12th to 6th.
- **Fine-tuned a 7B LLM as a classifier.** `Qwen2.5-7B` (base, not instruct) with hand-written **QLoRA**: 4-bit frozen
  weights, LoRA adapters r=16, and a classification head on the last token. Lesson learned the hard way: a randomly
  initialised head on Qwen's large-norm hidden states starts at loss 4.4, so the head is **zero-initialised**. The LLM's
  errors differ from the encoders', and blending added **+1.2 pt** (split-half estimate, ±0.5).
- **Five encoders fine-tuned with a hand-written PyTorch loop:** Twitter-RoBERTa base/large, BERTweet base/large,
  DeBERTa-v3-large — label smoothing, AdamW, warm-up + linear decay, partially frozen lower layers, fp16.
- **Error analysis.** The ensemble is well calibrated (top confidence decile 98% correct, bottom decile 39%). A manual
  audit of 45 errors: ~40% label noise (e.g. 172 empty tweets with near-random labels), ~38% genuinely ambiguous,
  ~22% real model errors (sarcasm, wrong target entity, world knowledge).

## Results

| Submission | Validation | Public LB | Private LB |
|---|---|---|---|
| v1 — 2 Twitter encoders (base) + TF-IDF, CPU | 0.6965 (fold 0) | 0.69905 | 0.73127 |
| v2 — 4 Twitter encoders + TF-IDF | 0.7095 (OOF, all originals) | 0.70532 | 0.73421 |
| v3 — + DeBERTa-v3-large, blend chosen by nested CV | 0.7153 (OOF) | — | 0.74155 |
| **v4.2 — encoders + Qwen2.5-7B QLoRA (2 runs + full-data refit), a = 0.6** | **0.7301 (fold 0)** | **0.72413** | **0.76064** |
| v4.3 — same with a 3rd LLM run, a = 0.5 | 0.7319 (fold 0) | 0.71159 | 0.75770 |

Baseline for reference: TF-IDF (word + char n-grams) + logistic regression — 0.5862 OOF.

## Repository

```
notebooks/
  sentiment_pipeline_best_private.ipynb   v4.2 — the submission that scored 0.76064 (6th place)
  sentiment_pipeline_submitted.ipynb      v4.3 — the notebook sent to the organisers with the final submission
  helpers/
    llm_full_refit_kaggle.ipynb           full-data LLM refit, run on a Kaggle T4 after the Colab quota ran out
    llm_run3_kaggle.ipynb                 3rd LLM run (v4.3 only)
experiments/
  finetune.py                             first CPU fine-tuning script (v1)
  pilot_augmentation.ipynb                do augmented copies / longer inputs help? (no: −0.1…−0.3 pt)
  pilot_large_unfreeze.ipynb              full fine-tuning and layer-wise LR decay for the large encoder
data/README.md                            where the private data goes
```

The two main notebooks share the same code; v4.3 only adds the third LLM run to `LLM_RUNS`. Each notebook explains
every step in markdown: EDA, cleaning, baseline, encoders, ensemble selection, error analysis, the LLM, and a
compliance section with the resources used.

## How to run

1. Open a notebook in Google Colab → Runtime → Change runtime type → **T4 GPU**.
2. Runtime → **Run all**, allow Google Drive (used as a cache for finished model runs), upload `CAIRLAB.csv` and
   `test.csv` when asked.
3. From scratch the full pipeline takes about 6 h on a T4 (25 encoder runs + the LLM). Cached runs are reused.

Seeds are fixed and deterministic CUDA kernels are enabled; bit-exact reproduction needs the same GPU type and library
versions ([`requirements.txt`](requirements.txt)).

## Tech

Python, PyTorch, Hugging Face Transformers, bitsandbytes (4-bit), scikit-learn, pandas, Google Colab / Kaggle (T4).
