"""CPU-friendly fine-tuning of Twitter-RoBERTa for entity-level sentiment.

Trains on the *original* tweet of every training ID (the first of its 6 rows) from the
training folds, evaluates on the original tweets of the held-out GroupKFold fold, and
predicts the test set. Outputs probabilities to features/ so the notebook can blend them.
"""
import os, random, time, argparse
import numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

p = argparse.ArgumentParser()
p.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
p.add_argument("--tag", default="ft_cardiff")
p.add_argument("--fold", type=int, default=0)          # -1 = train on all training data
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--freeze", type=int, default=4)         # freeze embeddings + this many lower layers
p.add_argument("--max_len", type=int, default=96)
p.add_argument("--smoothing", type=float, default=0.1)
p.add_argument("--n_aug", type=int, default=0)          # extra augmented variants per ID to train on
p.add_argument("--seed", type=int, default=42)
p.add_argument("--threads", type=int, default=8)
args = p.parse_args()

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); os.environ["PYTHONHASHSEED"] = str(s)
seed_everything(args.seed)
torch.set_num_threads(args.threads)
torch.use_deterministic_algorithms(True)

train = pd.read_csv("CAIRLAB.csv", header=None, names=["ID", "Entity", "Sentiment", "Text"])
test = pd.read_csv("test.csv")
for d in (train, test): d["Text"] = d["Text"].fillna("").astype(str)
CLASSES = np.array(sorted(train.Sentiment.unique()))
train["y"] = train.Sentiment.map({c: i for i, c in enumerate(CLASSES)})
train["pos"] = train.groupby("ID").cumcount()

folds = list(GroupKFold(n_splits=5).split(train, train.y, train.ID))
if args.fold >= 0:
    tr_rows, va_rows = folds[args.fold]
else:
    tr_rows, va_rows = np.arange(len(train)), np.array([], dtype=int)
tr_df = train.iloc[tr_rows]; tr_df = tr_df[tr_df.pos <= args.n_aug]
va_df = train.iloc[va_rows]; va_df = va_df[va_df.pos == 0]
print(f"train rows {len(tr_df)} | valid (original tweets) {len(va_df)}", flush=True)

def prep(t):  # CardiffNLP preprocessing
    return " ".join("@user" if w.startswith("@") and len(w) > 1 else "http" if w.startswith("http") else w
                    for w in t.split(" "))

tok = AutoTokenizer.from_pretrained(args.model, **({"normalization": True} if "bertweet" in args.model else {}))
model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=len(CLASSES), ignore_mismatched_sizes=True)
base = model.roberta
for prm in base.embeddings.parameters(): prm.requires_grad = False
for layer in base.encoder.layer[:args.freeze]:
    for prm in layer.parameters(): prm.requires_grad = False

def encode(df):
    if len(df) == 0: return []
    return tok(df.Entity.tolist(), [prep(t) for t in df.Text], truncation=True, max_length=args.max_len)["input_ids"]

def batches(ids, order, bs):
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        L = max(len(ids[j]) for j in idx)
        x = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        for r, j in enumerate(idx): x[r, :len(ids[j])] = torch.tensor(ids[j])
        yield idx, x, (x != tok.pad_token_id).long()

def bucket_order(ids, rng, bs, chunk=50):
    perm = rng.permutation(len(ids)); out = []
    for c in range(0, len(perm), bs * chunk):
        part = sorted(perm[c:c + bs * chunk], key=lambda j: len(ids[j]))
        out += [part[i:i + bs] for i in range(0, len(part), bs)]
    rng.shuffle(out)
    return [j for b in out for j in b]

@torch.inference_mode()
def predict(ids):
    model.eval(); P = np.zeros((len(ids), len(CLASSES)), dtype=np.float32)
    order = np.argsort([len(x) for x in ids], kind="stable")
    for idx, x, m in batches(ids, order, 64):
        P[idx] = torch.softmax(model(input_ids=x, attention_mask=m).logits, -1).numpy()
    return P

tr_ids, va_ids, te_ids = encode(tr_df), encode(va_df), encode(test)
ytr = torch.tensor(tr_df.y.values)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.01)
steps = args.epochs * ((len(tr_ids) + args.bs - 1) // args.bs)
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)
rng = np.random.RandomState(args.seed)

t0 = time.time()
for ep in range(args.epochs):
    model.train(); tot = 0.0; n = 0
    for k, (idx, x, m) in enumerate(batches(tr_ids, bucket_order(tr_ids, rng, args.bs), args.bs)):
        loss = loss_fn(model(input_ids=x, attention_mask=m).logits, ytr[idx])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        tot += loss.item() * len(idx); n += len(idx)
        if k % 50 == 0: print(f"  ep{ep} step {k} loss {tot/n:.4f} {time.time()-t0:.0f}s", flush=True)
    msg = f"epoch {ep} train_loss {tot/n:.4f} {time.time()-t0:.0f}s"
    if len(va_ids):
        Pva = predict(va_ids); msg += f" | valid acc (originals) {(Pva.argmax(1) == va_df.y.values).mean():.4f}"
    print(msg, flush=True)

os.makedirs("features", exist_ok=True)
suffix = f"fold{args.fold}" if args.fold >= 0 else "full"
if len(va_ids):
    np.save(f"features/{args.tag}_{suffix}_valid.npy", Pva)
    np.save(f"features/{args.tag}_{suffix}_valid_rows.npy", va_df.index.values)
np.save(f"features/{args.tag}_{suffix}_test.npy", predict(te_ids))
print("done", time.time() - t0, flush=True)
