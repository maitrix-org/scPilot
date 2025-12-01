# ## whole 40 df
import pandas as pd
import os
import glob
# ---------- CONFIG ----------

### you just need to change here:
folder_path = "mouse_GRN"
trrust_path = "trrust_rawdata.mouse.tsv"
MODEL_NAME      = "gemini-2.0-pro-exp-02-05"
MODEL_PROVIDER  = "google"
PREDICT_CONTEXT = "Stomach"
OUTFILE     = "mouse_GRN/1_llm_reasoning_log_Stomach.jsonl"     # one JSON row per question
TASK_DF_LOCATION = "mouse_GRN/1_Stomach_tasks.csv" ### modify this
RESULT_OUTPUT_LOCATION = "mouse_GRN/1_Stomach_score.txt"
TEST_EVAL = "mouse_GRN/1_Stomach_cutoff_test.csv"

### no need to change these for now:
nes_threshold = 3.0
genie3_threshold = 0.003
file_pattern = os.path.join(folder_path, "whole_*.txt")
N_NEG_PER_POS   = 1        # class balance; 1 negative for each positive
# GCN model parameters
epochs = 20               ###### increase  this 
MAX_PROMPT_LEN  = 4096     # guardrail
LEARNING_RATE=1e-2
### LLM parameters:
BINARY_CUTOFF = 0.2
###################################################################
# --- 1.  LOAD DATA ----------------------------------------------------------
trrust_df = pd.read_csv(
    trrust_path,
    sep="\t",
    names=["TF", "Target", "Mode", "PMID"]
)
all_dfs = []

for file_path in glob.glob(file_pattern):
    df = pd.read_csv(file_path, sep="\t")
    
    # Filter for "High" confidence
    df_filtered = df[df["Confidence"] == "High"].copy()
    
    # Additional filters
    df_filtered = df_filtered[
        (df_filtered["NES"] >= nes_threshold) & 
        (df_filtered["Genie3Weight"].notnull()) & 
        (df_filtered["Genie3Weight"] >= genie3_threshold)
    ]
    
    # Extract context from filename
    filename = os.path.basename(file_path)
    context = filename.replace("whole_", "").replace("-regulons.txt", "")
    
    # Add context column
    df_filtered["Context"] = context
    
    all_dfs.append(df_filtered)

# Combine all filtered dataframes
df_combined = pd.concat(all_dfs, ignore_index=True)



df_combined["Context"] = df_combined["Context"].astype("category")


# ---------------------------------------------------------------------
# 0.  Imports & helpers
# ---------------------------------------------------------------------
import torch, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn   import GCNConv
import numpy  as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
import random, warnings

warnings.filterwarnings("ignore", category=UserWarning)   # PyG verbosity

# ---------------------------------------------------------------------
# 1.  Select which graphs are train vs. test
# ---------------------------------------------------------------------
# -- YOUR df_combined must contain at least the columns
#    'TF', 'gene', 'Genie3Weight', 'Context'
# ---------------------------------------------------------------------
# 39 training contexts  ->  put the names in a list
train_contexts = [ctx for ctx in df_combined["Context"].unique().tolist() if ctx != PREDICT_CONTEXT]
test_context   = PREDICT_CONTEXT          # held‑out graph

df_train = df_combined[df_combined.Context.isin(train_contexts)].copy()
df_test  = df_combined.query("Context == @test_context").copy()

print(f"train graphs = {train_contexts!r}")
print(f"test  graph  = {test_context!r}")
print(f"train edges = {len(df_train):,d} | test edges = {len(df_test):,d}")

# ---------------------------------------------------------------------
# 2.  Shared node index across *all* graphs
# ---------------------------------------------------------------------
all_nodes = pd.Index(df_combined.TF).union(df_combined.gene)
node2idx  = {n: i for i, n in enumerate(all_nodes)}
num_nodes = len(all_nodes)

def edges_to_index(df):
    src = df.TF  .map(node2idx).to_numpy()
    dst = df.gene.map(node2idx).to_numpy()
    return torch.as_tensor(np.vstack([src, dst]), dtype=torch.long)

edge_index_train  = edges_to_index(df_train)
edge_weight_train = torch.tensor(df_train.Genie3Weight.values,
                                 dtype=torch.float32)

edge_index_test   = edges_to_index(df_test)          # for later

# ---------------------------------------------------------------------
# 3.  Node features – simple trainable embeddings
# ---------------------------------------------------------------------
feat_dim = 128
x_embed  = torch.nn.Embedding(num_nodes, feat_dim)

# PyG Data object that *includes* edge weights
data = Data(x=x_embed.weight,
            edge_index=edge_index_train,
            edge_weight=edge_weight_train)

# ---------------------------------------------------------------------
# 4.  GCN encoder + dot‑product decoder (unchanged)
# ---------------------------------------------------------------------
class GCNLink(torch.nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.conv2 = GCNConv(hid,  hid)

    def forward(self, x, edge_index, w):
        h = F.relu(self.conv1(x, edge_index, w))
        h = self.conv2(h, edge_index, w)
        return h

def dot_score(h, pairs):                       # pairs = 2×N indices
    return (h[pairs[0]] * h[pairs[1]]).sum(dim=-1)

# ---------------------------------------------------------------------
# 5.  Negative‑edge sampler (uniform corruption, unchanged)
# ---------------------------------------------------------------------
pos_set = set(zip(edge_index_train[0].tolist(),
                  edge_index_train[1].tolist()))

def sample_neg(num_neg):
    u = torch.randint(0, num_nodes, (num_neg,))
    v = torch.randint(0, num_nodes, (num_neg,))
    mask = torch.tensor([(u[i].item(), v[i].item()) not in pos_set
                         for i in range(num_neg)])
    return torch.stack([u[mask], v[mask]], 0)

# ---------------------------------------------------------------------
# 6.  Training loop
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data   = data.to(device)
model  = GCNLink(feat_dim).to(device)

opt    = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, epochs + 1):
    model.train(); opt.zero_grad()
    h = model(data.x, data.edge_index, data.edge_weight)

    # positive & negative scores
    pos_s = dot_score(h, data.edge_index)
    neg_i = sample_neg(pos_s.size(0)).to(device)
    neg_s = dot_score(h, neg_i)

    y_true = torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s)])
    y_pred = torch.cat([pos_s,              neg_s            ])
    loss   = F.binary_cross_entropy_with_logits(y_pred, y_true)

    loss.backward(); opt.step()

    if epoch % 20 == 0:
        print(f"epoch {epoch:03d} | loss = {loss.item():.4f}")


import pandas as pd, numpy as np, re, ast, json
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from utils.LLM import query_llm

# --- 2.  TASK BUILDER -------------------------------------------------------
# ---------------------------------------------------------------------
#  2‑bis.  TASK BUILDER  (train‑pool  →  held‑out graph)
# ---------------------------------------------------------------------
def build_tasks_multi(df, trrust,
                      train_ctxs,           # list of training Context names
                      test_ctx,             # single held‑out Context name
                      n_neg_per_pos=1,
                      max_known=50):  
    """Return a list[dict] where each dict is one binary TF-gene question."""
    
    df_train = df[df.Context.isin(train_ctxs)].copy()
    df_test  = df.query("Context == @test_ctx").copy()
    
    tasks = []
    
    # ---- TFs that have at least one edge in *both* pools --------------
    tf_common = set(df_train.TF).intersection(df_test.TF)
    
    for tf in tf_common:
        # ----------------  context A  (union over 39 graphs)  ----------
        known_A = (
            df_train.loc[df_train.TF == tf, "gene"]
                     .unique()
                     .tolist()
        )
        context_A = df_train.loc[df_train.TF == tf, "Context"].unique().tolist()
        if not known_A:
            continue
        
        # ----------------  context B  (held‑out graph)  ----------------
        cand_B = (
            df_test.loc[df_test.TF == tf, "gene"]
                    .unique()
                    .tolist()
        )
        if not cand_B:
            continue
        
        # ----------------  positives / negatives -----------------------
        pos_set = set(trrust[trrust.TF == tf].Target) & set(cand_B)
        neg_set = set(cand_B) - pos_set
        
        if len(pos_set) == 0 or len(neg_set) == 0:
            continue
        
        # balanced negative sampling
        rng = np.random.default_rng(0)  # reproducible
        n_neg = min(len(pos_set)*n_neg_per_pos, len(neg_set))
        neg_sample = rng.choice(list(neg_set), size=n_neg, replace=False)
        
        # build question dicts
        for gene, label in (
            list(zip(pos_set, [1]*len(pos_set))) +
            list(zip(neg_sample, [0]*len(neg_sample)))
        ):
            tasks.append({
                "TF"        : tf,
                "gene"      : gene,
                "context_A" : context_A,      
                "context_B" : test_ctx,
                "known_A"   : known_A[:max_known],
                "label"     : label
            })
    return tasks


train_contexts  = [c for c in df_combined.Context.unique()
                   if c != PREDICT_CONTEXT]
tasks = build_tasks_multi(df_combined, trrust_df,
                          train_ctxs=train_contexts,
                          test_ctx=PREDICT_CONTEXT,
                          n_neg_per_pos=1)

print(f"Total binary questions: {len(tasks)}  "
      f"({sum(t['label'] for t in tasks)} positives)")


# ---------------------------------------------------------------------
# 7.  Embeddings for *all* nodes after training
# ---------------------------------------------------------------------
model.eval()
with torch.no_grad():
    H = model(data.x, data.edge_index, data.edge_weight).cpu()

# ---------------------------------------------------------------------
# 8.  Helper: evaluate on a paired‑question list (unchanged API)
# ---------------------------------------------------------------------
from torch.nn.functional import sigmoid

def gcn_predict(tasks, H, node2idx, thresh=0.5):
    y_true, y_pred, y_score = [], [], []
    for t in tasks:
        i_tf   = node2idx.get(t["TF"])
        i_gene = node2idx.get(t["gene"])
        if i_tf is None or i_gene is None:      # unseen node guard
            continue
        logit = torch.dot(H[i_tf], H[i_gene]).item()
        prob  = sigmoid(torch.tensor(logit)).item()
        y_true.append(t["label"])
        y_score.append(prob)
        y_pred.append(1 if prob >= thresh else 0)
    return np.array(y_true), np.array(y_pred), np.array(y_score)

# ---------------------------------------------------------------------
# 9.  Example evaluation
#     (replace `tasks` with your actual paired‑question list)
# ---------------------------------------------------------------------
y_true_gcn_whole, y_pred_gcn_whole, y_score_gcn_whole = gcn_predict(tasks, H, node2idx)

def evaluate(y_true, y_pred, y_score):
    p,r,f,_ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    print(f"Precision = {p:.2f}\nRecall    = {r:.2f}"
          f"\nF1 score  = {f:.2f}\nAUROC     = {auc:.2f}")
    return (f"{p:.2f}",f"{r:.2f}",f"{f:.2f}",f"{auc:.2f}")
    

Precision,Recall,F1, AUROC = evaluate(y_true_gcn_whole, y_pred_gcn_whole, y_score_gcn_whole)



with open(RESULT_OUTPUT_LOCATION, "a") as f:
    f.write("GCN \n")
    f.write(f"Precision={Precision} Recall={Recall} F1={F1} AUROC={AUROC}\n")
print(f"Results written to {RESULT_OUTPUT_LOCATION}")

pd.DataFrame(tasks).to_csv(TASK_DF_LOCATION)


### direct

# --- 3.  PROMPT ENGINE ------------------------------------------------------
def make_prompt_basic(task):
    tf, ctxA, ctxB, gene = task["TF"], task["context_A"], task["context_B"], task["gene"]

    prompt = f"""
Decide how much possible {tf} directly regulates {gene} in ({ctxB}):

The possibility is a number from 0 to 1.

Return exactly:
Reasoning: <your reasoning>
Possibility is: <your possibility>
""".strip()

    return prompt[:MAX_PROMPT_LEN]

def run_llm_basic(tasks):
    y_true, y_pred, y_score = [], [], []
    y_possibility = []
    rows = []
    counter = 0
    time_list = []
    for task in tasks:
        counter += 1
        if counter % 20 == 0:
            print(f"Question {counter}")
        try:
            prompt = make_prompt_basic(task)
            import time
            start = time.time()
            #print(start)
            resp   = query_llm(
                prompt,
                system_role="Expert in gene regulatory networks",
                model_name=MODEL_NAME,
                model_provider=MODEL_PROVIDER
            ).strip()
            end = time.time()
            #print("time: "+str(end-start))
            time_list.append(float(end-start))
            match = re.search(r"Possibility is:\s*[*]*\s*([0-9]+(?:\.[0-9]+)?)", resp)
            if match:
                possibility = match.group(1).strip()
            else:
                print(str(counter)+" No choice found. Skipping task")
                continue
            y_possibility.append(possibility)
            y_true.append(task["label"])
            # map to 0/1  (+ probability proxy = 1 if 'yes', else 0)
            pred = 1 if float(possibility) >= 0.5 else 0
            score = 1.0 if float(possibility) >= 0.5 else 0.0

            y_pred.append(pred)
            y_score.append(score)
            
            rows.append({
                "index": counter,
                "task"      : task,
                "possibility"  : possibility,
                "correct" : pred == task['label'],
                "response": resp
            })
        except Exception as e:
            print(f"Error: {e}. Skipping.")
            continue

    # write full reasoning once per run
    with open(OUTFILE, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    return np.array(y_true), np.array(y_pred), np.array(y_score),np.array(y_possibility),time_list

# ---------------------- RUN -------------------------------------------------

y_true_basic, y_pred_basic, y_score_basic,y_possibility_basic,time_list = run_llm_basic(tasks)
avg_time = np.array(time_list).mean()

with open(RESULT_OUTPUT_LOCATION, "a") as f:
    f.write("\nmost basic: \n")
    for item in y_true_basic:
        f.write(f"{item},")
    f.write("\n")
    for item in y_possibility_basic:
        f.write(f"{item},")
    f.write("\n")
    f.write("\nmodel avg run time: \n")
    f.write(str(avg_time))
    f.write("\n")
print(f"Array written to {RESULT_OUTPUT_LOCATION}")

binary_arr = (y_possibility_basic.astype(float) > BINARY_CUTOFF).astype(int)
Precision,Recall,F1, AUROC = evaluate(y_true_basic, binary_arr, binary_arr)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true_basic, binary_arr)
# cm = [[TN, FP],
#       [FN, TP]]
print("Confusion matrix:\n", cm)

with open(RESULT_OUTPUT_LOCATION, "a") as f:
    f.write(f"Precision={Precision} Recall={Recall} F1={F1} AUROC={AUROC}\n")
    f.write("Confusion matrix:\n")
    f.write(f"{cm}\n")
print(f"Most Basic Results written to {RESULT_OUTPUT_LOCATION}")



# ## scPilot

# ---------------------------------------------------------------------
# 0 - bis.  Fetch GO terms for a symbol → fast local cache
# ---------------------------------------------------------------------
import mygene, shelve, os, textwrap
mg = mygene.MyGeneInfo()
GO_CACHE = os.path.expanduser("~/.go_symbol_cache")

def get_go_terms(symbol, n_max=15, refresh=False):
    key = symbol.upper()          # cache insensitive to case
    with shelve.open(GO_CACHE, writeback=True) as db:
        if not refresh and key in db:
            return db[key]

        hits = mg.query(
            symbol, scopes="symbol",
            fields="go.BP.term", species=10090, size=5
        )

        terms = []
        for h in hits.get("hits", []):
            bp = h.get("go", {}).get("BP", [])
            # bp may be dict, list‑of‑dicts, or list‑of‑strings
            if isinstance(bp, dict):
                bp = [bp]

            for entry in bp:
                if isinstance(entry, dict):
                    term = entry.get("term")
                    if term:
                        terms.append(term)
                elif isinstance(entry, str):
                    # you only have the GO ID; keep it or look it up later
                    terms.append(entry)
            if terms:
                break                         # we found at least one hit

        terms = sorted(set(terms))[:n_max]    # de‑dupe & trim
        db[key] = terms                       # cache (even if empty)
        return terms


# one positive and one negative exemplar from a held‑out TF
FEW_SHOTS = [
    dict(
        tf="Tcf3", ctxA="liver", known="Arhgap25, Ripk3, Thy1",
        ctxB="bone marrow", gene="Pax5",
        reasoning="Tcf3 controls B cell lineage commitment; Pax5 is essential for B cell differentiation…",
        answer="I choose A"        # **positive**
    ),
    dict(
        tf="Stat1", ctxA="liver", known="Axl, Cand1, Cybb",
        ctxB="bone marrow", gene="Rnf8",
        reasoning="Stat1 drives antiviral genes; Rnf8 is a DNA-damage E3 ligase unrelated to IFN signalling…",
        answer="I choose B"        # **negative**
    ),
]
def few_shot_block():
    txt = ""
    for ex in FEW_SHOTS:
        txt += (
            f"Example\n"
            f"TF: {ex['tf']}  |  Tissue A: {ex['ctxA']}\n"
            f"Known targets: {ex['known']}\n"
            f"Tissue B: {ex['ctxB']}, Candidate gene: {ex['gene']}\n"
            f"Reasoning: {ex['reasoning']}\n"
            f"{ex['answer']}\n\n"
        )
    return txt


def overlap_terms(tf, gene, k=3):
    tf_terms   = set(get_go_terms(tf,   n_max=20))
    gene_terms = set(get_go_terms(gene, n_max=20))
    common = sorted(tf_terms & gene_terms)[:k]
    return "; ".join(common) if common else "none"


def make_prompt_adv_new(task):
    tf, gene   = task["TF"], task["gene"]
    ctxA, ctxB = task["context_A"], task["context_B"]
    overlap = overlap_terms(tf, gene)
    #known = ", ".join(task["known_A"][:30]) or "none"
    #{known}
    prompt = few_shot_block() + f"""
*Task*: \n
• TF: {tf} and Context A tissues ({ctxA})
• Functional overlap (shared GO BP terms): {overlap}

### Decide how much possible {tf} directly regulates {gene} in ({ctxB}):

The possibility is a number from 0 to 1.

Think step by step:
1. Recall TF {tf}'s biological role.
2. Compare {gene} with known {tf} targets.
3. Conclude which statement fits better (<= 4 sentences).

Return exactly:
Reasoning: <your reasoning>
Possibility is: <your possibility>
""".strip()
    return prompt[:MAX_PROMPT_LEN]

def run_llm_adv(tasks):
    y_true, y_pred, y_score = [], [], []
    y_possibility = []
    rows = []
    counter = 0
    for task in tasks:
        counter += 1
        if counter % 20 == 0:
            print(f"Question {counter}")
        try:
            prompt = make_prompt_adv_new(task)
            resp   = query_llm(
                prompt,
                system_role="Expert in gene regulatory networks",
                model_name=MODEL_NAME,
                model_provider=MODEL_PROVIDER
            ).strip()
            match = re.search(r"Possibility is:\s*[*]*\s*([0-9]+(?:\.[0-9]+)?)", resp)
            if match:
                possibility = match.group(1).strip()
            else:
                print(str(counter)+" No choice found. Skipping task")
                continue
            y_possibility.append(possibility)
            y_true.append(task["label"])
            # map to 0/1  (+ probability proxy = 1 if 'yes', else 0)
            pred = 1 if float(possibility) >= 0.5 else 0
            score = 1.0 if float(possibility) >= 0.5 else 0.0

            y_pred.append(pred)
            y_score.append(score)
            
            rows.append({
                "index": counter,
                "task"      : task,
                "possibility"  : possibility,
                "correct" : pred == task['label'],
                "response": resp
            })
        except Exception as e:
            print(f"Error: {e}. Skipping.")
            continue

    # write full reasoning once per run
    with open(OUTFILE, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    return np.array(y_true), np.array(y_pred), np.array(y_score),np.array(y_possibility)



# ---------------------- RUN -------------------------------------------------
y_true_adv, y_pred_adv, y_score_adv,y_possibility_adv = run_llm_adv(tasks)


with open(RESULT_OUTPUT_LOCATION, "a") as f:
    f.write("\nadvanced: \n")
    for item in y_true_adv:
        f.write(f"{item},")
    f.write("\n")
    for item in y_possibility_adv:
        f.write(f"{item},")
    f.write("\n")
print(f"Array written to {RESULT_OUTPUT_LOCATION}")


binary_arr = (y_possibility_adv.astype(float) > BINARY_CUTOFF).astype(int)
Precision,Recall,F1, AUROC = evaluate(y_true_adv, binary_arr, binary_arr)
cm = confusion_matrix(y_true_adv, binary_arr)
print("Confusion matrix:\n", cm)
with open(RESULT_OUTPUT_LOCATION, "a") as f:
    f.write(f"Precision={Precision} Recall={Recall} F1={F1} AUROC={AUROC}\n")
    f.write("Confusion matrix:\n")
    f.write(f"{cm}\n")
print(f"Advanced Results written to {RESULT_OUTPUT_LOCATION}")

results = []
for test_cutoff in [0.2, 0.3, 0.4]:
    for method_name, y_true_arr, y_poss_arr in [
        #("Most basic", y_true_basic, y_possibility_basic),
        #("Intermediate", y_true, y_possibility),
        ("Advanced", y_true_adv, y_possibility_adv)
    ]:
        binary_arr = (y_poss_arr.astype(float) > test_cutoff).astype(int)
        Precision, Recall, F1, AUROC = evaluate(y_true_arr, binary_arr, binary_arr)
        cm = confusion_matrix(y_true_arr, binary_arr)
        tn, fp, fn, tp = cm.ravel()

        results.append({
            "test_cutoff": test_cutoff,
            "method": method_name,
            "precision": Precision,
            "recall": Recall,
            "f1_score": F1,
            "auroc": AUROC,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        })

# Create DataFrame
df_results = pd.DataFrame(results)

df_results.to_csv(TEST_EVAL, index=False)
print(f"Saved results to {TEST_EVAL}")