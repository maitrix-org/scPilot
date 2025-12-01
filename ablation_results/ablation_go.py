# ## whole 40 df
import pandas as pd
import os
import glob
# ---------- CONFIG ----------
folder_path = "mouse_GRN"
file_pattern = os.path.join(folder_path, "whole_*.txt")
trrust_path = "trrust_rawdata.mouse.tsv"
nes_threshold = 3.0
genie3_threshold = 0.003
MODEL_NAME      = "gpt-4o"
MODEL_PROVIDER  = "openai"
N_NEG_PER_POS   = 1        # class balance; 1 negative for each positive
OUTFILE     = "ablation_results/go_overlap/4o/2_llm_reasoning_log_Stomach.jsonl"     # one JSON row per question
TASK_DF_LOCATION = "ablation_results/go_overlap/4o/2_Stomach_tasks.csv" ### modify this
RESULT_OUTPUT_LOCATION = "ablation_results/go_overlap/4o/2_Stomach_score.txt"
TEST_EVAL = "ablation_results/go_overlap/4o/2_Stomach_cutoff_test.csv"
# GCN model parameters
PREDICT_CONTEXT = "Stomach"
MAX_PROMPT_LEN  = 4096     # guardrail
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


# ## most advanced

# ---------------------------------------------------------------------
# 0‑bis.  Fetch GO terms for a symbol → fast local cache
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
        answer="Possibility is 0.9"        # **positive**
    ),
    dict(
        tf="Stat1", ctxA="liver", known="Axl, Cand1, Cybb",
        ctxB="bone marrow", gene="Rnf8",
        reasoning="Stat1 drives antiviral genes; Rnf8 is a DNA-damage E3 ligase unrelated to IFN signalling…",
        answer="Possibility is 0.1"        # **negative**
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
#y_true_adv, y_pred_adv, y_score_adv,y_possibility_adv = run_llm_adv(tasks)

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
'''
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

'''
############# ablation on GO

def no_overlap_terms(tf, gene, k=10):
    all_terms = get_go_terms("Stat1", n_max=100)
    sampled_terms = random.sample(all_terms, k)
    not_common = sorted(sampled_terms)[:k]
    return "; ".join(not_common) if not_common else "none"

def make_prompt_ablate(task):
    tf, gene   = task["TF"], task["gene"]
    ctxA, ctxB = task["context_A"], task["context_B"]
    overlap = no_overlap_terms(tf, gene)
    #known = ", ".join(task["known_A"][:30]) or "none"
    #{known}
    prompt = f"""
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

def run_llm_ablate(tasks):
    y_true, y_pred, y_score = [], [], []
    y_possibility = []
    rows = []
    counter = 0
    for task in tasks:
        counter += 1
        if counter % 20 == 0:
            print(f"Question {counter}")
        try:
            prompt = make_prompt_ablate(task)
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
y_true_ablate, y_pred_ablate, y_score_ablate,y_possibility_ablate = run_llm_ablate(tasks)

with open(RESULT_OUTPUT_LOCATION, "a") as f:
    f.write("\n ablated: \n")
    for item in y_true_ablate:
        f.write(f"{item},")
    f.write("\n")
    for item in y_possibility_ablate:
        f.write(f"{item},")
    f.write("\n")
print(f"Array written to {RESULT_OUTPUT_LOCATION}")
    
binary_arr = (y_possibility_ablate.astype(float) > BINARY_CUTOFF).astype(int)
Precision,Recall,F1, AUROC = evaluate(y_true_ablate, binary_arr, binary_arr)
cm = confusion_matrix(y_true_ablate, binary_arr)
print("Confusion matrix:\n", cm)
with open(RESULT_OUTPUT_LOCATION, "a") as f:
    f.write(f"Precision={Precision} Recall={Recall} F1={F1} AUROC={AUROC}\n")
    f.write("Confusion matrix:\n")
    f.write(f"{cm}\n")
print(f"Ablate Results written to {RESULT_OUTPUT_LOCATION}")
