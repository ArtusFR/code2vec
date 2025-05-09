# %% [markdown]
# # Importation

# %%
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import sys
import pickle

import models

from tqdm import tqdm

#from torch_geometric.nn import summary

# %% [markdown]
# # Paramètres

# %%
SEED = 1234
DATA_DIR = 'data'
DATASET_PATH = 'java-small-preprocessed-code2vec/java-small'
DATASET_NAME = 'java-small'
EMBEDDING_DIM = 128
DROPOUT = 0.25
BATCH_SIZE = 128
MAX_LENGTH = 200
LOG_EVERY = 1000 #print log of results after every LOG_EVERY batches
N_EPOCHS = 20
START_EPOCHS = 0
LOG_DIR = 'logs'
SAVE_DIR = 'checkpoints'
LOG_PATH = os.path.join(LOG_DIR, f'{DATASET_NAME}-log.txt')
STATE_FILE = os.path.join(SAVE_DIR, f"state_file.pth")
# MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f'{DATASET_NAME}-{curent_epoch:02}-model.pt') # besion de definir curent_epoch -> (1 ocurence)
LOAD = True #set true if you want to load model from MODEL_SAVE_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Log func

# %%
def logfunc(log):
    with open(LOG_PATH, 'a+') as f:
        f.write(log+'\n')
    print(log)

logfunc(f"will use : {device}")

# %% [markdown]
# ## Dir init

# %%
if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

if not os.path.isdir(f'{LOG_DIR}'):
    os.makedirs(f'{LOG_DIR}')

""" if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH) """

# %% [markdown]
# # Chargement des données

# %% [markdown]
# ## Dict des word (variables), path, target

# %%
with open(f'{DATA_DIR}/{DATASET_PATH}/{DATASET_NAME}.dict.c2v', 'rb') as file:
    word2count = pickle.load(file)
    path2count = pickle.load(file)
    target2count = pickle.load(file)
    n_training_examples = pickle.load(file)

# create vocabularies, initialized with unk and pad tokens

word2idx = {'<unk>': 0, '<pad>': 1}
path2idx = {'<unk>': 0, '<pad>': 1}
target2idx = {'<unk>': 0, '<pad>': 1}

for w in word2count.keys():
    word2idx[w] = len(word2idx)

for p in path2count.keys():
    path2idx[p] = len(path2idx)

for t in target2count.keys():
    target2idx[t] = len(target2idx)

idx2word = {v: k for k, v in word2idx.items()}
idx2path = {v: k for k, v in path2idx.items()}
idx2target = {v: k for k, v in target2idx.items()}

# %%
del pickle

# %%
logfunc(f"nb_target : {len(idx2target)}, nb_var : {len(idx2word)}, nb_path {len(idx2path)}")

# %% [markdown]
# ## File Reading

# %%
def load_data(file_path):
    with open(file_path, 'r') as f:
        return [
            (line.split(' ')[0], [t.split(',') for t in line.split(' ')[1:] if t.strip()])
            for line in f if len(line.split(' ')) - 1 <= MAX_LENGTH
        ]

# %%
def load_data(file_path):
    data = []
    
    with open(file_path, 'r') as f:
        for line in tqdm(f.readlines(), f"load {file_path}"):
            parts = line.strip().split(' ')
            if len(parts) - 1 > MAX_LENGTH:
                continue
            
            name = target2idx.get(parts[0], target2idx['<unk>'])
            
            path_contexts = [tuple(t.split(',')) for t in parts[1:] if t.strip()]
            left, path, right = zip(*path_contexts) if path_contexts else ([], [], [])
            
            left_tensor = torch.tensor([word2idx.get(l, word2idx['<unk>']) for l in left], dtype=torch.long)
            path_tensor = torch.tensor([path2idx.get(p, path2idx['<unk>']) for p in path], dtype=torch.long)
            right_tensor = torch.tensor([word2idx.get(r, word2idx['<unk>']) for r in right], dtype=torch.long)

            data.append((torch.tensor(name, dtype=torch.long), left_tensor, path_tensor, right_tensor))
    
    return data

# %%
data_test = load_data(f'{DATA_DIR}/{DATASET_PATH}/{DATASET_NAME}.test.c2v')

# %%
data_val = load_data(f'{DATA_DIR}/{DATASET_PATH}/{DATASET_NAME}.val.c2v')

# %%
data_train = load_data(f'{DATA_DIR}/{DATASET_PATH}/{DATASET_NAME}.train.c2v')

# %%
logfunc(f"len(data_test)={len(data_test)}, len(data_val)={len(data_val)}, len(data_train)={len(data_train)}")

# %%
n_training_examples

# %% [markdown]
# ## Data Loader

# %%
def collate_fn(samples):
    name_idx = torch.stack([e[0] for e in samples])
    
    max_length = max(len(e[1]) for e in samples)
    
    def pad_tensor(tensor_list, pad_value):
        return torch.stack([torch.cat([t, torch.full((max_length - len(t),), pad_value)]) for t in tensor_list])

    left_tensor = pad_tensor([e[1] for e in samples], word2idx['<pad>'])
    path_tensor = pad_tensor([e[2] for e in samples], path2idx['<pad>'])
    right_tensor = pad_tensor([e[3] for e in samples], word2idx['<pad>'])

    return name_idx, left_tensor, path_tensor, right_tensor

# %%
train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, collate_fn=collate_fn,
                          pin_memory=True, shuffle=True, num_workers=0, prefetch_factor=None)
test_loader = DataLoader(data_test, batch_size=BATCH_SIZE, collate_fn=collate_fn, 
                         pin_memory=True, shuffle=False, num_workers=0, prefetch_factor=None)
eval_loader = DataLoader(data_val, batch_size=BATCH_SIZE, collate_fn=collate_fn, 
                         pin_memory=True, shuffle=False, num_workers=0, prefetch_factor=None)

# %%
len(train_loader), len(test_loader), len(eval_loader)

# %%
c = [0 for i in range(4)]
for ts in tqdm(train_loader, "test for 0 in train tensor"):
    for j, t in enumerate(ts):
        c[j] += t.eq(0).sum().item()
print(c)

# %%
del c, ts

# %%
m, Ma = sys.maxsize, 0
for v in path2count.values():
    m, Ma = min(m,v), max(Ma,v)
print(f"path count range : {(m, Ma)}, mean : {sum(path2count.values())/len(path2count.values())}")
del m, Ma

# %%
train_loader.desc = "train"
test_loader.desc = "test"
eval_loader.desc = "eval"

# %% [markdown]
# # Instanciation

# %% [markdown]
# ## Seed Fixing

# %%
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

# %% [markdown]
# ## State load and save func

# %%
metrics_history = {
    "train_loss": [-1],
    "train_acc": [-1],
    "train_p": [-1],
    "train_r": [-1],
    "train_f1": [-1],
    "valid_loss": [-1],
    "valid_acc": [-1],
    "valid_p": [-1],
    "valid_r": [-1],
    "valid_f1": [-1]
}

def save_state(filepath: str, curent_epoch: int):
    """Save RNG states for PyTorch, CUDA and epochs number."""
    states = {
        'torch_state': torch.get_rng_state(),
        'torch_cuda_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'metrics_history': metrics_history,
        'curent_epoch': curent_epoch
    }
    torch.save(states, filepath)
    logfunc(f"RNG states saved to {filepath}")

def load_state(filepath: str):
    """Load RNG states for PyTorch, CUDA and epochs number."""
    global metrics_history

    states = torch.load(filepath)
    
    curent_epoch=states['curent_epoch']
    metrics_history=states['metrics_history']

    torch.set_rng_state(states['torch_state'])
    if torch.cuda.is_available() and states['torch_cuda_state'] is not None:
        torch.cuda.set_rng_state_all(states['torch_cuda_state'])
    
    logfunc(f"RNG states loaded from {filepath}")
    return curent_epoch

# %% [markdown]
# ## instanciation

# %%
model = models.Code2Vec(
    nodes_dim=      len(word2idx),      # nb de "var"
    paths_dim=      len(path2idx),      # nb de path
    embedding_dim=  EMBEDDING_DIM,      # à découpé
    output_dim=     len(target2idx),    # nb de classe
    dropout=        DROPOUT).to(device)

# %% [markdown]
# ## weight loading, curent_epoch and rng restore 

# %%
try: 
    curent_epoch = load_state(STATE_FILE)
except:
    curent_epoch = START_EPOCHS

if LOAD and (curent_epoch != START_EPOCHS):
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f'{DATASET_NAME}-{curent_epoch:02}-model.pt')

    logfunc(f'Loading model from {MODEL_SAVE_PATH}, restart from {curent_epoch}')
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))


# %%
optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss().to(device)
eval_criterion = nn.CrossEntropyLoss(reduction='sum').to(device)

# %% [markdown]
# ## Overview

# %%
logfunc(f"\nModel structure: {model}\n")

# %%
for i in train_loader:
    a=i
    break
#logfunc(summary(model, *[b.to(device) for b in a][1:]))
logfunc(f"shape for sumary: {[i.shape for i in a]}")
logfunc("\n")

# %% [markdown]
# # Training

# %% [markdown]
# ## métrique

# %%
def calculate_accuracy(fx:torch.Tensor, y:torch.Tensor):
    """
    Calculate top-1 accuracy

    fx = [batch size, output dim]
     y = [batch size]
    """
    pred_idxs = fx.max(1, keepdim=True)[1]
    correct = pred_idxs.eq(y.view_as(pred_idxs)).sum()
    acc = correct.float()/pred_idxs.shape[0]
    return acc

def calculate_f1(fx, y):
    """
    Calculate precision, recall and F1 score
    - Takes top-1 predictions
    - Converts to strings
    - Splits into sub-tokens
    - Calculates TP, FP and FN
    - Calculates precision, recall and F1 score

    fx = [batch size, output dim]
     y = [batch size]
    """
    pred_idxs = fx.max(1, keepdim=True)[1]
    pred_names = [idx2target[i.item()] for i in pred_idxs]
    original_names = [idx2target[i.item()] for i in y]
    true_positive, false_positive, false_negative = 0, 0, 0
    for p, o in zip(pred_names, original_names):
        predicted_subtokens = p.split('|')
        original_subtokens = o.split('|')
        for subtok in predicted_subtokens:
            if subtok in original_subtokens:
                true_positive += 1
            else:
                false_positive += 1
        for subtok in original_subtokens:
            if not subtok in predicted_subtokens:
                false_negative += 1
    try:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        precision, recall, f1 = 0, 0, 0
    return precision, recall, f1


def get_metrics(tensor_n, tensor_l, tensor_p, tensor_r, model, criterion):
    """
    Takes inputs, calculates loss, accuracy and other metrics, then calculates gradients and updates parameters

    if optimizer is None, then we are doing evaluation so no gradients are calculated and no parameters are updated
    """

    fx = model(tensor_l, tensor_p, tensor_r)

    loss = criterion(fx, tensor_n)

    acc = calculate_accuracy(fx, tensor_n)
    precision, recall, f1 = calculate_f1(fx, tensor_n)

    return loss, acc, precision, recall, f1

# %% [markdown]
# ## Eval func

# %%
def evaluate(model:torch.nn, eval_loader:DataLoader, criterion, device:torch.device):
    """
    Evaluation loop using DataLoader.
    Wraps computations in `torch.no_grad()` to avoid unnecessary gradient calculations.
    """

    model.eval()  # Set model to evaluation mode

    cuml_loss, cuml_acc = 0, 0
    true_positive, false_positive, false_negative = 0, 0, 0

    nb_ex = len(eval_loader.dataset)

    with torch.no_grad():
        for tensor_n, tensor_l, tensor_p, tensor_r in tqdm(eval_loader, desc=f"eval for {eval_loader.desc} batch", position=1):
            # Move tensors to GPU
            tensor_n = tensor_n.to(device, non_blocking=True)
            tensor_l = tensor_l.to(device, non_blocking=True)
            tensor_p = tensor_p.to(device, non_blocking=True)
            tensor_r = tensor_r.to(device, non_blocking=True)
            if torch.cuda.is_available(): torch.cuda.synchronize(device)

            fx = model(tensor_l, tensor_p, tensor_r)

            cuml_loss += criterion(fx, tensor_n)

            # top-1 prediction
            pred_idxs = fx.max(1, keepdim=True)[1]

            #acc = calculate_accuracy(fx, tensor_n)
            cuml_acc += pred_idxs.eq(tensor_n.view_as(pred_idxs)).sum()

            #p, r, f1 = calculate_f1(fx, tensor_n)
            """Calculate precision, recall and F1 score
            - Converts to strings
            - Splits into sub-tokens
            - Calculates TP, FP and FN
            - Calculates precision, recall and F1 score"""
            pred_names = [idx2target[i.item()] for i in pred_idxs]
            original_names = [idx2target[i.item()] for i in tensor_n]
            for p, o in zip(pred_names, original_names):
                predicted_subtokens = p.split('|')
                original_subtokens = o.split('|')
                for subtok in predicted_subtokens:
                    if subtok in original_subtokens:
                        true_positive += 1
                    else:
                        false_positive += 1
                for subtok in original_subtokens:
                    if not subtok in predicted_subtokens:
                        false_negative += 1

    try:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        precision, recall, f1 = 0, 0, 0

    return cuml_loss / nb_ex, cuml_acc / nb_ex, precision, recall, f1


# %% [markdown]
# ## Training func

# %%
def train(model:torch.nn, train_loader:DataLoader, optimizer, criterion, device:torch.device):
    """
    Training loop using DataLoader for batch streaming
    """
    model.train()

    n_batches = 0

    for tensor_n, tensor_l, tensor_p, tensor_r in tqdm(train_loader, desc="batch - trainning", position=1):
        # Move tensors to GPU
        tensor_n = tensor_n.to(device, non_blocking=True)
        tensor_l = tensor_l.to(device, non_blocking=True)
        tensor_p = tensor_p.to(device, non_blocking=True)
        tensor_r = tensor_r.to(device, non_blocking=True)
        if torch.cuda.is_available(): torch.cuda.synchronize(device)

        # Forward pass
        optimizer.zero_grad()
               
        fx = model(tensor_l, tensor_p, tensor_r)
        loss = criterion(fx, tensor_n)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update metrics
        n_batches += 1


# %% [markdown]
# ## Training Loop

# %%
import gc
collected = gc.collect()
print(collected)

# %%
best_valid_loss = float('inf')

for epoch in tqdm(range(curent_epoch+1, N_EPOCHS+1), desc="epoch", position=0):
    logfunc(f"Epoch: {epoch:02} - Training")
    train(model, train_loader, optimizer, criterion, device)

    logfunc(f"Epoch: {epoch:02} - Validation - train dataset")
    train_loss, train_acc, train_p, train_r, train_f1 = evaluate(model, train_loader, eval_criterion, device)

    logfunc(f"Epoch: {epoch:02} - Validation - valid dataset")
    valid_loss, valid_acc, valid_p, valid_r, valid_f1 = evaluate(model, eval_loader, eval_criterion, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss

    metrics_history["train_loss"].append(train_loss), metrics_history["train_acc"].append(train_acc), metrics_history["train_p"].append(train_p), metrics_history["train_r"].append(train_r), metrics_history["train_f1"].append(train_f1), 
    metrics_history["valid_loss"].append(valid_loss), metrics_history["valid_acc"].append(valid_acc), metrics_history["valid_p"].append(valid_p), metrics_history["valid_r"].append(valid_r), metrics_history["valid_f1"].append(valid_f1)
    
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{DATASET_NAME}-{epoch:02}-model.pt'))

    save_state(STATE_FILE, epoch)

    log = f"| Epoch: {epoch:02} |\n"
    log += f"| Train Loss: {train_loss:.3f} | Train Precision: {train_p:.3f} | Train Recall: {train_r:.3f} | Train F1: {train_f1:.3f} | Train Acc: {train_acc * 100:.2f}% |\n"
    log += f"| Val. Loss: {valid_loss:.3f} | Val. Precision: {valid_p:.3f} | Val. Recall: {valid_r:.3f} | Val. F1: {valid_f1:.3f} | Val. Acc: {valid_acc * 100:.2f}% |"
    logfunc(log)


# %% [markdown]
# # Testing

# %%
logfunc('Testing')

# model.load_state_dict(torch.load(MODEL_SAVE_PATH))

test_loss, test_acc, test_p, test_r, test_f1 = evaluate(model, test_loader, criterion, device)

logfunc(f'| Test Loss: {test_loss:.3f} | Test Precision: {test_p:.3f} | Test Recall: {test_r:.3f} | Test F1: {test_f1:.3f} | Test Acc: {test_acc*100:.2f}% |')


