# utils.py
import torch
import streamlit as st
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, roc_auc_score

def train_step(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    y_true = data.y[mask].cpu()
    y_pred = pred[mask].cpu()
    f1 = f1_score(y_true, y_pred, average='weighted')
    prob = torch.nn.functional.softmax(out[mask], dim=-1).cpu()
    try:
        roc_auc = roc_auc_score(y_true, prob, multi_class='ovr')
    except ValueError:
        roc_auc = 0.0
    return acc, f1, roc_auc, y_true, y_pred

@st.cache_data(persist=True)
def get_tsne_embeddings(_model, _data):
    _model.eval()
    with torch.no_grad():
        embeddings = _model(_data.x, _data.edge_index)
    
    # CORRECTED LINE: 'n_iter' is renamed to 'max_iter' in newer scikit-learn versions
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300)
    tsne_results = tsne.fit_transform(embeddings.cpu().numpy())
    
    df = pd.DataFrame({
        "tsne_1": tsne_results[:, 0],
        "tsne_2": tsne_results[:, 1],
        "label": _data.y.cpu().numpy()
    })
    return df
