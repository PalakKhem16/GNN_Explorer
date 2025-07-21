# visualizations.py
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch_geometric.utils import to_networkx, k_hop_subgraph
import optuna

def plot_label_distribution(data, dataset_name):
    label_counts = pd.Series(data.y.numpy()).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(label_counts.index, label_counts.values, color='skyblue')
    ax.set_title(f"Label Distribution for {dataset_name}")
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Number of Nodes")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    return fig

def plot_subgraph(data):
    g = to_networkx(data, to_undirected=True, node_attrs=['y'])
    subgraph_nodes = list(g.nodes())[:20]
    subgraph = g.subgraph(subgraph_nodes)
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.spring_layout(subgraph, seed=42)
    node_colors = [data.y[n] for n in subgraph.nodes()]
    nx.draw_networkx(subgraph, pos, ax=ax, with_labels=True, node_color=node_colors, cmap=plt.cm.jet, node_size=300, font_size=10)
    ax.set_title("Sample Subgraph (20 Nodes)")
    return fig

def plot_training_curves(results_df):
    fig = px.line(results_df, x='Epoch', y='Validation Accuracy', color='Run', title='Validation Accuracy Over Epochs', markers=True)
    fig.update_layout(legend_title_text='Experimental Runs')
    return fig

def plot_tsne(tsne_df):
    fig = px.scatter(tsne_df, x='tsne_1', y='tsne_2', color='label', title='2D t-SNE of Node Embeddings', labels={'color': 'Node Class'}, hover_data={'label': True})
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    return fig

def plot_confusion_matrix(y_true, y_pred, class_count):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    tick_marks = range(class_count)
    ax.set_xticks(tick_marks, labels=tick_marks, rotation=45)
    ax.set_yticks(tick_marks, labels=tick_marks, rotation=0)
    return fig

@torch.no_grad()
def plot_interactive_neighborhood(data, model, node_idx, hops=1):
    model.eval()
    all_preds = model(data.x, data.edge_index).argmax(dim=1)
    all_true_labels = data.y
    nodes_subset, _, mapping, _ = k_hop_subgraph(
        node_idx=node_idx, num_hops=hops, edge_index=data.edge_index, relabel_nodes=True)
    subgraph_data = data.subgraph(nodes_subset)
    g = to_networkx(subgraph_data, to_undirected=True)
    pos = nx.spring_layout(g, seed=42)
    edge_x, edge_y = [], []
    for edge in g.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y = [], []
    for node in g.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
        marker=dict(showscale=True, colorscale='Jet', size=10, 
                    colorbar=dict(thickness=15, title=dict(text='Predicted Class', side='right'), xanchor='left')))
    custom_data, node_colors, node_sizes, node_borders = [], [], [], []
    original_node_ids = nodes_subset.cpu().numpy()
    for i, node in enumerate(g.nodes()):
        original_id = original_node_ids[i]
        pred_class = all_preds[original_id].item()
        true_class = all_true_labels[original_id].item()
        node_colors.append(pred_class)
        custom_data.append([pred_class, true_class])
        node_sizes.append(20 if original_id == node_idx else 10)
        node_borders.append('black' if original_id == node_idx else 'white')
    node_trace.marker.color = node_colors
    node_trace.marker.size = node_sizes
    node_trace.marker.line = dict(width=[2 if s == 20 else 1 for s in node_sizes], color=node_borders)
    node_trace.customdata = custom_data
    node_trace.text = [f"Node ID: {original_node_ids[i]}" for i in range(len(g.nodes()))]
    node_trace.hovertemplate = "<b>%{text}</b><br>Predicted Class: %{customdata[0]}<br>Actual Class: %{customdata[1]}<extra></extra>"
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title=dict(text=f'<br>Interactive {hops}-Hop Neighborhood of Node {node_idx}', font=dict(size=16)),
                                     showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig

# --- OPTUNA VISUALIZATIONS ---
def plot_optimization_history(study):
    return optuna.visualization.plot_optimization_history(study)

def plot_param_importances(study):
    return optuna.visualization.plot_param_importances(study)