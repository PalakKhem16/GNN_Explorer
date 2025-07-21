# app.py
import streamlit as st
import torch
import pandas as pd
import time
import plotly.express as px
import optuna
import io
import json

from data_loader import load_dataset
from models import get_model, DynamicGNN
from utils import train_step, evaluate, get_tsne_embeddings
from visualizations import (
    plot_label_distribution, plot_subgraph, plot_training_curves,
    plot_tsne, plot_confusion_matrix, plot_interactive_neighborhood,
    plot_optimization_history, plot_param_importances
)

st.set_page_config(page_title="GNN Explorer", layout="wide")
st.title("GNN Explorer: Interactive Graph Neural Network Analysis")

# --- Session State Initialization ---
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = "Cora"
if 'layer_configs' not in st.session_state:
    st.session_state.layer_configs = [{'type': 'GCN', 'hidden_channels': 64}]
if 'experiment_queue' not in st.session_state:
    st.session_state.experiment_queue = []
if 'run_results' not in st.session_state:
    st.session_state.run_results = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'run_predictions' not in st.session_state:
    st.session_state.run_predictions = {}
if 'single_model_trained' not in st.session_state:
    st.session_state.single_model_trained = None
if 'optuna_study' not in st.session_state:
    st.session_state.optuna_study = None

def on_dataset_change():
    st.session_state.single_model_trained = None
    st.session_state.run_results = None
    st.session_state.trained_models = {}
    st.session_state.run_predictions = {}
    st.session_state.experiment_queue = []
    st.session_state.optuna_study = None
    st.session_state.layer_configs = [{'type': 'GCN', 'hidden_channels': 64}]

st.sidebar.header("Global Controls")
st.sidebar.selectbox(
    "Select Dataset", ("Cora", "Citeseer", "PubMed"),
    key='dataset_name', on_change=on_dataset_change
)

data = load_dataset(st.session_state.dataset_name)
num_features = data.num_features
num_classes = data.y.max().item() + 1

tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Model Builder & Training", "Comparative Analysis", "Hyperparameter Tuning"])

with tab1:
    st.header(f"Dataset: {st.session_state.dataset_name}")
    st.markdown(f"- **Nodes:** `{data.num_nodes}`\n- **Edges:** `{data.num_edges // 2}`\n- **Features:** `{num_features}`\n- **Classes:** `{num_classes}`")
    c1, c2 = st.columns(2)
    c1.pyplot(plot_label_distribution(data, st.session_state.dataset_name))
    c2.pyplot(plot_subgraph(data))

with tab2:
    st.header("Train a Single GNN Model")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Dynamic Architecture Builder")
        for i, layer in enumerate(st.session_state.layer_configs):
            cols = st.columns([3, 2, 1])
            layer['type'] = cols[0].selectbox(f"Layer {i+1} Type", ["GCN", "GAT", "GraphSAGE"], key=f"type_{i}")
            layer['hidden_channels'] = cols[1].number_input("Hidden Channels", 16, 256, layer['hidden_channels'], key=f"hidden_{i}")
            if cols[2].button("Del", key=f"del_{i}"):
                st.session_state.layer_configs.pop(i)
                st.rerun()

        if st.button("Add Layer"):
            st.session_state.layer_configs.append({'type': 'GCN', 'hidden_channels': 64})
            st.rerun()
        
        st.subheader("Training Configuration")
        lr = st.number_input("Learning Rate", 0.001, 0.1, 0.01, format="%.3f", key="single_lr")
        epochs = st.number_input("Epochs", 10, 500, 100, key="single_epochs")
        
        if st.button("Train Model", key="single_train", type="primary"):
            model = DynamicGNN(num_features, num_classes, st.session_state.layer_configs)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss()
            status_text = st.empty()
            chart_placeholder = st.empty()
            training_history = []
            for epoch in range(1, epochs + 1):
                loss = train_step(model, data, optimizer, criterion)
                val_acc, _, _, _, _ = evaluate(model, data, data.val_mask)
                training_history.append({'Epoch': epoch, 'Loss': loss, 'Validation Accuracy': val_acc})
                status_text.text(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")
                history_df = pd.DataFrame(training_history)
                fig = px.line(history_df, x='Epoch', y=['Loss', 'Validation Accuracy'], title='Real-Time Training Progress')
                chart_placeholder.plotly_chart(fig, use_container_width=True)
            st.session_state.single_model_trained = model
            status_text.success(f"Training complete on {st.session_state.dataset_name}!")

    with c2:
        st.subheader("Final Performance Analysis")
        if st.session_state.single_model_trained:
            model = st.session_state.single_model_trained
            test_acc, f1, roc_auc, y_true, y_pred = evaluate(model, data, data.test_mask)
            c2_1, c2_2, c2_3 = st.columns(3)
            c2_1.metric("Test Accuracy", f"{test_acc:.4f}")
            c2_2.metric("F1-Score", f"{f1:.4f}")
            c2_3.metric("ROC-AUC", f"{roc_auc:.4f}")
            with st.expander("Show Final Confusion Matrix"):
                st.pyplot(plot_confusion_matrix(y_true, y_pred, num_classes))
            
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            st.download_button(
                label="Save Trained Model",
                data=buffer,
                file_name=f"model_{st.session_state.dataset_name}.pt",
                mime="application/octet-stream"
            )
        else:
            st.info("Train a model to see performance metrics.")

    st.divider()
    st.header("Interactive Neighborhood Inspection")
    if st.session_state.single_model_trained:
        node_idx_inspect = st.number_input("Enter Node ID to Inspect:", 0, data.num_nodes - 1, 0, 1, key="inspect_node")
        hops = st.slider("Hops (k):", 1, 5, 2, key="inspect_hops")
        fig = plot_interactive_neighborhood(data, st.session_state.single_model_trained, node_idx_inspect, hops)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train a model to enable neighborhood inspection.")

with tab3:
    st.header("Compare Multiple Experimental Runs")
    with st.form("experiment_form"):
        st.subheader("Define an Experiment")
        c1, c2, c3, c4 = st.columns(4)
        exp_model = c1.selectbox("Model", ("GCN", "GAT", "GraphSAGE"), key="exp_model")
        exp_hidden = c2.slider("Hidden Channels", 16, 128, 32, key="exp_hidden")
        exp_lr = c3.number_input("Learning Rate", 0.001, 0.1, 0.01, format="%.3f", key="exp_lr")
        exp_epochs = c4.number_input("Epochs", 10, 200, 50, key="exp_epochs")
        if st.form_submit_button("Add to Queue"):
            run_name = f"{exp_model}_H{exp_hidden}_LR{exp_lr}"
            st.session_state.experiment_queue.append({"name": run_name, "model": exp_model, "hidden": exp_hidden, "lr": exp_lr, "epochs": exp_epochs})
            st.success(f"Added '{run_name}' to the queue!")

    if st.session_state.experiment_queue:
        st.subheader("Experiment Queue")
        st.table(pd.DataFrame(st.session_state.experiment_queue))
        c1, c2 = st.columns([1, 5])
        if c1.button("Run All Experiments", type="primary"):
            progress_bar = st.progress(0, text="Starting experiments...")
            results_list, curves_list = [], []
            st.session_state.trained_models.clear()
            st.session_state.run_predictions.clear()
            for i, exp in enumerate(st.session_state.experiment_queue):
                progress_bar.progress((i) / len(st.session_state.experiment_queue), f"Running {i+1}/{len(st.session_state.experiment_queue)}: {exp['name']}")
                model = get_model(exp['model'], num_features, exp['hidden'], num_classes)
                optimizer = torch.optim.Adam(model.parameters(), lr=exp['lr'])
                criterion = torch.nn.CrossEntropyLoss()
                for epoch in range(1, exp['epochs'] + 1):
                    train_step(model, data, optimizer, criterion)
                    val_acc, _, _, _, _ = evaluate(model, data, data.val_mask)
                    curves_list.append({"Run": exp['name'], "Epoch": epoch, "Validation Accuracy": val_acc})
                test_acc, f1, roc_auc, y_true, y_pred = evaluate(model, data, data.test_mask)
                results_list.append({"Run Name": exp['name'], "Model": exp['model'], "F1-Score": f1, "ROC-AUC": roc_auc, "Test Accuracy": test_acc})
                st.session_state.trained_models[exp['name']] = model
                st.session_state.run_predictions[exp['name']] = {'true': y_true, 'pred': y_pred}
            progress_bar.progress(1.0, "All experiments complete!")
            time.sleep(1)
            progress_bar.empty()
            st.session_state.run_results = {
                "dataset_name": st.session_state.dataset_name,
                "metrics": pd.DataFrame(results_list),
                "curves": pd.DataFrame(curves_list)
            }
            st.session_state.experiment_queue = []
            st.success(f"All experiments on the {st.session_state.dataset_name} dataset are complete!")
        if c2.button("Clear Queue"):
            st.session_state.experiment_queue = []
            st.rerun()

    if st.session_state.run_results:
        st.subheader(f"Comparison Dashboard for {st.session_state.run_results['dataset_name']}")
        dash_tab1, dash_tab2, dash_tab3 = st.tabs(["Metric Comparison", "Training Curves", "Detailed Analysis"])
        with dash_tab1:
            st.dataframe(st.session_state.run_results["metrics"].set_index("Run Name"))
        with dash_tab2:
            st.plotly_chart(plot_training_curves(st.session_state.run_results["curves"]), use_container_width=True)
        with dash_tab3:
            selected_run = st.selectbox("Select run for detailed analysis:", list(st.session_state.trained_models.keys()))
            if selected_run:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("t-SNE Embeddings")
                    model_to_plot = st.session_state.trained_models[selected_run]
                    tsne_df = get_tsne_embeddings(model_to_plot, data)
                    fig = plot_tsne(tsne_df)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.subheader("Confusion Matrix")
                    preds = st.session_state.run_predictions[selected_run]
                    st.pyplot(plot_confusion_matrix(preds['true'], preds['pred'], num_classes))

with tab4:
    st.header("Automated Hyperparameter Tuning with Optuna")

    def objective(trial):
        model_name = trial.suggest_categorical("model", ["GCN", "GAT", "GraphSAGE"])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_channels = trial.suggest_categorical("hidden_channels", [16, 32, 64, 128])
        
        model = get_model(model_name, num_features, hidden_channels, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(50):
            train_step(model, data, optimizer, criterion)
        
        val_acc, _, _, _, _ = evaluate(model, data, data.val_mask)
        return val_acc

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Tuning Configuration")
        n_trials = st.number_input("Number of trials to run:", min_value=5, max_value=100, value=20, step=5)
        if st.button("Start Tuning", type="primary"):
            with st.spinner("Running hyperparameter optimization... This may take a while."):
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=n_trials)
                st.session_state.optuna_study = study
                st.success("Optimization finished!")

    with c2:
        st.subheader("Tuning Results")
        if st.session_state.optuna_study:
            study = st.session_state.optuna_study
            st.write("Best trial:")
            st.write(f"  Value: {study.best_value:.4f}")
            st.write("  Params: ")
            for key, value in study.best_params.items():
                st.write(f"    {key}: {value}")
            
            st.plotly_chart(plot_optimization_history(study), use_container_width=True)
            st.plotly_chart(plot_param_importances(study), use_container_width=True)
        else:
            st.info("Run a tuning study to see results.")
