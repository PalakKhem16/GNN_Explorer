# data_loader.py
import streamlit as st
from torch_geometric.datasets import Planetoid

@st.cache_data(persist=True)
def load_dataset(name):
    """
    Loads one of the Planetoid datasets (Cora, Citeseer, PubMed).
    Uses Streamlit's caching to prevent reloading the data on every interaction.
    """
    # This will download the dataset to a 'data' directory in your project folder
    dataset = Planetoid(root='data', name=name)
    return dataset[0] # Return the single graph object from the dataset
