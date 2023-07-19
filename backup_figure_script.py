# %% [markdown]
# # Imports

# %%



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import streamlit as st
import numpy as np
import pandas as pd
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
from pathlib import Path
# import plotly.express as px
from torch.utils.data import DataLoader
from typing import Union, List, Optional, Callable, Tuple, Dict, Literal, Set
from jaxtyping import Float, Int
from functools import partial


# %%
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# %% [markdown]
# # Model, Dataset, Helper Functions
neg_m_heads = [(10,7), (11,10)]
name_mover_heads = [(9,9), (9,6), (10,0)]
backup_heads = [(9,0), (9,7), (10,1), (10,2), (10,6), (10,10), (11,2), (11,9)]
key_backup_heads = [(10,2), (10,10), (11,2)]
strong_neg_backup_heads = [(11,2), (10,2), (11,6)]



head_names = ["Negative", "Name Mover", "Backup"]
head_list = [neg_m_heads, name_mover_heads, backup_heads]

# %%
import pickle

results_storage = {}

# to load dictionary (if you dont have, simply comment out the below lines):
with open('saved_dictionary.pkl', 'rb') as f:
    results_storage = pickle.load(f)

# %%
def return_head_from_string(head_type):
    
    assert head_type == "backup" or head_type == "NMH" or head_type == "negative" or head_type == "key_backup"

    if head_type == "backup":
        heads = backup_heads
    elif head_type == "NMH":
        heads = name_mover_heads
    elif head_type == "negative":
        heads = neg_m_heads
    elif head_type == "key_backup":
        heads = key_backup_heads

    return heads

    

# %%
def run_query_intervention_with_setup(scaling_amounts, head_type, scores = True,  freeze_ln = False, only_S1 = True, head_to_control = (9,9)):
    """
    currently doesnt allow for ablations in encoding, nor does it allow for post_ln
    """
    encode_name = ("1" if freeze_ln else "0") +  ("1" if only_S1 else "0") + ("1" if scores else "0") + head_type + str(head_to_control)

    if encode_name in results_storage.keys():
        #print("returning cached value")
        return results_storage[encode_name]
    


    matrix = model.W_O[head_to_control[0], head_to_control[1]]
    layer_output = clean_cache[utils.get_act_name("z", head_to_control[0])]
    layer_result = einops.einsum(matrix, layer_output, "d_head d_model, batch seq h_idx d_head -> batch seq h_idx d_model")
    output_of_head = layer_result[:, -1, head_to_control[1], :]     #[:, 10, :]  # 10 = Subject,   4 = IO

    dot_product = einops.einsum(output_of_head, IO_unembed_direction, "batch d_model, batch d_model -> batch")
    length_of_vector = einops.einsum(IO_unembed_direction, IO_unembed_direction, "batch d_model, batch d_model -> batch")
    projected_lengths = dot_product / length_of_vector
    io_in_direction = IO_unembed_direction * einops.repeat(projected_lengths, "batch -> batch d_model", d_model = model.cfg.d_model)
    everything_but_direction = output_of_head - io_in_direction

    heads = return_head_from_string(head_type)

    into_head_results = get_attn_results_into_head_dirs(heads, output_of_head, scaling_amounts, ablate_heads=[], freeze_ln=freeze_ln, only_S1 = only_S1, scores = scores)
    perp_into_head_results = get_attn_results_into_head_dirs(heads, everything_but_direction, scaling_amounts, ablate_heads=[], freeze_ln=freeze_ln, only_S1 = only_S1, scores = scores)
    parallel_into_head_results = get_attn_results_into_head_dirs(heads, io_in_direction, scaling_amounts, ablate_heads=[], freeze_ln=freeze_ln, only_S1 = only_S1, scores = scores)    


    result = [into_head_results, perp_into_head_results, parallel_into_head_results]
    results_storage[encode_name] = result
    return result



# Define the label-to-variable mapping
label_mapping = {
    "Key Backup Heads": "key_backup",
    "All Backup Heads": "backup",
    "Name Mover Heads": "NMH",
    "Negative Heads": "negative"
}


# Define reverse mapping
reverse_label_mapping = {
    "key_backup": "Key Backup Head",
    "backup": "All Backup Head",
    "NMH": "Name Mover Head",
    "negative": "Negative Head"
}



# %%
def display_attn_scores(head_type, scores = True,  freeze_ln = False, only_S1 = True, head = (9,9)):
    
    nine_nine_into_backup_results, nine_nine_perp_into_backup_results, nine_nine_parallel_into_backup_results = run_query_intervention_with_setup(scaling_amounts, head_type, scores, freeze_ln, only_S1, head_to_control = head)

    heads = return_head_from_string(head_type)


    fig = make_subplots(rows = 3, cols = 5, subplot_titles = ["IO", "S1", "S2","IO - S1", "BOS"], shared_yaxes=True)
    fig.update_layout(height = 900, width = 1200)

    fig.update_layout(title = reverse_label_mapping[head_type] +  f" Attention {'Scores' if scores else 'Probabilities'} from Adding " + str(head) + " Components into Queries " + f"(Freeze Layernorm = {freeze_ln})")
    colors = [
        "pink", "darkviolet", "blue", "purple", "turquoise", "red", "green", "yellow", "orange", "cyan", "magenta",
        "lime", "maroon", "navy", "olive", "teal", "aqua", "silver", "gray", "black", "white", "indigo", "gold", "brown",
        "coral", "crimson", "darkblue", "darkcyan", "darkgoldenrod", "darkgray", "darkgreen", "darkkhaki", "darkmagenta",
        "darkolivegreen", "darkorange", "darkorchid", "darkred", "darksalmon", "darkslateblue", "darkslategray",
        "darkturquoise", "darkviolet", "deeppink", "deepskyblue", "dimgray", "dodgerblue", "firebrick", "forestgreen",
        "fuchsia", "gainsboro", "gold", "goldenrod", "gray", "greenyellow", "honeydew", "hotpink", "indianred", "indigo",
        "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon", "lightblue", "lightcoral",
        "lightcyan", "lightgoldenrodyellow", "lightgray", "lightgreen", "lightpink", "lightsalmon", "lightseagreen",
        "lightskyblue", "lightslategray", "lightsteelblue", "lightyellow", "lime", "limegreen", "linen", "magenta",
        "maroon", "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen", "mediumslateblue",
        "mediumspringgreen", "mediumturquoise", "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin",
        "navajowhite", "navy", "oldlace", "olive", "olivedrab", "orangered", "orchid", "palegoldenrod", "palegreen",
        "paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru", "plum", "powderblue", "rebeccapurple",
        "rosybrown", "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen", "sienna", "skyblue", "slateblue",
        "slategray", "snow", "springgreen", "steelblue", "tan", "thistle", "tomato", "turquoise", "violet", "wheat",
        "whitesmoke", "yellowgreen"
    ]

    for col, data in enumerate(nine_nine_into_backup_results):
        for i in range(len(data)):
            fig.add_trace(go.Scatter(x = scaling_amounts, y = data[i], name = str(heads[i]), line_color = colors[i], legendgroup= str(i)), row = 1, col = col + 1)

    for col, data in enumerate(nine_nine_perp_into_backup_results):
        for i in range(len(data)):
            fig.add_trace(go.Scatter(x = scaling_amounts, y = data[i], name = str(heads[i]) + " ⊥ IO", line_color = colors[i], legendgroup= str(i)), row = 2, col = col + 1)

    for col, data in enumerate(nine_nine_parallel_into_backup_results):
        for i in range(len(data)):
            fig.add_trace(go.Scatter(x = scaling_amounts, y = data[i], name = str(heads[i]) + " ∥ IO", line_color = colors[i], legendgroup= str(i)), row = 3, col = col + 1)


    fig.update_xaxes(title = f"Added Scaled {head} Output", row = 1)
    fig.update_xaxes(title = f"Added Scaled {head} ⊥ IO ", row = 2)
    fig.update_xaxes(title = f"Added Scaled {head} ∥ IO ", row = 3)
    
    if scores:
        fig.update_yaxes(title = "Attn Score on token")
    else:
        fig.update_yaxes(title = "Attn Probs on token")
    st.plotly_chart(fig)
   

# %%
results_storage.keys()

# %%
scaling_amounts = torch.linspace(-8, 8, 21)
for freeze_ln in [True, False]:
    for scores in [True, False]:
        for headtype in ["backup", "NMH", "negative", "key_backup"]:
            for head_to_control in [(9,9), (9,6), (10,0)]:
                run_query_intervention_with_setup(scaling_amounts, headtype, scores, freeze_ln, True, head_to_control= head_to_control)

# %%



st.write("## Plots of Attention Scores/Probs against Query Intervention in GPT2-Small")

st.write("*Turn on Wide Mode (Settings -> Wide mode) if this isn't showing fully and is annoying*")

st.write("The following is an intervention in which we take the output of a certain head and add (a scaled version of) it into the query of another one. We can (sort of) use this to detect copy suppression. While we see more confirming results for the Negative Heads, the results for the [Name Mover Heads and Backup Heads](https://arxiv.org/abs/2211.00593) are a bit more nuanced.")

# insert intervention_image.jpeg on streamlit page. make it 400 pixels wide
st.image("intervention_diagram.png", width = 400)

st.write("Note that the three rows are different inputs into the head - the whole head output, the output of the head perpendicular to the IO unembedding, and the output of the head parallel to it.")

# Create a list of display labels
display_labels = list(label_mapping.keys())
# Display the select box
selected_label = st.selectbox('Receiver Heads', display_labels)
# Retrieve the corresponding variable based on the selected label
selected_variable = label_mapping[selected_label]

display_attn_scores(selected_variable, scores = st.checkbox('Use Attention Scores', value = True), freeze_ln = st.checkbox('Freeze LayerNorm', value = False), head=st.selectbox('Head Output to Use', [(9,9), (9,6), (10,0)]))

# %%
