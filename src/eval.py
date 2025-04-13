import torch
import numpy as np
import os
from .utils import load
import plotly
from plotly import graph_objects as go
import matplotlib.pyplot as plt

## DO NOT LOOK HERE. LOOK AT INDIVIDUAL FUNCTIONS IN TESTING

## TODO: i think we can convert to numpy/tensor and use unique to simulate set
# baseline format: list[list[tuple]], second elem is gain, first is id
def get_baseline_ids(file_name): # results of submodlib
    res = load(file_name)
    return [[x[0] for x in l] for l in res]

def get_result_ids(file_name):
    return (load(file_name)).cpu().tolist()

def q_set_count(dataset, emb='bert_bert-base-uncased'):
    qmask_path = f"./pickles/embeddings/mask_{emb}_{dataset}.pkl"
    qmask = load(qmask_path)[1]
    return torch.sum(qmask, dim=1).cpu().tolist()

def jacard_sim(ids_1, ids_2):
    return len(set(ids_1).intersection(set(ids_2))) / len(set(ids_1).union(set(ids_2)))

def plot_jacard_sim_for_exact_cmp(dataset_name, kvals, wvals, Q_filter=None):
    filename = lambda w, k: f"./pickles/results/greedy_muvera_w{w}_mbase_k{k}_d{1538*(2**k)*2*w}_fec_{dataset_name}_bf.pkl"
    baseline = f"./pickles/results/greedy_base_{dataset_name}_bf.pkl"
    set_counts = q_set_count(dataset_name)
    traces = {}
    base = get_baseline_ids(baseline)
    b_vals = 1, 2, 3, 5, 10, 15, 20, 25
    fig = go.Figure()
    for w in wvals:
        for k in kvals:
            if not os.path.exists(filename(w, k)):
                print(f"Skipping {filename(w, k)}")
                continue
            res = get_result_ids(filename(w, k))
            vals = [[] for _ in b_vals]
            for q_ct, (base_id, res_id) in zip(set_counts, zip(base, res)):
                if Q_filter and q_ct <= Q_filter:
                    continue
                
                for b, t in zip(b_vals, vals):
                    t.append(jacard_sim(base_id[:b], res_id[:b]))
            avg_vals = [np.mean(t) for t in vals]
            trace = go.Scatter(
                x=b_vals,
                y=avg_vals,
                mode='lines+markers',
                name=f'w={w}, k={k}',
                visible='legendonly'
            )
            traces[f'w={w}, k={k}'] = trace
            fig.add_trace(trace)

    fig.update_layout(
        title=f'Jaccard Similarity grouped by w for {dataset_name}',
        xaxis_title='b values',
        yaxis_title='Jaccard Similarity'
    )

    return fig

def plot_jacard_sim_for_pm(dataset_name, kvals, Q_filter=None):
    filename = lambda k: f"./pickles/results/greedy_pm_mbase_k{k}_d{1538*(2**(k+1))}_fec_{dataset_name}_bf.pkl"
    baseline = f"./pickles/results/greedy_base_{dataset_name}_bf.pkl"
    set_counts = q_set_count(dataset_name)
    traces = {}
    base = get_baseline_ids(baseline)
    b_vals = 1, 2, 3, 5, 10, 15, 20, 25
    fig = go.Figure()

    for k in kvals:
        if not os.path.exists(filename(k)):
            print(f"Skipping {filename(k)}")
            continue
        res = get_result_ids(filename(k))
        vals = [[] for _ in b_vals]
        for q_ct, (base_id, res_id) in zip(set_counts, zip(base, res)):
            if Q_filter and q_ct <= Q_filter:
                continue
            
            for b, t in zip(b_vals, vals):
                t.append(jacard_sim(base_id[:b], res_id[:b]))
        avg_vals = [np.mean(t) for t in vals]
        trace = go.Scatter(
            x=b_vals,
            y=avg_vals,
            mode='lines+markers',
            name=f'k={k}',
            visible='legendonly'
        )
        traces[f'k={k}'] = trace
        fig.add_trace(trace)

    fig.update_layout(
        title=f'Jaccard Similarity grouped by w for {dataset_name}',
        xaxis_title='b values',
        yaxis_title='Jaccard Similarity'
    )

    return fig

if __name__ == "__main__":
    os.makedirs("./plots", exist_ok=True)
    dataset_name = "scifact"
    kvals = [4, 6, 8, 10, 12, 14]
    wvals = [4, 8, 12, 16, 20]
    fig_all = plot_jacard_sim_for_exact_cmp(dataset_name, kvals, wvals)
    plotly.io.write_html(fig_all, f"./plots/jaccard_sim_{dataset_name}_all.html")
    fig_above_mean = plot_jacard_sim_for_exact_cmp(dataset_name, kvals, wvals, Q_filter=20)
    plotly.io.write_html(fig_above_mean, f"./plots/jacard_sim_{dataset_name}_above_{20}.html")
    
    # fig_all_pm = plot_jacard_sim_for_pm(dataset_name, kvals)
    # plotly.io.write_html(fig_all_pm, f"./plots/pm_jaccard_sim_{dataset_name}_all.html")
    # fig_above_mean_pm = plot_jacard_sim_for_pm(dataset_name, kvals, Q_filter=20)
    # plotly.io.write_html(fig_above_mean_pm, f"./plots/pm_jacard_sim_{dataset_name}_above_{20}.html")
    print("Done")