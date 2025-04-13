import torch
import pickle
import os
import numpy as np

import plotly
import plotly.graph_objects as go

## Uses the stats at final retrieval to estimate the mean union size
def get_threshold_est(b_values:list,data)->dict:
    mean_threshold = {}
    for b in b_values:
    # for b in (10,):
        try:
            threshold_file = f"{'/mnt/nas/kartikn/CMUVERA_headless/' if data=='scifact' else '/mnt/elk/data/kartikn/CMUVERA//'}colbert_aug_n2_d128_rh8_threshold1_{data}_b{b}.pkl"
            threshold_data = load(threshold_file)
        except:
            continue
        # for threshold_values in (1,2,3,4):
        for threshold_values in (1,):            
            merged_buckets = []
            for key in range(len(threshold_data)):
                merged_buckets.append(len([ elem for elem, count in threshold_data[key].items() if count >= threshold_values]))
            mean_threshold[b,threshold_values] = np.mean(merged_buckets).item()
    return mean_threshold

def plot_b_vs_metric(dataset,k_values,b_values):
    
    mean_threshold = get_threshold_est(b_values,dataset)
    for k in k_values:
        tx = lambda t,b: f"pickles/results/BERT/colbertv2-plaid/dblnorm_aug_n2_d128_rh8_threshold{t}_{data}_k15_rerankperh{b}.pkl"

        baseline = f"pickles/results/greedy_base_0_128_k{15}_{data}_bf.pkl"
        
        topk = f"pickles/results/BERT/colbertv2-plaid/norm_base_n2_d128_{data}_k15.pkl"
        
        baseline_indices, baseline_scores = load_from_baseline(baseline)
        baseline_indices = [set(x.tolist()) for x in baseline_indices[:,:k]]
        baseline_scores = baseline_scores[:,k-1]
        
        topk_indices, topk_scores = load_from_file(topk)
        topk_indices = [set(x.tolist()) for x in topk_indices[:,:k]]
        topk_scores = topk_scores[:,k-1]
        
        topk_score_line = go.Scatter(
            x=[0, 800], 
            y=[topk_scores.mean().item(), topk_scores.mean().item()], 
            mode='lines+text', 
            name=f'Top-k', 
            line=dict(color='blue', width=2, dash='dash'),
            text=[f"{topk_scores.mean().item():.2f}", f"{topk_scores.mean().item():.2f}"],
            textposition="top right"
        )
        
        baseline_score_line = go.Scatter(
            x=[0, 800], 
            y=[baseline_scores.mean().item(), baseline_scores.mean().item()], 
            mode='lines+text', 
            name=f'S*{k}', 
            line=dict(color='red', width=2, dash='dash'),
            text=[f"{baseline_scores.mean().item():.2f}", f"{baseline_scores.mean().item():.2f}"],
            textposition="top right"
        )
        
        # line for Jaccard similarity between topk and baseline
        jaccard_similarity = np.mean([
            len(baseline_indices[i].intersection(topk_indices[i])) / len(baseline_indices[i].union(topk_indices[i]))
            for i in range(len(baseline_indices))
        ])
        jaccard_line = go.Scatter(
            x=[0, 800],
            y=[jaccard_similarity, jaccard_similarity],
            mode='lines+text',
            name=f'Jaccard Top-k vs S*{k}',
            line=dict(color='green', width=2, dash='dot'),
            text=[f"{jaccard_similarity:.2f}", f"{jaccard_similarity:.2f}"],
            textposition="top right"
        )

        # line for intersection size between topk and baseline
        intersection_size = np.mean([
            len(baseline_indices[i].intersection(topk_indices[i]))
            for i in range(len(baseline_indices))
        ])
        intersection_line = go.Scatter(
            x=[0, 800],
            y=[intersection_size, intersection_size],
            mode='lines+text',
            name=f'Intersection Top-k vs S*{k}',
            line=dict(color='purple', width=2, dash='dot'),
            text=[f"{intersection_size:.2f}", f"{intersection_size:.2f}"],
            textposition="top right"
        )
        
        
            
        
        score_figs = go.Figure()
        jaccard_figs = go.Figure()
        intersection_figs = go.Figure()
        jaccard_figs.add_trace(jaccard_line)
        intersection_figs.add_trace(intersection_line)
        score_figs.add_trace(baseline_score_line)
        score_figs.add_trace(topk_score_line)
        
        # Add horizontal lines for greedy submodlib methods
        for method, color in zip(
            ["NaiveGreedy", "LazyGreedy", "LazierThanLazyGreedy"],
            ["orange", "brown", "pink"]
        ):

            submodlib_file = f"pickles/results/greedy_submodlib_{method}_k15_{data}_bf_k15.pkl"
            submodlib_indices, submodlib_scores = load_from_file(submodlib_file)
            submodlib_mean_score = submodlib_scores.mean(dim=0).numpy()[-1]
            submodlib_line = go.Scatter(
                x=[0, 800],
                y=[submodlib_mean_score, submodlib_mean_score],
                mode='lines+text',
                name=f'{method}',
                line=dict(color=color, width=2, dash='dash'),
                text=[f"{submodlib_mean_score:.2f}", f"{submodlib_mean_score:.2f}"],
                textposition="top right"
            )
            score_figs.add_trace(submodlib_line)
            
            
            submodlib_indices = [set(x.tolist()) for x in submodlib_indices[:, :k]]

            submodlib_jaccard = np.mean([
                len(baseline_indices[i].intersection(submodlib_indices[i])) / len(baseline_indices[i].union(submodlib_indices[i]))
                for i in range(len(baseline_indices))
            ])
            submodlib_intersection = np.mean([
                len(baseline_indices[i].intersection(submodlib_indices[i]))
                for i in range(len(baseline_indices))
            ])

            # Add Jaccard similarity line for submodlib methods
            submodlib_jaccard_line = go.Scatter(
                x=[0, 800],
                y=[submodlib_jaccard, submodlib_jaccard],
                mode='lines+text',
                name=f'{method} Jaccard',
                line=dict(color=color, width=2, dash='dot'),
                text=[f"{submodlib_jaccard:.2f}", f"{submodlib_jaccard:.2f}"],
                textposition="top right"
            )
            jaccard_figs.add_trace(submodlib_jaccard_line)

            # Add intersection size line for submodlib methods
            submodlib_intersection_line = go.Scatter(
                x=[0, 800],
                y=[submodlib_intersection, submodlib_intersection],
                mode='lines+text',
                name=f'{method} Intersection',
                line=dict(color=color, width=2, dash='dot'),
                text=[f"{submodlib_intersection:.2f}", f"{submodlib_intersection:.2f}"],
                textposition="top right"
            )
            intersection_figs.add_trace(submodlib_intersection_line)

        
        
        stoch = lambda bsz: f"pickles/results/greedy_base_{bsz}_128_k15_{data}_bf.pkl"
        b_vals = [50,100,200,400,800,1000]
        real_scores = []
        b_real = []
        real_jaccards = []
        real_intersections = []
        for bs in b_vals:
            try:
                inds,scores = load_from_baseline(stoch(bs))
                real_scores.append((scores[:,k-1]).mean().item())
                inds = [set(x.tolist()) for x in inds[:,:k]]
                real_jaccards.append(np.mean([len(baseline_indices[i].intersection(inds[i]))/len(baseline_indices[i].union(inds[i])) for i in range(len(baseline_indices))]))
                real_intersections.append(np.mean([len(baseline_indices[i].intersection(inds[i])) for i in range(len(baseline_indices))]))
                b_real.append(bs)
            except:
                continue
        score_figs.add_trace(go.Scatter(x=b_real,y=real_scores,mode="lines+markers",name=f"Stochastic"))
        jaccard_figs.add_trace(go.Scatter(x=b_real,y=real_jaccards,mode="lines+markers",name=f"Stochastic"))
        intersection_figs.add_trace(go.Scatter(x=b_real,y=real_intersections,mode="lines+markers",name=f"Stochastic"))
        
        thresholds=(1,)
        for t in thresholds:
            
            y_val = []
            y_jac = []
            y_int = []
            b_values_real = []
            for b in b_values:
                try:
                    tx_ind, tx_scores = load_from_file(tx(t,b))
                except FileNotFoundError:
                    continue
                tx_ind = [set(x.tolist()) for x in tx_ind[:,:k]]
                tx_scores = tx_scores[:,k-1]
                y_val.append(tx_scores.mean().item())
                y_jac.append(np.mean([len(baseline_indices[i].intersection(tx_ind[i]))/len(baseline_indices[i].union(tx_ind[i])) for i in range(len(baseline_indices))]))
                y_int.append(np.mean([len(baseline_indices[i].intersection(tx_ind[i])) for i in range(len(baseline_indices))]))
                b_values_real.append(b)
                
                
            x_axis = [mean_threshold[b,t] for b in b_values_real]
            score_figs.add_trace(go.Scatter(x=x_axis, y=y_val, mode='lines+markers', name=f"Threshold {t}"))
            jaccard_figs.add_trace(go.Scatter(x=x_axis, y=y_jac, mode='lines+markers', name=f"Threshold {t}"))
            intersection_figs.add_trace(go.Scatter(x=x_axis, y=y_int, mode='lines+markers', name=f"Threshold {t}"))
        score_figs.update_layout(
            title=f"{data}  : Mean Scores at (k={k}) for Different Thresholds",
            xaxis_title="Effective Bucket Size",
            yaxis_title="Mean Score",
            legend_title="Threshold"
        )
        jaccard_figs.update_layout(
            title=f"{data}  Similarity at (k={k}) for Different Thresholds",
            xaxis_title="Effective Bucket Size",
            yaxis_title="Jaccard Similarity",
            legend_title="Threshold"
        )
        intersection_figs.update_layout(
            title=f"{data}  : Intersection Size at (k={k}) for Different Thresholds",
            xaxis_title="Effective Bucket Size",
            yaxis_title="Intersection Size",
            legend_title="Threshold"
        )
        score_figs.write_html(f"plots/html/bvF_{data}_dblnorm_k{k}_{'.'.join(map(str, b_values))}.html")
        score_figs.write_image(f"plots/images/bvF_{data}_dblnorm_k{k}_{'.'.join(map(str, b_values))}.jpeg")
        jaccard_figs.write_html(f"plots/html/bvJ_{data}_dblnorm_k{k}_{'.'.join(map(str, b_values))}.html")
        jaccard_figs.write_image(f"plots/images/bvJ_{data}_dblnorm_k{k}_{'.'.join(map(str, b_values))}.jpeg")
        intersection_figs.write_html(f"plots/html/bvI_{data}_dblnorm_k{k}_{'.'.join(map(str, b_values))}.html")
        intersection_figs.write_image(f"plots/images/bvI_{data}_dblnorm_k{k}_{'.'.join(map(str, b_values))}.jpeg")
        

def plot_k_vs_metric(dataset,k,bsizes=[10,25,50,100,200]):
    score_plot = go.Figure()
    for method in "LazyGreedy", "LazierThanLazyGreedy", "NaiveGreedy":
        inds, scores = pickle.load(open(f"pickles/results/greedy_submodlib_{method}_k{k}_{dataset}_bf_k{k}.pkl", "rb"))
        score_plot.add_trace(
            go.Scatter(
                x=np.arange(1, len(scores) + 1),
                y=scores.mean(dim=0).numpy(),
                mode="lines+markers",
                name=method,
                line=dict(width=1),
            )
        )
    exact_inds, exact_scores = load_from_baseline(f"pickles/results/greedy_base_0_128_k{k}_{dataset}_bf.pkl")
    
    score_plot.add_trace(
        go.Scatter(
            x=np.arange(1, len(exact_scores) + 1),
            y=exact_scores.mean(dim=0).numpy(),
            mode="lines+markers",
            name="Exact",
            line=dict(width=1),
        )
    )
    
    for bsize in bsizes:
        aug_inds, aug_scores = load(f"pickles/results/BERT/colbertv2-plaid/dblnorm_aug_n2_d128_rh8_threshold1_{dataset}_k{k}_rerankperh{bsize}.pkl")
        score_plot.add_trace(
            go.Scatter(
                x=np.arange(1, len(aug_scores) + 1),
                y=aug_scores.mean(dim=0).numpy(),
                mode="lines+markers",
                name=f"Augmented - {bsize} per table",
                line=dict(width=1),
            )
        )
    score_plot.update_layout(
        title=f"Score Plot for {dataset}",
        xaxis_title="k",
        yaxis_title="F(S)",
    )
    score_plot.write_html(f"plots/html/kvF_{dataset}_dblnorm_{'.'.join(list(map(str,bsizes)))}.html")
    score_plot.write_image(f"plots/images/kvF_{dataset}_dblnorm_{'.'.join(list(map(str,bsizes)))}.jpeg")


###### 
def load_from_baseline(fname):
    with open(fname, 'rb') as f:
        p = pickle.load(f)
        p = torch.tensor(p)
        indices = p[:,:,0]
        scores = p[:,:,1]
        return indices.to(torch.int64),scores
    
def load(fname):
    with open(fname, 'rb') as f:
        p = pickle.load(f)
        return p
    
def load_from_file(filename):
    with open(filename, 'rb') as f:
        indices,scores = pickle.load(f)
        return indices.to(device="cpu",dtype=torch.int64),scores.cpu()
######


if __name__ == "__main__":
    os.makedirs("plots/html",exist_ok=True)
    os.makedirs("plots/images",exist_ok=True)   
    # Load the data
    datasets = ["nfcorpus","scifact"]
    b_sizes_aug = [10,25,50,100,200]
    k_values=[5,10,15]
    max_k = 15
    for data in datasets:
        plot_k_vs_metric(data,max_k,b_sizes_aug)
        plot_b_vs_metric(data,k_values,b_sizes_aug)