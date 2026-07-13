import json
import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

DATA_PATH = "data/"
OUTPUT_HEATMAP_PATH = 

def plot_one_graph(binaries, results, rankings=False):
    n = len(binaries)
    flat_scores = [v for row in results for v in row]
    vmax = max(flat_scores) if flat_scores else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig_size = max(8, min(24, 4 + 0.35 * n))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(results, cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest")
    title = "Cosine Similarity of Average Function Embeddings of Target/Candidate files"
    if rankings:
        title = "Ranking of Average Function Embedding Cosine Similarity of Target/Candidate files"
    ax.set_title(title)
    ax.set_xlabel("Candidate Binary")
    ax.set_ylabel("Object Binary")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(binaries, rotation=90, fontsize=7)
    ax.set_yticklabels(binaries, fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average Similarity Score")

    fig.tight_layout()
    output_path = os.path.join(DATA_PATH, "function_embeddings_heatmap.png")
    if rankings:
        output_path = os.path.join(DATA_PATH, "function_embeddings_heatmap_ranking.png")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_embeddings():
    target_emb_path = os.path.join(DATA_PATH, "4_embedding/target_in9_embedding.json")
    candidate_emb_path = os.path.join(DATA_PATH, "4_embedding/candidate_in9_embedding.json")
    
    results, binaries = read_embeddings_files(target_emb_path, candidate_emb_path, False)
    plot_one_graph(results, binaries, False)
    results, binaries = read_embeddings_files(target_emb_path, candidate_emb_path, True)
    plot_one_graph(results, binaries, True)
    return

def read_embeddings_files(target_emb_filepath, candidate_emb_filepath, return_rankings=False):
    all_target_embeddings = []
    all_candidate_embeddings = []
    with open(target_emb_filepath, "r") as f:
        all_target_embeddings = json.load(f)
    with open(candidate_emb_filepath, "r") as f:
        all_candidate_embeddings = json.load(f)

    return read_all_embeddings(
        all_target_embeddings,
        all_candidate_embeddings,
        return_rankings=return_rankings,
    )


def rank_sim_scores(sim_scores):
    rankings = []
    for target in sim_scores:
        target_scores = sim_scores[target]
        candidate_keys = list(target_scores.keys())
        sorted_candidates = sorted(candidate_keys, key=lambda c: target_scores[c])
        candidate_to_rank = {
            candidate: rank for rank, candidate in enumerate(sorted_candidates)
        }
        rankings.append([candidate_to_rank[candidate] for candidate in candidate_keys])

    return rankings


def read_all_embeddings(all_target_embeddings, all_candidate_embeddings, return_rankings=False):
    sim_scores, binaries = compare_all_embeddings(
        all_target_embeddings, 
        all_candidate_embeddings
        )

    if return_rankings:
        return rank_sim_scores(sim_scores), binaries

    # Convert dict of scores to array
    score_array = []
    for key in sim_scores:
        target_scores = sim_scores[key]
        score_array.append([])
        for candidate_key in target_scores:
            sim_score = target_scores[candidate_key]
            score_array[-1].append(sim_score)
    
    return score_array, binaries

def read_embeddings_from_json_object(object):
    file_map = dict()
    binaries = []
    target_filename = ""
    for file_function_key in object:
        arr = file_function_key.split("|||")
        target_filename = arr[0]
        target_function = arr[1]
        embedding = object[file_function_key]
        if target_filename not in file_map:
            file_map[target_filename] = {target_function: embedding}
        else:
            file_map[target_filename][target_function] = embedding
    file_map = {k: v for k,v in sorted(file_map.items())}
    print(file_map.keys())
    binaries = list(file_map.keys())
    return file_map, binaries

def compare_all_embeddings(target_embeddings, candidate_embeddings):
    target_map, binaries = read_embeddings_from_json_object(target_embeddings)
    candidate_map, _ = read_embeddings_from_json_object(candidate_embeddings)
    
    sim_scores = dict()
    for target in target_map:
        sim_scores[target] = dict()
        for candidate in candidate_map:
            similarity_score = compare_one_file_embeddings(
                target_map[target], 
                candidate_map[candidate]
                )
            sim_scores[target][candidate] = similarity_score
    
    return sim_scores, binaries
    

        


def compare_one_file_embeddings(target_file_embeddings: dict, candidate_file_embeddings: dict):
    avg_target_emb = None
    for target_func in target_file_embeddings:
        cur_emb = np.array(target_file_embeddings[target_func])
        if avg_target_emb  is None:
            avg_target_emb = cur_emb
        else:
            avg_target_emb += cur_emb
    
    avg_candidate_emb = None
    for candidate_func in candidate_file_embeddings:
        cur_emb = np.array(candidate_file_embeddings[candidate_func])
        if avg_candidate_emb is None:
            avg_candidate_emb = cur_emb
        else:
            avg_candidate_emb += cur_emb

    avg_target_emb /= len(target_file_embeddings.keys())
    avg_candidate_emb /= len(candidate_file_embeddings.keys())
    # Cosine similarity
    similarity = np.dot(avg_target_emb[0], avg_candidate_emb[0]) / (norm(avg_target_emb[0]) * norm(avg_candidate_emb[0]))

    return similarity

plot_embeddings()