import json
import os
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

DATA_PATH = "data/"
OUTPUT_HEATMAP_PATH = os.path.join(DATA_PATH, "6_tpl_fast_result", "function_embeddings_heatmap.png")

def plot_embeddings():
    target_emb_path = os.path.join(DATA_PATH, "4_embedding/", "target_in9_embedding.json")
    candidate_emb_path = os.path.join(DATA_PATH, "4_embedding/", "candidate_in9_embedding.json")
    
    binaries, results = read_embeddings_files(target_emb_path, candidate_emb_path)
    n = len(binaries)
    flat_scores = [v for row in results for v in row]
    vmax = max(flat_scores) if flat_scores else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig_size = max(8, min(24, 4 + 0.35 * n))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(results, cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest")

    ax.set_title("Cosine Similarity of Average Function Embeddings of Target/Candidate files")
    ax.set_xlabel("Candidate Binary")
    ax.set_ylabel("Object Binary")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(binaries, rotation=90, fontsize=7)
    ax.set_yticklabels(binaries, fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average Similarity Score")

    fig.tight_layout()
    fig.savefig(OUTPUT_HEATMAP_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return

def read_embeddings_files(target_emb_filepath, candidate_emb_filepath):
    all_target_embeddings = []
    all_candidate_embeddings = []
    with open(target_emb_filepath, "r") as f:
        all_target_embeddings = json.load(f)
    with open(candidate_emb_filepath, "r") as f:
        all_candidate_embeddings = json.load(f)
    sim_scores, binaries = compare_all_embeddings(
        all_target_embeddings, 
        all_candidate_embeddings
        )
    return sim_scores, binaries

def read_embeddings_from_json_object(object):
    file_map = dict()
    binaries = []
    for file_function_key in object:
        arr = file_function_key.split("|||")
        target_filename = arr[0]
        target_function = arr[1]
        embedding = object[file_function_key]
        if target_filename not in file_map:
            file_map[target_filename] = {target_function: embedding}
        else:
            file_map[target_filename][target_function] = embedding

        binaries.append(target_filename)
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
        if avg_target_emb  == None:
            avg_target_emb = cur_emb
        else:
            avg_target_emb += cur_emb
    
    avg_candidate_emb = None
    for candidate_func in candidate_file_embeddings:
        cur_emb = np.array(target_file_embeddings[candidate_func])
        if avg_candidate_emb == None:
            avg_candidate_emb = cur_emb
        else:
            avg_candidate_emb += cur_emb

    avg_target_emb /= len(target_file_embeddings.keys())
    avg_candidate_emb /= len(candidate_file_embeddings.keys())

    # Cosine similarity
    similarity = np.dot(avg_target_emb, avg_candidate_emb) / norm(avg_target_emb) * norm(avg_candidate_emb)

    return similarity

plot_embeddings()