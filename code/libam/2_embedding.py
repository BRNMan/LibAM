import sys, os
import argparse
import shutil
import tempfile
# import click
from settings import *
sys.path.append("code/anchor_detection/semantic_anchor_detection")
sys.path.append("code/binary_preprocess")
sys.path.append("code/embeddings_generate")
sys.path.append("code/anchor_reinforcement/anchor_alignment")
sys.path.append("code/reuse_area_exploration/Embeded-GNN")
sys.path.append("code/reuse_area_exploration/TPL_detection")
sys.path.append("code/reuse_area_exploration/reuse_area_detection")


import Generate_func_embedding as embeddings_generate_module


def _resolve_feature_filename(feature_dir, selected_name):
    if selected_name.endswith(".json"):
        candidate = selected_name
    else:
        candidate = selected_name + ".json"

    full_path = os.path.join(feature_dir, candidate)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            "Feature file not found: {} (looked in {})".format(candidate, feature_dir)
        )
    return candidate


def _prepare_feature_input(feature_dir, selected_name):
    if not selected_name:
        return feature_dir, None

    selected_file = _resolve_feature_filename(feature_dir, selected_name)
    temp_dir = tempfile.mkdtemp(prefix="libam_feature_subset_")
    shutil.copy2(os.path.join(feature_dir, selected_file), os.path.join(temp_dir, selected_file))
    return temp_dir, temp_dir


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate function embeddings, AFCGs, and subgraphs."
    )
    parser.add_argument(
        "--target-feature",
        default=None,
        help="Optional single target feature file name (with or without .json).",
    )
    parser.add_argument(
        "--candidate-feature",
        default=None,
        help="Optional single candidate feature file name (with or without .json).",
    )
    # In notebooks (e.g., Colab), the kernel injects extra argv like "-f <kernel.json>".
    # Use parse_known_args so script options still work under exec(...).
    args, _ = parser.parse_known_args()
    return args



def cli():
    args = _parse_args()
    print("hello libAE")

    if (args.target_feature is None) != (args.candidate_feature is None):
        raise ValueError(
            "Please provide both --target-feature and --candidate-feature, or neither."
        )

    target_feature_dir = DATA_PATH + "2_target/feature"
    candidate_feature_dir = DATA_PATH + "3_candidate/feature"

    target_input_dir, target_tmp_dir = _prepare_feature_input(
        target_feature_dir, args.target_feature
    )
    candidate_input_dir, candidate_tmp_dir = _prepare_feature_input(
        candidate_feature_dir, args.candidate_feature
    )
    
    # 1. get feature and fcg
    # print("start bianry preprocess......")
    # binary_preprocess_module.getAllFiles(DATA_PATH+"2_target/timecost", DATA_PATH+"1_binary/target", DATA_PATH+"2_target/", mode="1")
    # binary_preprocess_module.getAllFiles(DATA_PATH+"3_candidate/timecost", DATA_PATH + "1_binary/candidate", DATA_PATH + "3_candidate/", mode="1")
    
    # # # 2. get embedding
    print("start embeding generation......")
    try:
        embeddings_generate_module.function_embedding(DATA_PATH+"4_embedding/timecost",
                                                    target_input_dir,
                                                    DATA_PATH+"4_embedding/target_in9_bl5_embedding.json",
                                                    model_path=WORK_PATH + "/code/embeddings_generate/gnn-best.pt")
        embeddings_generate_module.function_embedding(DATA_PATH+"4_embedding/timecost",
                                                    candidate_input_dir,
                                                    DATA_PATH+"4_embedding/candidate_in9_bl5_embedding.json",
                                                    model_path=WORK_PATH + "/code/embeddings_generate/gnn-best.pt")

        embeddings_generate_module.generate_afcg(DATA_PATH+"4_embedding/tar_afcg",
                                                os.path.join(DATA_PATH, "2_target/fcg"), 
                                                DATA_PATH+"4_embedding/target_in9_embedding.json",
                                                model_path=os.path.join(WORK_PATH, "code/reuse_area_exploration/Embeded-GNN/fcg_gnn-best-0.01.pt"))
        
        embeddings_generate_module.generate_subgraph(DATA_PATH+"4_embedding/tar_subgraph",
                                                os.path.join(DATA_PATH, "2_target/fcg"), 
                                                DATA_PATH+"4_embedding/target_in9_embedding.json",
                                                model_path=os.path.join(WORK_PATH, "code/reuse_area_exploration/Embeded-GNN/fcg_gnn-best-0.01.pt"))
    
        embeddings_generate_module.generate_afcg(DATA_PATH+"4_embedding/cdd_afcg",
                                                os.path.join(DATA_PATH, "3_candidate/fcg"), 
                                                DATA_PATH+"4_embedding/candidate_in9_embedding.json",
                                                model_path=os.path.join(WORK_PATH, "code/reuse_area_exploration/Embeded-GNN/fcg_gnn-best-0.01.pt"))
        
        embeddings_generate_module.generate_subgraph(DATA_PATH+"4_embedding/cdd_subgraph",
                                                os.path.join(DATA_PATH, "3_candidate/fcg"), 
                                                DATA_PATH+"4_embedding/candidate_in9_embedding.json",
                                                model_path=os.path.join(WORK_PATH, "code/reuse_area_exploration/Embeded-GNN/fcg_gnn-best-0.01.pt"))
    finally:
        if target_tmp_dir is not None and os.path.exists(target_tmp_dir):
            shutil.rmtree(target_tmp_dir)
        if candidate_tmp_dir is not None and os.path.exists(candidate_tmp_dir):
            shutil.rmtree(candidate_tmp_dir)
   

if __name__ == "__main__":
    cli()
