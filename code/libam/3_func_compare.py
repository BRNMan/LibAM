import sys, os
import argparse
import json
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


import all_func_compare_isrd as anchor_detection_module


def _normalize_binary_name(name):
    if name is None:
        return None
    return name[:-5] if name.endswith(".json") else name


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run anchor detection on all binaries or a selected target/candidate pair."
    )
    parser.add_argument(
        "--target-binary",
        default=None,
        help="Optional target binary name (with or without .json).",
    )
    parser.add_argument(
        "--candidate-binary",
        default=None,
        help="Optional candidate binary name (with or without .json).",
    )
    # In notebooks (e.g., Colab), the kernel injects extra argv like "-f <kernel.json>".
    # Use parse_known_args so script options still work under exec(...).
    args, _ = parser.parse_known_args()
    return args


def _filter_embedding_json(src_path, dst_path, binary_name):
    with open(src_path, "r") as f:
        payload = json.load(f)

    prefix = binary_name + "|||"
    filtered = {k: v for k, v in payload.items() if k.startswith(prefix)}
    if not filtered:
        raise ValueError("No embeddings found for binary: {} in {}".format(binary_name, src_path))

    with open(dst_path, "w") as f:
        json.dump(filtered, f)


def cli():
    args = _parse_args()
    print("hello libAE")

    if (args.target_binary is None) != (args.candidate_binary is None):
        raise ValueError(
            "Please provide both --target-binary and --candidate-binary, or neither."
        )

    target_emb_path = os.path.join(DATA_PATH, "4_embedding/target_in9_embedding.json")
    candidate_emb_path = os.path.join(DATA_PATH, "4_embedding/candidate_in9_embedding.json")

    tmp_dir = None
    if args.target_binary and args.candidate_binary:
        tmp_dir = tempfile.mkdtemp(prefix="libam_stage3_pair_")
        target_binary = _normalize_binary_name(args.target_binary)
        candidate_binary = _normalize_binary_name(args.candidate_binary)

        target_filtered_path = os.path.join(tmp_dir, "target_in9_embedding.json")
        candidate_filtered_path = os.path.join(tmp_dir, "candidate_in9_embedding.json")
        _filter_embedding_json(target_emb_path, target_filtered_path, target_binary)
        _filter_embedding_json(candidate_emb_path, candidate_filtered_path, candidate_binary)

        target_emb_path = target_filtered_path
        candidate_emb_path = candidate_filtered_path


    # # 3. function_compare
    print("start anchor detection......")
    try:
        anchor_detection_module.func_compare_annoy_fast_multi(target_emb_path,
            candidate_emb_path,
            os.path.join(DATA_PATH, "5_func_compare_result/score"),
            os.path.join(DATA_PATH, "5_func_compare_result/top_scores"),
            os.path.join(DATA_PATH, "5_func_compare_result"),
            os.path.join(DATA_PATH, "5_func_compare_result/embedding_annoy"))
    finally:
        if tmp_dir is not None and os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir)
    
   

if __name__ == "__main__":
    cli()
