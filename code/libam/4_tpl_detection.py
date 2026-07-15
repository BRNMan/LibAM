import sys, os
import argparse
import json
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


import get_tainted_graph as anchor_reinforcement_module
import get_final_result_dict as TPL_detection_module2


def _normalize_binary_name(name):
    if name is None:
        return None
    return name[:-5] if name.endswith(".json") else name


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run TPL detection on all binaries or one selected target/candidate pair."
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


def _build_filtered_score_dir(score_dir, target_binary, candidate_binary):
    input_name = target_binary + "_reuse_func_dict.json"
    input_path = os.path.join(score_dir, input_name)
    if not os.path.exists(input_path):
        raise FileNotFoundError("Target score file not found: {}".format(input_path))

    with open(input_path, "r") as f:
        payload = json.load(f)

    candidate_prefix = "||||" + candidate_binary + "----"
    filtered = {k: v for k, v in payload.items() if candidate_prefix in k}
    if not filtered:
        raise ValueError(
            "No target/candidate anchor pairs found for {} vs {} in {}".format(
                target_binary, candidate_binary, input_path
            )
        )

    temp_dir = tempfile.mkdtemp(prefix="libam_stage4_pair_")
    with open(os.path.join(temp_dir, input_name), "w") as f:
        json.dump(filtered, f)
    return temp_dir



def cli():
    args = _parse_args()
    print("hello libAE")

    if (args.target_binary is None) != (args.candidate_binary is None):
        raise ValueError(
            "Please provide both --target-binary and --candidate-binary, or neither."
        )
    
    # # 4. TPL detection
    # print("start fast TPL detection......")
    save_path = "6_tpl_fast_result/"
    # Default to top-k pruned anchors for faster TPL detection.
    # Override with LIBAM_TPL_SCORE_DIR=score when full recall analysis is needed.
    tpl_score_dir = os.environ.get("LIBAM_TPL_SCORE_DIR", "top_scores")
    score_input_dir = os.path.join(DATA_PATH, "5_func_compare_result", tpl_score_dir)
    tmp_score_dir = None
    if args.target_binary and args.candidate_binary:
        tmp_score_dir = _build_filtered_score_dir(
            score_input_dir,
            _normalize_binary_name(args.target_binary),
            _normalize_binary_name(args.candidate_binary),
        )
        score_input_dir = tmp_score_dir

    try:
        anchor_reinforcement_module.tpl_detection_fast_annoy_simple_with_logging(
                            os.path.join(DATA_PATH, "2_target/fcg"),
                            os.path.join(DATA_PATH, "3_candidate/fcg"), 
                            score_input_dir + "/", 
                            os.path.join(DATA_PATH, save_path+"tpl_fast_result"), 
                            os.path.join(DATA_PATH, save_path+"tpl_fast_area"), 
                            os.path.join(DATA_PATH, save_path+"tpl_fast_time"),
                            os.path.join(DATA_PATH, "4_embedding"),
                            os.path.join(DATA_PATH, save_path+"sim_func_list"),
                            os.path.join(DATA_PATH, "4_embedding/target_in9_bl5_embedding.json"), 
                            os.path.join(DATA_PATH, "4_embedding/candidate_in9_bl5_embedding.json"),
                            os.path.join(WORK_PATH, "code/reuse_area_exploration/Embeded-GNN/fcg_gnn-best-0.01.pt"),
                            DATA_PATH+"4_embedding/tar_afcg",
                            DATA_PATH+"4_embedding/cdd_afcg",
                            DATA_PATH+"4_embedding/tar_subgraph",
                            DATA_PATH+"4_embedding/cdd_subgraph")
        TPL_detection_module2.get_result_json(os.path.join(DATA_PATH, save_path+"tpl_fast_result"), os.path.join(DATA_PATH, save_path+"tpl_fast_result.json"))
    finally:
        if tmp_score_dir is not None and os.path.exists(tmp_score_dir):
            shutil.rmtree(tmp_score_dir)
    # TPL_detection_module3.cal_libae_result(os.path.join(DATA_PATH, save_path+"tpl_fast_result.json"), os.path.join(GT_PATH, "tpl_ground_truth.json"), os.path.join(DATA_PATH, save_path+"TPL_score/"))
    
    
   

if __name__ == "__main__":
    cli()
