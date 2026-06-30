import sys, os
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
import binary_preprocess as binary_preprocess_module
import Generate_func_embedding as embeddings_generate_module
import get_tainted_graph as anchor_reinforcement_module
import fcg_gnn_score as embeded_gnn_module
import get_final_score_multi as TPL_detection_module1
import get_final_result_dict as TPL_detection_module2
import cal_result as TPL_detection_module3
import adjust_area as area_adjustment_module
import compare_area as reuse_area_detection_module


def cli():
    print("hello libAE")
    
    # # 4. TPL detection
    # print("start fast TPL detection......")
    save_path = "6_tpl_fast_result/"
    # Default to top-k pruned anchors for faster TPL detection.
    # Override with LIBAM_TPL_SCORE_DIR=score when full recall analysis is needed.
    tpl_score_dir = os.environ.get("LIBAM_TPL_SCORE_DIR", "top_scores")
    anchor_reinforcement_module.tpl_detection_fast_annoy_simple_with_logging(os.path.join(DATA_PATH, "2_target/fcg"),
                        os.path.join(DATA_PATH, "3_candidate/fcg"), 
                        os.path.join(DATA_PATH, "5_func_compare_result", tpl_score_dir) + "/", 
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
    # TPL_detection_module3.cal_libae_result(os.path.join(DATA_PATH, save_path+"tpl_fast_result.json"), os.path.join(GT_PATH, "tpl_ground_truth.json"), os.path.join(DATA_PATH, save_path+"TPL_score/"))
    
    
   

if __name__ == "__main__":
    cli()
