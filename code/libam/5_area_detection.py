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
    print("start reuse area detection......")
    if "dataset2" in DATA_PATH:
        reuse_area_detection_module.get_area_result_several(os.path.join(DATA_PATH, "7_reuse_detection_result/reuse_detection_area/"), 
            os.path.join(DATA_PATH, "8_reuse_area_result/reuse_detection_area"), 
            os.path.join(GT_PATH, "area_ground_truth.json"),
            os.path.join(DATA_PATH, "2_target/fcg"), 
            os.path.join(DATA_PATH, "3_candidate/fcg") )
    elif "dataset3" in DATA_PATH:
        reuse_area_detection_module.get_area_result_for_each(os.path.join(DATA_PATH, "7_reuse_detection_result/reuse_detection_area/"), 
            os.path.join(DATA_PATH, "8_reuse_area_result/reuse_detection_area_for_each"), 
            os.path.join(GT_PATH, "area_ground_truth.json"),
            os.path.join(DATA_PATH, "2_target/fcg"), 
            os.path.join(DATA_PATH, "3_candidate/fcg") )   

if __name__ == "__main__":
    cli()
