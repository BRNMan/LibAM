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
