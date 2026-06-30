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


def cli():
    print("hello libAE")


    # # 3. function_compare
    print("start anchor detection......")
    anchor_detection_module.func_compare_annoy_fast_multi(os.path.join(DATA_PATH, "4_embedding/target_in9_embedding.json"), 
        os.path.join(DATA_PATH, "4_embedding/candidate_in9_embedding.json"), 
        os.path.join(DATA_PATH, "5_func_compare_result/score"), 
        os.path.join(DATA_PATH, "5_func_compare_result/top_scores"), 
        os.path.join(DATA_PATH, "5_func_compare_result"),
        os.path.join(DATA_PATH, "5_func_compare_result/embedding_annoy"))
    
   

if __name__ == "__main__":
    cli()
