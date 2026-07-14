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


import binary_preprocess as binary_preprocess_module


def cli():
    print("hello libAE")
    
    # 1. get feature and fcg
    print("start bianry preprocess......")
    binary_preprocess_module.getAllFiles(DATA_PATH+"2_target/timecost", DATA_PATH+"1_binary/target", DATA_PATH+"2_target/", mode="1")
    binary_preprocess_module.getAllFiles(DATA_PATH+"3_candidate/timecost", DATA_PATH + "1_binary/candidate", DATA_PATH + "3_candidate/", mode="1")
   

if __name__ == "__main__":
    cli()
