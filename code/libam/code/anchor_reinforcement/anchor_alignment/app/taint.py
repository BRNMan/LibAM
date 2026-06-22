from app.utils import judge_in_graph, tpl_detection_fast_utils_annoy_v2


def tpl_detection_fast_core_annoy(
    object_name,
    candidate_name,
    object_graph,
    matched_func_list,
    candidate_graph,
    obj_com_funcs,
    cdd_com_funcs,
    cdd_func_embeddings,
    gnn,
    fcgs_num,
    tar_afcg_dict,
    cdd_afcg_dict,
    tar_subgraph_dict,
    cdd_subgraph_dict,
):
    matched_func_ingraph_list = judge_in_graph(object_graph, candidate_graph, matched_func_list)
    return tpl_detection_fast_utils_annoy_v2(
        object_name,
        candidate_name,
        matched_func_ingraph_list,
        object_graph,
        candidate_graph,
        obj_com_funcs,
        cdd_com_funcs,
        cdd_func_embeddings,
        gnn,
        fcgs_num,
        tar_afcg_dict,
        cdd_afcg_dict,
        tar_subgraph_dict,
        cdd_subgraph_dict,
    )
