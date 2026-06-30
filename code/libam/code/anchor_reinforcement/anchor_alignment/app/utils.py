import random
import os

import torch
import torch.nn.functional as F
import tqdm


def build_local_fcg_from_afcg(func_name, afcg_dict):
    # Fallback local structure for no-GNN mode: root function + its AFCG children.
    children = []
    if func_name in afcg_dict:
        children = [c for c in afcg_dict[func_name] if c != func_name]
    feature = [func_name]
    for child in children:
        if child not in feature:
            feature.append(child)
    return {"feature": feature, "n_num": len(feature), "embedding": None}


def get_afcg_one_annoy(func_pair, sim_funcs, all_afcg):
    afcg = []
    if func_pair in all_afcg:
        afcg_pre = all_afcg[func_pair]
        for child_node in afcg_pre:
            # Strictly check if the anchor has children that are also anchors
            # This is a prerequisite for the anchor alignment algorithm. 
            if child_node in sim_funcs and child_node != func_pair and child_node not in afcg:
                afcg.append(child_node)
    return afcg


def judge_in_graph(object_graph, candidate_graph, matched_func_list):
    in_graph_node = []
    obj_node_list = list(object_graph.nodes())
    cdd_node_list = list(candidate_graph.nodes())

    for matched_func in matched_func_list:
        if "|||" in matched_func[0]:
            matched_func[0] = matched_func[0].split("|||")[-1]
            matched_func[1] = matched_func[1].split("|||")[-1]
        if matched_func[0] in obj_node_list and matched_func[1] in cdd_node_list:
            in_graph_node.append(matched_func)

    return in_graph_node


def filter_200_lib(object_cdd_func_dict):
    filtered_lib_dict = {}
    filer_lib_dict = {}
    for matched_item in object_cdd_func_dict:
        lib_name = matched_item.split("||||")[1].split("----")[0]
        if lib_name not in filer_lib_dict:
            filer_lib_dict[lib_name] = 0
        filer_lib_dict[lib_name] += 1

    filer_lib_dict_sorted = list(filer_lib_dict.keys())
    filer_lib_dict_sorted.sort(key=filer_lib_dict.__getitem__, reverse=True)

    for matched_item in object_cdd_func_dict:
        lib_name = matched_item.split("||||")[1].split("----")[0]
        if lib_name in filer_lib_dict_sorted[:200]:
            filtered_lib_dict[matched_item] = object_cdd_func_dict[matched_item]

    return filtered_lib_dict


def filter_500_anchor(object_cdd_func_dict):
    return sorted(object_cdd_func_dict.items(), key=lambda d: d[1], reverse=True)


def get_cdd_func_dict(object_cdd_func_dict):
    cdd_project_dict = {}
    object_cdd_func_dict = filter_200_lib(object_cdd_func_dict)
    object_cdd_func_list = filter_500_anchor(object_cdd_func_dict)
    for matched_item in object_cdd_func_list:
        cdd_item = matched_item[0].split("||||")[1].split("----")[0]
        obj_func_item = matched_item[0].split("||||")[0].split("----")[1]
        cdd_func_item = matched_item[0].split("||||")[1].split("----")[1]
        if cdd_item not in cdd_project_dict:
            cdd_project_dict[cdd_item] = []
        cdd_project_dict[cdd_item].append(["".join(obj_func_item), "".join(cdd_func_item)])
    return cdd_project_dict


def Alignment_v2(
    obj_func,
    cdd_func,
    obj_afcg,
    cdd_afcg,
    obj_sim_funcs_dict,
    length,
    obj_sim_funcs,
    object_graph,
    cdd_sim_funcs,
    candidate_graph,
    matched_func_ingraph_list,
    tar_afcg_dict,
    cdd_afcg_dict,
):
    N = 0
    if len(length) >= 3:
        return N, length

    a_tar_child = random.sample(obj_afcg, 1)[0]

    a_tpl_child = False
    obj_related_funcs_item = obj_sim_funcs_dict[a_tar_child]
    a_tpl_child_item_list = []
    for a_tpl_child_item in obj_related_funcs_item:
        if a_tpl_child_item in cdd_afcg:
            a_tpl_child_item_list.append(a_tpl_child_item)
    if a_tpl_child_item_list:
        a_tpl_child = random.sample(a_tpl_child_item_list, 1)[0]

    if a_tpl_child:
        N += 1
        length.append([a_tar_child, a_tpl_child])
        obj_afcg_child = get_afcg_one_annoy(a_tar_child, obj_sim_funcs, tar_afcg_dict)
        obj_related_funcs_new = obj_sim_funcs_dict[a_tar_child]
        cdd_afcg_child = get_afcg_one_annoy(a_tpl_child, cdd_sim_funcs, cdd_afcg_dict)
        if len(obj_afcg_child) > 0 and len(cdd_afcg_child) > 0 and len(obj_related_funcs_new) > 0:
            l, length = Alignment_v2(
                a_tar_child,
                a_tpl_child,
                obj_afcg_child,
                cdd_afcg_child,
                obj_sim_funcs_dict,
                length,
                obj_sim_funcs,
                object_graph,
                cdd_sim_funcs,
                candidate_graph,
                matched_func_ingraph_list,
                tar_afcg_dict,
                cdd_afcg_dict,
            )
            N += l
        return N, length

    return N, length


def RARM_score(alignment_num_score, node_gnn_score, node_fcg_scale_score, node_fcg_scale_diff_score, align_rate):
    align_rate_score = 0.3 * align_rate + 0.7
    return node_gnn_score * align_rate_score


def tpl_detection_fast_utils_annoy_v2(
    object_name,
    candidate_name,
    matched_func_ingraph_list,
    object_graph,
    candidate_graph,
    obj_com_funcs,
    cdd_com_funcs,
    gnn,
    fcgs_num,
    tar_afcg_dict,
    cdd_afcg_dict,
    tar_subgraph,
    cdd_subgraph_dict,
):
    reuse_flag = False
    disable_gnn = os.environ.get("LIBAM_TPL_DISABLE_GNN", "1") == "1"
    black_list = [
        "_start",
        "__libc_start_main",
        "main",
        "mainSort.isra.1",
        "mainSort.isra.0",
        "usage",
        "mainGtU.part.0",
        "mainSort",
        "__libc_csu_init",
        "frame_dummy",
        "deregister_tm_clones",
        "register_tm_clones",
    ]

    alignment_tred = 3
    enable_progress = os.environ.get("LIBAM_TPL_PROGRESS", "1") == "1"

    obj_sim_funcs = []
    obj_sim_funcs_dict = {}
    cdd_sim_funcs = []
    cdd_sim_funcs_dict = {}
    for func_pair in matched_func_ingraph_list:
        if func_pair[0] not in obj_sim_funcs:
            obj_sim_funcs.append(func_pair[0])
            obj_sim_funcs_dict[func_pair[0]] = []
        if func_pair[1] not in obj_sim_funcs_dict[func_pair[0]]:
            obj_sim_funcs_dict[func_pair[0]].append(func_pair[1])
        if func_pair[1] not in cdd_sim_funcs:
            cdd_sim_funcs.append(func_pair[1])
            cdd_sim_funcs_dict[func_pair[1]] = []
        if func_pair[0] not in cdd_sim_funcs_dict[func_pair[1]]:
            cdd_sim_funcs_dict[func_pair[1]].append(func_pair[0])

    target_reuse_area_dict = {}
    stats = {
        "pairs_total": len(matched_func_ingraph_list),
        "skip_empty_afcg": 0,
        "skip_blacklist": 0,
        "skip_missing_subgraph": 0,
        "skip_low_gnn": 0,
        "skip_low_align_rate": 0,
        "skip_short_alignment": 0,
        "skip_scale_guard": 0,
        "skip_final_guard": 0,
        "skip_no_child_funcs_found": 0,
        "accepted": 0,
    }

    pair_iter = tqdm.tqdm(
        matched_func_ingraph_list,
        desc=f"TPL {object_name}->{candidate_name}",
        leave=False,
        disable=not enable_progress,
    )

    for pair_idx, func_pair in enumerate(pair_iter, start=1):
        if enable_progress and pair_idx % 50 == 0:
            pair_iter.set_postfix(
                accepted=stats["accepted"],
                low_gnn=stats["skip_low_gnn"],
                low_align=stats["skip_low_align_rate"],
                short_align=stats["skip_short_alignment"],
            )

        obj_afcg = get_afcg_one_annoy(func_pair[0], obj_sim_funcs, tar_afcg_dict)
        cdd_afcg = get_afcg_one_annoy(func_pair[1], cdd_sim_funcs, cdd_afcg_dict)
        if len(obj_afcg) == 0 or len(cdd_afcg) == 0:
            stats["skip_empty_afcg"] += 1
            continue
        if func_pair[1] in black_list:
            stats["skip_blacklist"] += 1
            continue

        if disable_gnn:
            obj_fcg = build_local_fcg_from_afcg(func_pair[0], tar_afcg_dict)
            cdd_fcg = build_local_fcg_from_afcg(func_pair[1], cdd_afcg_dict)
            gnn_score = 1.0
        else:
            if func_pair[0] not in tar_subgraph or func_pair[1] not in cdd_subgraph_dict:
                stats["skip_missing_subgraph"] += 1
                continue
            obj_fcg = tar_subgraph[func_pair[0]]
            cdd_fcg = cdd_subgraph_dict[func_pair[1]]

            obj_embedding = torch.tensor(obj_fcg["embedding"])
            cdd_embedding = torch.tensor(cdd_fcg["embedding"])
            gnn_score = F.cosine_similarity(obj_embedding, cdd_embedding, eps=1e-10, dim=1)
            gnn_score = (1 + gnn_score.cpu().detach().numpy()[0]) / 2.0

        if gnn_score < 0.8:
            stats["skip_low_gnn"] += 1
            continue

        obj_num = len(set(obj_fcg["feature"]))
        cdd_num = len(set(cdd_fcg["feature"]))

        obj_com_num = obj_sim_num = 0
        for obj_func in set(obj_fcg["feature"]):
            if obj_func in obj_com_funcs:
                obj_com_num += 1
                if obj_func in obj_sim_funcs_dict and list(set(obj_sim_funcs_dict[obj_func]).intersection(set(cdd_fcg["feature"]))) != []:
                    obj_sim_num += 1

        cdd_com_num = cdd_sim_num = 0
        for cdd_func in set(cdd_fcg["feature"]):
            if cdd_func in cdd_com_funcs:
                cdd_com_num += 1
                if cdd_func in cdd_sim_funcs_dict and list(set(cdd_sim_funcs_dict[cdd_func]).intersection(set(obj_fcg["feature"]))) != []:
                    cdd_sim_num += 1

        if obj_com_num == 0 or cdd_com_num == 0:
            stats["skip_no_child_funcs_found"] += 1
            continue
        if obj_com_num <= cdd_com_num:
            align_rate = obj_sim_num / obj_com_num
        else:
            align_rate = cdd_sim_num / cdd_com_num

        align_rate_score = 0.3 * align_rate + 0.7
        if gnn_score * align_rate_score < 0.8:
            stats["skip_low_align_rate"] += 1
            continue

        l_max = 0
        lenth_max = [func_pair]
        n = 0
        while True:
            length = [func_pair]
            l, length = Alignment_v2(
                func_pair[0],
                func_pair[1],
                obj_afcg,
                cdd_afcg,
                obj_sim_funcs_dict,
                length,
                obj_sim_funcs,
                object_graph,
                cdd_sim_funcs,
                candidate_graph,
                matched_func_ingraph_list,
                tar_afcg_dict,
                cdd_afcg_dict,
            )
            if l > l_max:
                l_max = l
                lenth_max = length
                n = 0
            else:
                n += 1
            if n >= 100 or len(lenth_max) >= alignment_tred:
                break

        if len(lenth_max) < 2:
            stats["skip_short_alignment"] += 1
            continue

        alignment_temp = len(lenth_max)
        if (abs(obj_num - cdd_num) - min(obj_num, cdd_num) > 2 * min(obj_num, cdd_num) and max(obj_num, cdd_num) > 100) or (abs(obj_num - cdd_num) > 200):
            alignment_temp = 0

        if not ((obj_fcg["n_num"] >= 3 and cdd_fcg["n_num"] >= 3 and alignment_temp >= alignment_tred) or (obj_num <= 10 and cdd_num <= 10 and alignment_temp >= 2)):
            stats["skip_scale_guard"] += 1
            continue

        node_pair = func_pair
        node_pair_feature = {
            str(node_pair): {
                "obj_fcg": obj_fcg,
                "cdd_fcg": cdd_fcg,
                "alignment_rate": align_rate,
                "fcg_scale": (obj_num, cdd_num),
                "gnn_score": str(gnn_score),
                "obj_full_fcg_num": str(fcgs_num[object_name]),
                "alignment_num": alignment_temp,
            }
        }

        node_pair_str = str(node_pair)
        node_alignment_num_score = node_pair_feature[node_pair_str]["alignment_num"]
        node_fcg_scale_pair = node_pair_feature[node_pair_str]["fcg_scale"]
        node_gnn_score = float(node_pair_feature[node_pair_str]["gnn_score"])
        align_rate = float(node_pair_feature[node_pair_str]["alignment_rate"])
        node_fcg_scale_score = (node_fcg_scale_pair[0] + node_fcg_scale_pair[1]) / 2
        node_fcg_scale_diff_score = 0.3 * min(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) / max(node_fcg_scale_pair[0], node_fcg_scale_pair[1]) + 0.7

        if node_alignment_num_score <= 0 or node_fcg_scale_pair[0] < 2 or node_fcg_scale_pair[1] < 2:
            stats["skip_final_guard"] += 1
            continue

        final_score = RARM_score(
            node_alignment_num_score,
            node_gnn_score,
            node_fcg_scale_score,
            node_fcg_scale_diff_score,
            align_rate,
        )
        node_pair_feature[node_pair_str]["final_score"] = final_score

        if (final_score >= 0.8 and node_alignment_num_score >= alignment_tred) or (final_score >= 0.95 and node_alignment_num_score >= 2):
            if candidate_name not in target_reuse_area_dict:
                target_reuse_area_dict[candidate_name] = {}
            if node_pair_str not in target_reuse_area_dict[candidate_name]:
                target_reuse_area_dict[candidate_name][node_pair_str] = []
            target_reuse_area_dict[candidate_name][node_pair_str].append(node_pair_feature[node_pair_str])
            reuse_flag = True
            stats["accepted"] += 1
            break

    if enable_progress:
        tqdm.tqdm.write(
            "[tpl-diag] {} -> {} | pairs={} accepted={} empty_afcg={} blacklist={} missing_subgraph={} low_gnn={} low_align={} short_align={} scale_guard={} final_guard={}, no_child_funcs_found={}, disable_gnn={}".format(
                object_name,
                candidate_name,
                stats["pairs_total"],
                stats["accepted"],
                stats["skip_empty_afcg"],
                stats["skip_blacklist"],
                stats["skip_missing_subgraph"],
                stats["skip_low_gnn"],
                stats["skip_low_align_rate"],
                stats["skip_short_alignment"],
                stats["skip_scale_guard"],
                stats["skip_final_guard"],
                stats["skip_no_child_funcs_found"],
                int(disable_gnn),
            )
        )

    if reuse_flag:
        return reuse_flag, target_reuse_area_dict
    return reuse_flag, {}
