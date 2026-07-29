import random
import os

import torch
import torch.nn.functional as F
import tqdm


def _extract_sub_addr(func_name):
    if not isinstance(func_name, str):
        return None
    name = func_name.split("|||", 1)[-1]
    if not name.startswith("sub_"):
        return None
    try:
        return int(name[4:], 16)
    except ValueError:
        return None


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
        obj_func = matched_func[0]
        cdd_func = matched_func[1]
        anchor_dist = matched_func[2] if len(matched_func) > 2 else None

        if "|||" in obj_func:
            obj_func = obj_func.split("|||")[-1]
        if "|||" in cdd_func:
            cdd_func = cdd_func.split("|||")[-1]

        if obj_func in obj_node_list and cdd_func in cdd_node_list:
            in_graph_node.append([obj_func, cdd_func, anchor_dist])

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
    return sorted(object_cdd_func_dict.items(), key=lambda d: d[1], reverse=False)


def get_cdd_func_dict(object_cdd_func_dict):
    cdd_project_dict = {}
    object_cdd_func_dict = filter_200_lib(object_cdd_func_dict)
    object_cdd_func_list = filter_500_anchor(object_cdd_func_dict)
    for matched_item in object_cdd_func_list:
        cdd_item = matched_item[0].split("||||")[1].split("----")[0]
        obj_func_item = matched_item[0].split("||||")[0].split("----")[1]
        cdd_func_item = matched_item[0].split("||||")[1].split("----")[1]
        try:
            anchor_dist = float(matched_item[1])
        except (TypeError, ValueError):
            anchor_dist = None
        if cdd_item not in cdd_project_dict:
            cdd_project_dict[cdd_item] = []
        cdd_project_dict[cdd_item].append(["".join(obj_func_item), "".join(cdd_func_item), anchor_dist])
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
    disable_gnn = os.environ.get("LIBAM_TPL_DISABLE_GNN", "0") == "1"
    collect_pair_stats = os.environ.get("LIBAM_TPL_COLLECT_PAIR_STATS", "0") == "1"
    pair_stats = []
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
    debug_filter_raw = os.environ.get("LIBAM_TPL_DEBUG_FUNC", "rpl_mbrtowc")
    debug_filters = [item.strip() for item in debug_filter_raw.split(",") if item.strip()]
    debug_all_low_align = os.environ.get("LIBAM_TPL_DEBUG_ALL_LOW_ALIGN", "0") == "1"
    try:
        debug_sample_limit = max(1, int(os.environ.get("LIBAM_TPL_DEBUG_SAMPLE_LIMIT", "8")))
    except ValueError:
        debug_sample_limit = 8

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

    # Early exit threshold for no-match binaries.
    threshold_pairs = max(1, int(stats["pairs_total"] * 0.20))

    for pair_idx, func_pair in enumerate(pair_iter, start=1):
        processed_pairs = pair_idx - 1
        if processed_pairs >= threshold_pairs and stats["accepted"] == 0:
            if enable_progress:
                tqdm.tqdm.write(
                    f"[tpl-diag] Stopping early: searched {processed_pairs}/{stats['pairs_total']} pairs (20% threshold), found 0 matches"
                )
            break

        pair_record = {
            "object_name": object_name,
            "candidate_name": candidate_name,
            "pair_index": pair_idx,
            "obj_func": func_pair[0],
            "cdd_func": func_pair[1],
            "obj_addr_int": _extract_sub_addr(func_pair[0]),
            "cdd_addr_int": _extract_sub_addr(func_pair[1]),
            "obj_addr_hex": None,
            "cdd_addr_hex": None,
            "anchor_distance": None,
            "gnn_score": None,
            "align_rate": None,
            "alignment_num": None,
            "obj_n_num": None,
            "cdd_n_num": None,
            "obj_unique_funcs": None,
            "cdd_unique_funcs": None,
            "final_score": None,
        }
        if pair_record["obj_addr_int"] is not None:
            pair_record["obj_addr_hex"] = hex(pair_record["obj_addr_int"])
        if pair_record["cdd_addr_int"] is not None:
            pair_record["cdd_addr_hex"] = hex(pair_record["cdd_addr_int"])
        if len(func_pair) > 2:
            try:
                pair_record["anchor_distance"] = float(func_pair[2])
            except (TypeError, ValueError):
                pair_record["anchor_distance"] = None

        def _store_pair(accepted, reason):
            if not collect_pair_stats:
                return
            rec = dict(pair_record)
            rec["accepted"] = accepted
            rec["reject_reason"] = reason
            pair_stats.append(rec)

        def _fmt_func_overlap_items(items):
            if not items:
                return "[]"
            out_items = []
            for src_func, overlaps in items[:debug_sample_limit]:
                overlap_preview = overlaps[:debug_sample_limit]
                out_items.append(f"{src_func}->{overlap_preview}")
            return "[{}]".format(", ".join(out_items)
            )

        def _print_alignment_debug(reason, gnn_score_value, align_rate_value, chosen_side, obj_stats, cdd_stats):
            should_print = is_debug or (debug_all_low_align and reason == "low_align_rate")
            if not should_print:
                return
            obj_com, obj_sim, obj_hits, obj_miss = obj_stats
            cdd_com, cdd_sim, cdd_hits, cdd_miss = cdd_stats
            combined = gnn_score_value * (0.3 * align_rate_value + 0.7)
            print(
                "[DEBUG] {} vs {}: ALIGN_STATS reason={} chosen_side={} gnn={:.4f} align_rate={:.4f} combined={:.4f}".format(
                    func_pair[0],
                    func_pair[1],
                    reason,
                    chosen_side,
                    gnn_score_value,
                    align_rate_value,
                    combined,
                )
            )
            print(
                "[DEBUG]   obj: common={} sim={} ratio={:.4f} hit_examples={} miss_examples={}".format(
                    obj_com,
                    obj_sim,
                    (obj_sim / obj_com) if obj_com else 0.0,
                    _fmt_func_overlap_items(obj_hits),
                    obj_miss[:debug_sample_limit],
                )
            )
            print(
                "[DEBUG]   cdd: common={} sim={} ratio={:.4f} hit_examples={} miss_examples={}".format(
                    cdd_com,
                    cdd_sim,
                    (cdd_sim / cdd_com) if cdd_com else 0.0,
                    _fmt_func_overlap_items(cdd_hits),
                    cdd_miss[:debug_sample_limit],
                )
            )

        if enable_progress and pair_idx % 50 == 0:
            pair_iter.set_postfix(
                accepted=stats["accepted"],
                low_gnn=stats["skip_low_gnn"],
                low_align=stats["skip_low_align_rate"],
                short_align=stats["skip_short_alignment"],
            )

        obj_afcg = get_afcg_one_annoy(func_pair[0], obj_sim_funcs, tar_afcg_dict)
        cdd_afcg = get_afcg_one_annoy(func_pair[1], cdd_sim_funcs, cdd_afcg_dict)
        
        is_debug = any(token in func_pair[0] or token in func_pair[1] for token in debug_filters)
        
        if len(obj_afcg) == 0 or len(cdd_afcg) == 0:
            if is_debug:
                print(f"[DEBUG] {func_pair[0]} vs {func_pair[1]}: SKIPPED - empty_afcg (obj_afcg={len(obj_afcg)}, cdd_afcg={len(cdd_afcg)})")
            stats["skip_empty_afcg"] += 1
            _store_pair(False, "skip_empty_afcg")
            continue
        if func_pair[1] in black_list:
            if is_debug:
                print(f"[DEBUG] {func_pair[0]} vs {func_pair[1]}: SKIPPED - blacklist")
            stats["skip_blacklist"] += 1
            _store_pair(False, "skip_blacklist")
            continue

        if disable_gnn:
            obj_fcg = build_local_fcg_from_afcg(func_pair[0], tar_afcg_dict)
            cdd_fcg = build_local_fcg_from_afcg(func_pair[1], cdd_afcg_dict)
            gnn_score = 1.0
        else:
            if func_pair[0] not in tar_subgraph or func_pair[1] not in cdd_subgraph_dict:
                if is_debug:
                    print(f"[DEBUG] {func_pair[0]} vs {func_pair[1]}: SKIPPED - missing_subgraph (in_tar={func_pair[0] in tar_subgraph}, in_cdd={func_pair[1] in cdd_subgraph_dict})")
                stats["skip_missing_subgraph"] += 1
                _store_pair(False, "skip_missing_subgraph")
                continue
            obj_fcg = tar_subgraph[func_pair[0]]
            cdd_fcg = cdd_subgraph_dict[func_pair[1]]

            obj_embedding = torch.tensor(obj_fcg["embedding"])
            cdd_embedding = torch.tensor(cdd_fcg["embedding"])
            gnn_score = F.cosine_similarity(obj_embedding, cdd_embedding, eps=1e-10, dim=1)
            gnn_score = (1 + gnn_score.cpu().detach().numpy()[0]) / 2.0

        pair_record["gnn_score"] = float(gnn_score)
        pair_record["obj_n_num"] = int(obj_fcg.get("n_num", 0))
        pair_record["cdd_n_num"] = int(cdd_fcg.get("n_num", 0))

        # Lower the GNN threshold for pairs with a very close anchor embedding match.
        anchor_dist = func_pair[2] if len(func_pair) > 2 and func_pair[2] is not None else 1.0
        gnn_threshold = 0.8 - max(0.0, (0.5 - anchor_dist)) * 1.0

        if gnn_score < gnn_threshold:
            if is_debug:
                print(f"[DEBUG] {func_pair[0]} vs {func_pair[1]}: SKIPPED - low_gnn (score={gnn_score:.4f} < {gnn_threshold:.4f})")
            stats["skip_low_gnn"] += 1
            _store_pair(False, "skip_low_gnn")
            continue

        obj_feature_set = set(obj_fcg["feature"])
        cdd_feature_set = set(cdd_fcg["feature"])
        obj_num = len(obj_feature_set)
        cdd_num = len(cdd_feature_set)
        pair_record["obj_unique_funcs"] = int(obj_num)
        pair_record["cdd_unique_funcs"] = int(cdd_num)

        obj_com_num = obj_sim_num = 0
        obj_hit_examples = []
        obj_miss_examples = []
        for obj_func in obj_feature_set:
            if obj_func in obj_com_funcs:
                obj_com_num += 1
                related_cdd = set(obj_sim_funcs_dict.get(obj_func, []))
                overlap = sorted(list(related_cdd.intersection(cdd_feature_set)))
                if overlap:
                    obj_sim_num += 1
                    if is_debug or debug_all_low_align:
                        obj_hit_examples.append((obj_func, overlap))
                elif is_debug or debug_all_low_align:
                    obj_miss_examples.append(obj_func)

        cdd_com_num = cdd_sim_num = 0
        cdd_hit_examples = []
        cdd_miss_examples = []
        for cdd_func in cdd_feature_set:
            if cdd_func in cdd_com_funcs:
                cdd_com_num += 1
                related_obj = set(cdd_sim_funcs_dict.get(cdd_func, []))
                overlap = sorted(list(related_obj.intersection(obj_feature_set)))
                if overlap:
                    cdd_sim_num += 1
                    if is_debug or debug_all_low_align:
                        cdd_hit_examples.append((cdd_func, overlap))
                elif is_debug or debug_all_low_align:
                    cdd_miss_examples.append(cdd_func)

        if obj_com_num == 0 or cdd_com_num == 0:
            if is_debug:
                print(f"[DEBUG] {func_pair[0]} vs {func_pair[1]}: SKIPPED - no_child_funcs (obj_com={obj_com_num}, cdd_com={cdd_com_num})")
            stats["skip_no_child_funcs_found"] += 1
            _store_pair(False, "skip_no_child_funcs_found")
            continue
        chosen_side = "obj"
        if obj_com_num <= cdd_com_num:
            align_rate = obj_sim_num / obj_com_num
        else:
            chosen_side = "cdd"
            align_rate = cdd_sim_num / cdd_com_num

        pair_record["align_rate"] = float(align_rate)

        align_rate_score = 0.3 * align_rate + 0.7
        if gnn_score * align_rate_score < 0.8:
            _print_alignment_debug(
                "low_align_rate",
                gnn_score,
                align_rate,
                chosen_side,
                (obj_com_num, obj_sim_num, obj_hit_examples, obj_miss_examples),
                (cdd_com_num, cdd_sim_num, cdd_hit_examples, cdd_miss_examples),
            )
            if is_debug:
                print(f"[DEBUG] {func_pair[0]} vs {func_pair[1]}: SKIPPED - low_align_rate (gnn={gnn_score:.4f}, align_rate={align_rate:.4f}, score={gnn_score*align_rate_score:.4f} < 0.8)")
            stats["skip_low_align_rate"] += 1
            _store_pair(False, "skip_low_align_rate")
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
            if is_debug:
                print(f"[DEBUG] {func_pair[0]} vs {func_pair[1]}: SKIPPED - short_alignment (len={len(lenth_max)} < 2)")
            stats["skip_short_alignment"] += 1
            _store_pair(False, "skip_short_alignment")
            continue

        alignment_temp = len(lenth_max)
        if (abs(obj_num - cdd_num) - min(obj_num, cdd_num) > 2 * min(obj_num, cdd_num) and max(obj_num, cdd_num) > 100) or (abs(obj_num - cdd_num) > 200):
            alignment_temp = 0

        pair_record["alignment_num"] = int(alignment_temp)

        if not ((obj_fcg["n_num"] >= 3 and cdd_fcg["n_num"] >= 3 and alignment_temp >= alignment_tred) or (obj_num <= 10 and cdd_num <= 10 and alignment_temp >= 3)):
            if is_debug:
                print(f"[DEBUG] {func_pair[0]} vs {func_pair[1]}: SKIPPED - scale_guard (obj_n={obj_fcg['n_num']}, cdd_n={cdd_fcg['n_num']}, align_temp={alignment_temp})")
            stats["skip_scale_guard"] += 1
            _store_pair(False, "skip_scale_guard")
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
            if is_debug:
                print(f"[DEBUG] {func_pair[0]} vs {func_pair[1]}: SKIPPED - final_guard (align_num_score={node_alignment_num_score}, scale_pair={node_fcg_scale_pair})")
            stats["skip_final_guard"] += 1
            _store_pair(False, "skip_final_guard")
            continue

        final_score = RARM_score(
            node_alignment_num_score,
            node_gnn_score,
            node_fcg_scale_score,
            node_fcg_scale_diff_score,
            align_rate,
        )
        node_pair_feature[node_pair_str]["final_score"] = final_score
        pair_record["final_score"] = float(final_score)

        if (final_score >= 0.8 and node_alignment_num_score >= alignment_tred) or (final_score >= 0.95 and node_alignment_num_score >= 2):
            if candidate_name not in target_reuse_area_dict:
                target_reuse_area_dict[candidate_name] = {}
            if node_pair_str not in target_reuse_area_dict[candidate_name]:
                target_reuse_area_dict[candidate_name][node_pair_str] = []
            target_reuse_area_dict[candidate_name][node_pair_str].append(node_pair_feature[node_pair_str])
            reuse_flag = True
            stats["accepted"] += 1
            _store_pair(True, None)
        else:
            _store_pair(False, "skip_accept_threshold")
        
        if stats["accepted"] >= 5:
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
        return reuse_flag, target_reuse_area_dict, pair_stats
    return reuse_flag, {}, pair_stats
