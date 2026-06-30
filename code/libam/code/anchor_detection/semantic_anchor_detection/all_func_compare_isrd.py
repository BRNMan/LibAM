import json, os, sys, pickle, tqdm, time
import numpy as np
from multiprocessing import Process
sys.path.append(".")
from settings import DATA_PATH
from annoy import AnnoyIndex


def get_func_embeddings(object_path):
    
    detect_binary_func_vec = {}
    
    detect_vec = json.load(open(object_path, "r"))
    for detect_item in detect_vec:
        detect_name = "|||".join(detect_item.split("|||")[:-1])
        func_name = detect_item.split("|||")[-1]
        if detect_name not in detect_binary_func_vec:
            detect_binary_func_vec[detect_name] = {}
        if func_name not in detect_binary_func_vec[detect_name]:
            detect_binary_func_vec[detect_name][func_name] = np.array(detect_vec[detect_item]).reshape(-1,64)
    
    return detect_binary_func_vec


def select_diverse_top_matches(func_score_dict, top_k=100, per_binary_cap=20):
    """
    Keep nearest matches while limiting how many come from the same candidate binary.
    This prevents one binary/version from monopolizing the top-k list.
    """
    selected = []
    per_bin_count = {}

    for match_key, score in sorted(func_score_dict.items(), key=lambda d: d[1]):
        candidate_binary = match_key.split("----", 1)[0]
        used = per_bin_count.get(candidate_binary, 0)
        if used >= per_binary_cap:
            continue
        selected.append((match_key, score))
        per_bin_count[candidate_binary] = used + 1
        if len(selected) >= top_k:
            break

    return selected


def select_global_top_matches(selected_items, max_total=50000, per_candidate_cap=5000):
    """
    Apply a second-stage global cap after per-function selection.
    This keeps output size bounded even when a target binary has many functions.
    """
    final_items = []
    per_candidate_count = {}

    for match_key, score in sorted(selected_items, key=lambda d: d[1]):
        candidate_binary = match_key.split("||||", 1)[1].split("----", 1)[0]
        used = per_candidate_count.get(candidate_binary, 0)
        if used >= per_candidate_cap:
            continue
        final_items.append((match_key, score))
        per_candidate_count[candidate_binary] = used + 1
        if len(final_items) >= max_total:
            break

    return final_items


def _safe_json_dump(data, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


def save_all_candidate_index(candidate_binary_func_vec, embed_path, black_func_list, f=64, n_trees=100):
    ann_path = os.path.join(embed_path, "all_candidate.ann")
    func_path = os.path.join(embed_path, "all_candidate_func.json")
    bin_path = os.path.join(embed_path, "all_candidate_bin.json")

    # Try using existing cache first.
    if os.path.exists(ann_path) and os.path.exists(func_path) and os.path.exists(bin_path):
        try:
            t = AnnoyIndex(f, 'angular')
            t.load(ann_path)
            json.load(open(func_path, "r"))
            json.load(open(bin_path, "r"))
            return
        except Exception:
            # Corrupted or incompatible cache; rebuild.
            for p in (ann_path, func_path, bin_path):
                if os.path.exists(p):
                    os.remove(p)

    t = AnnoyIndex(f, 'angular')
    all_candidate_id_func_dict = {}
    all_candidate_id_bin_dict = {}
    i = 0
    for candidate_binary in tqdm.tqdm(candidate_binary_func_vec, desc="Building all_candidate Annoy index"):
        candidate_func_vec_dict = candidate_binary_func_vec[candidate_binary]
        for func_name in candidate_func_vec_dict:
            if func_name not in black_func_list:
                all_candidate_id_func_dict[str(i)] = func_name
                all_candidate_id_bin_dict[str(i)] = candidate_binary
                t.add_item(i, candidate_func_vec_dict[func_name].tolist()[0])
                i += 1

    t.build(n_trees)

    tmp_ann_path = ann_path + ".tmp"
    t.save(tmp_ann_path)
    os.replace(tmp_ann_path, ann_path)
    _safe_json_dump(all_candidate_id_func_dict, func_path)
    _safe_json_dump(all_candidate_id_bin_dict, bin_path)



def func_compare_annoy_fast_one(detect_binary_func_vec_list, detect_binary_func_vec, candidate_binary_func_vec, score_opath, score_opath2, time_opath, embed_path):
    black_func_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]
    enable_diag = os.environ.get("LIBAM_COMPARE_DIAG", "1") == "1"
    ann_top_n = max(1, int(os.environ.get("LIBAM_COMPARE_ANN_TOPN", "200")))
    dist_threshold = float(os.environ.get("LIBAM_COMPARE_DIST_THRESHOLD", "1.000"))
    topk_per_func = max(1, int(os.environ.get("LIBAM_COMPARE_TOPK_PER_FUNC", "50")))
    per_bin_cap = max(1, int(os.environ.get("LIBAM_COMPARE_PER_BIN_CAP", "50")))
    max_total = max(1, int(os.environ.get("LIBAM_COMPARE_MAX_TOTAL", "5000")))
    per_cdd_bin_cap = max(1, int(os.environ.get("LIBAM_COMPARE_MAX_PER_CDD_BIN", "5000")))
    for detect_binary in tqdm.tqdm(detect_binary_func_vec_list, desc="Target Binary Progress"):
        if detect_binary in detect_binary_func_vec and not os.path.exists(os.path.join(time_opath, detect_binary+"isrd_triple_loss_time.json")):
            time_dict = {}
            start = time.time()
            score_dict = {}
            deal_score_dict = {}
            raw_score_dict = {}
            detect_func_vec_dict =  detect_binary_func_vec[detect_binary]
    
            t = AnnoyIndex(64, 'angular')
            if os.path.exists(os.path.join(embed_path, "all_candidate_bin.json")):
                t.load(os.path.join(embed_path, "all_candidate.ann"))
                all_candidate_id_func_dict = json.load(open(os.path.join(embed_path, "all_candidate_func.json"), "r"))
                all_candidate_id_bin_dict = json.load(open(os.path.join(embed_path, "all_candidate_bin.json"), "r"))
            else:
                save_all_candidate_index(candidate_binary_func_vec, embed_path, black_func_list, f=64, n_trees=100)
                t.load(os.path.join(embed_path, "all_candidate.ann"))
                all_candidate_id_func_dict = json.load(open(os.path.join(embed_path, "all_candidate_func.json"), "r"))
                all_candidate_id_bin_dict = json.load(open(os.path.join(embed_path, "all_candidate_bin.json"), "r"))
                
            candidate_bin_dict = {}
            for target_funcname in tqdm.tqdm(detect_func_vec_dict, desc=f"\t Detecting candidate anchors in {detect_binary}", position=0, leave=True):
                if target_funcname in black_func_list:
                    continue
                query_result, distance_result = t.get_nns_by_vector(
                    detect_func_vec_dict[target_funcname].tolist()[0],
                    ann_top_n,
                    include_distances=True,
                )
                for i in range(len(query_result)):
                    if distance_result[i] < dist_threshold:#0.7483:
                        if target_funcname not in score_dict:
                            score_dict[target_funcname] = {}
                        score_dict[target_funcname][all_candidate_id_bin_dict[str(query_result[i])]+"----"+all_candidate_id_func_dict[str(query_result[i])]] = distance_result[i]
                        raw_score_dict[detect_binary+"----"+target_funcname+"||||"+all_candidate_id_bin_dict[str(query_result[i])]+"----"+all_candidate_id_func_dict[str(query_result[i])]] = distance_result[i]
                    else:
                        break
                        
            selected_candidate_bins = set()
            pre_global_selected = []
            for detect_func in score_dict:
                object_cdd_func_list = select_diverse_top_matches(
                    score_dict[detect_func],
                    top_k=topk_per_func,
                    per_binary_cap=per_bin_cap,
                )
                for object_cdd_func_item in object_cdd_func_list:
                    match_key = detect_binary+"----"+detect_func+"||||"+object_cdd_func_item[0]
                    pre_global_selected.append((match_key, object_cdd_func_item[1]))
                    selected_candidate_bins.add(object_cdd_func_item[0].split("----", 1)[0])

            for match_key, score in select_global_top_matches(
                pre_global_selected,
                max_total=max_total,
                per_candidate_cap=per_cdd_bin_cap,
            ):
                deal_score_dict[match_key] = score

            if enable_diag:
                raw_candidate_bins = set()
                for func_name in score_dict:
                    for match_key in score_dict[func_name]:
                        raw_candidate_bins.add(match_key.split("----", 1)[0])
                print(
                    "[diag] {}: raw_bins={} selected_bins={} raw_pairs={} pre_global_selected={} selected_pairs={} matched_funcs={} ann_top_n={} dist_th={} topk_per_func={} per_bin_cap={} max_total={} per_cdd_bin_cap={}".format(
                        detect_binary,
                        len(raw_candidate_bins),
                        len(selected_candidate_bins),
                        len(raw_score_dict),
                        len(pre_global_selected),
                        len(deal_score_dict),
                        len(score_dict),
                        ann_top_n,
                        dist_threshold,
                        topk_per_func,
                        per_bin_cap,
                        max_total,
                        per_cdd_bin_cap,
                    )
                )
            end = time.time()
            run_time = end - start
            time_dict[detect_binary] = run_time
            json.dump(raw_score_dict, open(os.path.join(score_opath, detect_binary+"_reuse_func_dict.json"), "w"))
            json.dump(deal_score_dict, open(os.path.join(score_opath2, detect_binary+"_reuse_func_dict.json"), "w"))
            json.dump(time_dict, open(os.path.join(time_opath, detect_binary+"isrd_triple_loss_time.json"), "w"))    


def func_compare_annoy_fast_multi(object_path, candidate_path, score_opath, score_opath2, time_opath, embed_path):
    detect_binary_func_vec = get_func_embeddings(object_path)
    candidate_binary_func_vec = get_func_embeddings(candidate_path)
    
    
    
    detect_binary_func_vec_list = list(detect_binary_func_vec.keys())

    if False == os.path.exists(score_opath):
        os.makedirs(score_opath)
    if False == os.path.exists(score_opath2):
        os.makedirs(score_opath2)
    if False == os.path.exists(time_opath):
        os.makedirs(time_opath)
    if False == os.path.exists(embed_path):
        os.makedirs(embed_path)

    black_func_list = ["_start", "__libc_start_main", "main", "mainSort.isra.1", "mainSort.isra.0", "usage", "mainGtU.part.0", "mainSort", "__libc_csu_init", "frame_dummy", "deregister_tm_clones", "register_tm_clones"]
    save_all_candidate_index(candidate_binary_func_vec, embed_path, black_func_list, f=64, n_trees=100)
    
    p_list = []
    Process_num = max(1, int(os.environ.get("LIBAM_COMPARE_PROCESSES", "1")))

    if Process_num == 1:
        func_compare_annoy_fast_one(
            detect_binary_func_vec_list,
            detect_binary_func_vec,
            candidate_binary_func_vec,
            score_opath,
            score_opath2,
            time_opath,
            embed_path,
        )
        return

    for i in range(Process_num):
        p = Process(target=func_compare_annoy_fast_one, args=(detect_binary_func_vec_list[int((i/Process_num)*len(detect_binary_func_vec_list)):int(((i+1)/Process_num)*len(detect_binary_func_vec_list))], detect_binary_func_vec, candidate_binary_func_vec, score_opath, score_opath2, time_opath, embed_path))
        p_list.append(p)
            #args_list.append([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
            # compare_one_cdd_bin([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
    for p in p_list:
        p.start()
        # time.sleep(15)
    for p in tqdm.tqdm(p_list):
        p.join()
