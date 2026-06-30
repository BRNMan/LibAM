import gc
import json
import multiprocessing
import os
import pickle
import sys
import time
from multiprocessing import Process

from tqdm import tqdm

from app.taint import tpl_detection_fast_core_annoy
from app.utils import get_cdd_func_dict

sys.setrecursionlimit(200000)

def tpl_detection_fast_one_annoy_simple_with_logging(
    func_path_list,
    tar_fcg_path,
    cdd_fcg_path,
    func_path,
    feature_save_path,
    time_path,
    com_funcs_path,
    sim_funcs_path,
    gnn,
    fcgs_num,
    obj_com_funcs_dict,
    cdd_com_funcs_dict,
    tar_afcg_dict,
    tar_subgraph_path,
    cdd_afcg_dict,
    cdd_subgraph_dict,
    area_save_path,
    cdd_func_embeddings_path,
    process_id,
    memory_log_path,
):
    import psutil
    disable_gnn = os.environ.get("LIBAM_TPL_DISABLE_GNN", "0") == "1"

    process = psutil.Process(os.getpid())
    memory_log = []

    print(f"Worker {process_id} starting", flush=True)
    sys.stdout.flush()

    for object_item in func_path_list:
        object_name = object_item.split("_reuse_func_dict")[0]
        reuse_result = {object_name: []}

        try:
            with open(os.path.join(tar_fcg_path, f"{object_name}_fcg.pkl"), "rb") as f:
                object_fcg = pickle.load(f)
        except FileNotFoundError:
            continue

        try:
            object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        except (FileNotFoundError, json.JSONDecodeError):
            del object_fcg
            continue

        if disable_gnn:
            tar_subgraph = {}
        else:
            try:
                with open(os.path.join(tar_subgraph_path, f"{object_name}_subgraph.json"), "r") as f:
                    tar_subgraph = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                del object_fcg
                del object_cdd_func_dict
                continue

        cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
        obj_com_funcs = obj_com_funcs_dict.get(object_name, [])

        for candidate_name in cdd_project_dict:
            if candidate_name not in cdd_com_funcs_dict:
                continue
            if object_name not in tar_afcg_dict or (not disable_gnn and tar_subgraph is None):
                continue
            if candidate_name not in cdd_afcg_dict:
                continue
            if not disable_gnn and candidate_name not in cdd_subgraph_dict:
                continue

            mem_before_cdd_load = process.memory_info().rss / 1024 / 1024

            cdd_com_funcs = cdd_com_funcs_dict.get(candidate_name, [])
            matched_func_list = cdd_project_dict[candidate_name]

            json.dump(
                matched_func_list,
                open(os.path.join(sim_funcs_path, f"{object_name}___{candidate_name}.json"), "w"),
            )

            try:
                with open(os.path.join(cdd_fcg_path, f"{candidate_name}_fcg.pkl"), "rb") as f:
                    candidate_fcg = pickle.load(f)
            except FileNotFoundError:
                continue

            mem_after_cdd_load = process.memory_info().rss / 1024 / 1024
            mem_before_detection = process.memory_info().rss / 1024 / 1024

            reuse_flag, reuse_dict = tpl_detection_fast_core_annoy(
                object_name,
                candidate_name,
                object_fcg,
                matched_func_list,
                candidate_fcg,
                obj_com_funcs,
                cdd_com_funcs,
                gnn,
                fcgs_num,
                tar_afcg_dict[object_name],
                cdd_afcg_dict[candidate_name],
                tar_subgraph,
                cdd_subgraph_dict[candidate_name],
            )

            mem_after_detection = process.memory_info().rss / 1024 / 1024

            if reuse_flag:
                reuse_result[object_name].append(candidate_name)
                with open(os.path.join(area_save_path, f"{object_name}___{candidate_name}_feature_result.json"), "w") as ff:
                    json.dump(reuse_dict, ff)

            mem_after_save = process.memory_info().rss / 1024 / 1024

            memory_log.append(
                {
                    "process_id": process_id,
                    "object_name": object_name,
                    "candidate_name": candidate_name,
                    "delta_cdd_load": mem_after_cdd_load - mem_before_cdd_load,
                    "delta_detection": mem_after_detection - mem_before_detection,
                    "delta_save": mem_after_save - mem_after_detection,
                    "peak_mem": mem_after_detection,
                }
            )

            del candidate_fcg
            del reuse_dict
            gc.collect()

        result_path = os.path.join(feature_save_path, f"{object_name}_reuse_result.json")
        if os.path.exists(result_path):
            try:
                with open(result_path, "r") as f:
                    existing = json.load(f)
                for k, v in existing.items():
                    if k not in reuse_result:
                        reuse_result[k] = v
                    else:
                        reuse_result[k] = list(set(reuse_result[k]) | set(v))
            except (json.JSONDecodeError, KeyError):
                pass

        json.dump(reuse_result, open(result_path, "w"))

        del object_fcg
        del tar_subgraph
        del object_cdd_func_dict
        del cdd_project_dict
        gc.collect()

    print(f"Worker {process_id} writing memory log to {memory_log_path}_{process_id}.json", flush=True)
    sys.stdout.flush()
    with open(f"{memory_log_path}_{process_id}.json", "w") as f:
        json.dump(memory_log, f, indent=2)

    print(f"Worker {process_id} done", flush=True)
    sys.stdout.flush()


def tpl_detection_fast_annoy_simple_with_logging(
    tar_fcg_path,
    cdd_fcg_path,
    func_path,
    feature_save_path,
    area_save_path,
    time_path,
    com_funcs_path,
    sim_funcs_path,
    obj_func_embeddings_path,
    cdd_func_embeddings_path,
    gnn_model_path,
    tar_afcg_path,
    cdd_afcg_path,
    tar_subgraph_path,
    cdd_subgraph_path,
):
    import psutil
    disable_gnn = os.environ.get("LIBAM_TPL_DISABLE_GNN", "1") == "1"

    main_process = psutil.Process(os.getpid())
    object_item_list = os.listdir(func_path)

    tar_afcg_dict = {}
    for tar_afcg_item in os.listdir(tar_afcg_path):
        tar_bin_name = tar_afcg_item.split("_afcg.json")[0]
        tar_afcg_dict[tar_bin_name] = json.load(open(os.path.join(tar_afcg_path, tar_afcg_item), "r"))

    if disable_gnn:
        cdd_subgraph_files = sorted(
            [item.replace("_afcg.json", "_subgraph.json") for item in os.listdir(cdd_afcg_path) if item.endswith("_afcg.json")]
        )
    else:
        cdd_subgraph_files = sorted(os.listdir(cdd_subgraph_path))

    for path in [feature_save_path, area_save_path, sim_funcs_path, time_path]:
        os.makedirs(path, exist_ok=True)

    fcgs_num = {}
    for fcg_p in os.listdir(tar_fcg_path):
        try:
            with open(os.path.join(tar_fcg_path, fcg_p), "rb") as f:
                fcg = pickle.load(f)
            fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
            del fcg
        except (FileNotFoundError, pickle.PickleError):
            continue

    for fcg_p in os.listdir(cdd_fcg_path):
        try:
            with open(os.path.join(cdd_fcg_path, fcg_p), "rb") as f:
                fcg = pickle.load(f)
            fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
            del fcg
        except (FileNotFoundError, pickle.PickleError):
            continue

    gc.collect()

    obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
    cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()

    obj_com_funcs_dict = {}
    for obj_item in obj_com_funcs_file:
        if obj_item.split("|||")[0] not in obj_com_funcs_dict:
            obj_com_funcs_dict[obj_item.split("|||")[0]] = []
        if obj_item.split("|||")[1] not in obj_com_funcs_dict[obj_item.split("|||")[0]]:
            obj_com_funcs_dict[obj_item.split("|||")[0]].append(obj_item.split("|||")[1])

    cdd_com_funcs_dict = {}
    for cdd_item in cdd_com_funcs_file:
        if cdd_item.split("|||")[0] not in cdd_com_funcs_dict:
            cdd_com_funcs_dict[cdd_item.split("|||")[0]] = []
        if cdd_item.split("|||")[1] not in cdd_com_funcs_dict[cdd_item.split("|||")[0]]:
            cdd_com_funcs_dict[cdd_item.split("|||")[0]].append(cdd_item.split("|||")[1])

    gnn = False
    candidates_per_phase = 3
    num_phases = (len(cdd_subgraph_files) + candidates_per_phase - 1) // candidates_per_phase

    memory_log_base = os.path.join(time_path, "worker_memory")
    phase_memory_log = []

    for phase in range(num_phases):
        phase_start = phase * candidates_per_phase
        phase_end = min((phase + 1) * candidates_per_phase, len(cdd_subgraph_files))
        phase_subgraph_files = cdd_subgraph_files[phase_start:phase_end]

        mem_before_load = main_process.memory_info().rss / 1024 / 1024

        cdd_afcg_dict = {}
        for cdd_subgraph_item in phase_subgraph_files:
            cdd_bin_name = cdd_subgraph_item.split("_subgraph.json")[0]
            afcg_file = f"{cdd_bin_name}_afcg.json"
            try:
                cdd_afcg_dict[cdd_bin_name] = json.load(open(os.path.join(cdd_afcg_path, afcg_file), "r"))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        cdd_subgraph_dict = {}
        if not disable_gnn:
            for cdd_subgraph_item in phase_subgraph_files:
                cdd_bin_name = cdd_subgraph_item.split("_subgraph.json")[0]
                try:
                    cdd_subgraph_dict[cdd_bin_name] = json.load(open(os.path.join(cdd_subgraph_path, cdd_subgraph_item), "r"))
                except (FileNotFoundError, json.JSONDecodeError):
                    continue

        mem_after_load = main_process.memory_info().rss / 1024 / 1024
        phase_memory_log.append(
            {
                "phase": phase + 1,
                "num_candidates": len(phase_subgraph_files),
                "mem_before_mb": mem_before_load,
                "mem_after_mb": mem_after_load,
                "delta_mb": mem_after_load - mem_before_load,
            }
        )

        p_list = []
        process_num = 2
        ctx = multiprocessing.get_context("fork")

        for i in range(process_num):
            start_idx = int((i / process_num) * len(object_item_list))
            end_idx = int(((i + 1) / process_num) * len(object_item_list))
            func_path_slice = object_item_list[start_idx:end_idx]
            if len(func_path_slice) == 0:
                continue

            p = ctx.Process(
                target=tpl_detection_fast_one_annoy_simple_with_logging,
                args=(
                    func_path_slice,
                    tar_fcg_path,
                    cdd_fcg_path,
                    func_path,
                    feature_save_path,
                    time_path,
                    com_funcs_path,
                    sim_funcs_path,
                    gnn,
                    fcgs_num,
                    obj_com_funcs_dict,
                    cdd_com_funcs_dict,
                    tar_afcg_dict,
                    tar_subgraph_path,
                    cdd_afcg_dict,
                    cdd_subgraph_dict,
                    area_save_path,
                    cdd_func_embeddings_path,
                    i,
                    memory_log_base,
                ),
            )
            p_list.append(p)
            p.start()

        for p in p_list:
            p.join()
        for p in p_list:
            p.close()

        del cdd_afcg_dict
        del cdd_subgraph_dict
        gc.collect()

    with open(os.path.join(time_path, "phase_memory_profile.json"), "w") as f:
        json.dump(phase_memory_log, f, indent=2)