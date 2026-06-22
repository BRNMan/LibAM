
# from sympy import fft
from app.explore import *
from app.taint import *
import sys, time
from tqdm import tqdm
sys.setrecursionlimit(200000)
from multiprocessing import Process
sys.path.append(".")
from settings import DATA_PATH

def get_reuse_area(fcg_path, func_path, feature_save_path, time_path):
    cal_time = {}
    for object_item in tqdm(os.listdir(func_path)):
        object_name = object_item.split("_reuse_func_dict.json")[0]
        if "bzip2" in object_name:
            object_fcg = get_fcg(fcg_path, object_name)
            object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
            cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
            start = time.time()
            print("********* compare "+object_name+"**********")
            for candidate_name in tqdm(cdd_project_dict):
                if "precomp" in candidate_name:
                    # if object_name=="bzip2_x86_O3" and candidate_name == "turbobench_datasets_isrd_to_gemini":
                    if object_item.split("_")[0] == candidate_name.split("_")[0]:
                        continue
                    matched_func_list = cdd_project_dict[candidate_name]
                    candidate_fcg = get_fcg(fcg_path, candidate_name)
                    # print("********* compare "+object_name+" and "+candidate_name+"**********")
                    
                    node_pair_feature, reuse_flag, max_alignment_num = reuse_area_exploration(object_fcg, matched_func_list, candidate_fcg)
                    
                    # if reuse_flag:
                    with open(os.path.join(feature_save_path, object_name+"----"+candidate_name+"_feature_result_"+str(max_alignment_num)+".json"), "w") as ff:
                        json.dump(node_pair_feature, ff)
            end = time.time()
            
            cal_time[object_item] = end - start
            with open(time_path, "w") as ff:
                json.dump(cal_time, ff)  





def get_reuse_area_one(func_path_list, func_path, fcg_path, feature_save_path, time_path):
    
    for object_item in tqdm(func_path_list):
        
        cal_time = {}
        # error_list = ["brotli"]
        object_name = object_item.split("_reuse_func_dict")[0]
        # if "lzbench" in object_name:
        if "brotli" in object_name:
            break
                # if "bzip2" in object_name:
                # object_fcg, find_fcg = get_libdb_fcg(fcg_path, object_name)
        object_fcg = get_isrd_fcg(fcg_path, object_name)
        # if find_fcg == False :
        #     with open("obj_error.txt","w") as eef:
        #         eef.write(object_name+"\n")
        #         continue
        object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
        start = time.time()
        # print("********* compare "+object_name+"**********")
        for candidate_name in tqdm(cdd_project_dict):
            # if "write_read_test" in candidate_name:
                    # if "precomp" in candidate_name:
                    # if object_name=="bzip2_x86_O3" and candidate_name == "turbobench_datasets_isrd_to_gemini":
            if object_item.split("_")[0] == candidate_name.split("_")[0]:
                continue
            matched_func_list = cdd_project_dict[candidate_name]
            # candidate_fcg,find_fcg = get_libdb_fcg(fcg_path, candidate_name)
            candidate_fcg = get_isrd_fcg(fcg_path, candidate_name)
            # if find_fcg == False :
            #     with open("cdd_error.txt","w") as eef:
            #         eef.write(candidate_name+"\n")
            #         continue
            # print("********* compare "+object_name+" and "+candidate_name+"**********")
            
            node_pair_feature, reuse_flag, max_alignment_num = reuse_area_exploration(object_fcg, matched_func_list, candidate_fcg)
            
            # if reuse_flag:
            with open(os.path.join(feature_save_path, object_name+"----"+candidate_name+"_feature_result_"+str(max_alignment_num)+".json"), "w") as ff:
                json.dump(node_pair_feature, ff)
        end = time.time()
        
        cal_time[object_item] = end - start
        with open(time_path, "w") as ff:
            json.dump(cal_time, ff)  





def get_reuse_area_one_v2_0(func_path_list, tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path):
    
    for object_item in tqdm(func_path_list):
        if not os.path.exists(os.path.join(time_path, object_item+"_timecost.json")):
            cal_time = {}
            # error_list = ["brotli"]
            object_name = object_item.split("_reuse_func_dict")[0]
            # if "lzbench" in object_name:
            # if "brotli" in object_name or "xz" in object_name:
            #     if "brotli|||x64|||O2" in object_name:
            #         print("warning")
                #     break
                        # if "bzip2" in object_name:
                        # object_fcg, find_fcg = get_libdb_fcg(fcg_path, object_name)
            object_fcg = get_isrd_fcg_new(tar_fcg_path, object_name)
            # if find_fcg == False :
            #     with open("obj_error.txt","w") as eef:
            #         eef.write(object_name+"\n")
            #         continue
            object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
            cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
            start = time.time()
            # print("********* compare "+object_name+"**********")
            for candidate_name in tqdm(cdd_project_dict):
                # if "write_read_test" in candidate_name:
                        # if "precomp" in candidate_name:
                        # if object_name=="bzip2_x86_O3" and candidate_name == "turbobench_datasets_isrd_to_gemini":
                if object_item.split("_")[0] == candidate_name.split("_")[0]:
                    continue
                matched_func_list = cdd_project_dict[candidate_name]
                # candidate_fcg,find_fcg = get_libdb_fcg(fcg_path, candidate_name)
                candidate_fcg = get_isrd_fcg_new(cdd_fcg_path, candidate_name)
                # if find_fcg == False :
                #     with open("cdd_error.txt","w") as eef:
                #         eef.write(candidate_name+"\n")
                #         continue
                # print("********* compare "+object_name+" and "+candidate_name+"**********")
                
                node_pair_feature, reuse_flag, max_alignment_num = anchor_alignment_area(object_fcg, matched_func_list, candidate_fcg)
                
                # if reuse_flag:
                with open(os.path.join(feature_save_path, object_name+"----"+candidate_name+"_feature_result_"+str(max_alignment_num)+".json"), "w") as ff:
                    json.dump(node_pair_feature, ff)
            end = time.time()
            
            cal_time[object_item] = end - start
            with open(os.path.join(time_path, object_item+"_timecost.json"), "w") as ff:
                json.dump(cal_time, ff)  



def reuse_area_detection_one(func_path_list, tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs_dict, cdd_com_funcs_dict, tpl_result, area_save_path):
    for object_item in tqdm(func_path_list):
        
        
        
        cal_time = {}
        reuse_result = {}

        object_name = object_item.split("_reuse_func_dict")[0]
        
        
        if object_name not in tpl_result or tpl_result[object_name] == []:
            continue
        
        # if "turbobench" in object_name:
        object_fcg = get_isrd_fcg_new(tar_fcg_path, object_name)

        object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
        obj_com_funcs = obj_com_funcs_dict[object_name]
        start = time.time()
        # print("********* compare "+object_name+"**********")
        for candidate_name in tqdm(cdd_project_dict):
            if candidate_name not in tpl_result[object_name]:
                continue
            # if "bzip2" in object_name:
            # if candidate_name == "liblzg" and object_name == "lzbench":
            
            # obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
            # for obj_item in obj_com_funcs_file:
            #     if obj_item.split("|||")[0] == object_name:
            #         obj_com_funcs.append(obj_item.split("|||")[1])
            cdd_com_funcs = cdd_com_funcs_dict[candidate_name]
            # cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()
            # for cdd_item in cdd_com_funcs_file:
            #     if cdd_item.split("|||")[0] == candidate_name:
            #         cdd_com_funcs.append(cdd_item.split("|||")[1])
            #     print("warning")
            if object_item.split("___")[0] == candidate_name.split("___")[0]:
                continue
            matched_func_list = cdd_project_dict[candidate_name]
            
            # json.dump(matched_func_list, open(os.path.join(sim_funcs_path, object_name+"___"+candidate_name+".json"), "w"))
            
            candidate_fcg = get_isrd_fcg_new(cdd_fcg_path, candidate_name)

        
            reuse_flag, reuse_dict = reuse_area_detection_core(object_name, candidate_name, object_fcg, matched_func_list, candidate_fcg, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num)

            if reuse_flag:
                print("find reuse------"+object_name+"-----"+candidate_name)
                if object_name not in reuse_result:
                    reuse_result[object_name] = []
                reuse_result[object_name].append(candidate_name)
                with open(os.path.join(area_save_path, object_name+"----"+candidate_name+"_feature_result.json"), "w") as ff:
                    json.dump(reuse_dict, ff)

        end = time.time()
        cal_time[object_item] = end - start
        json.dump(reuse_result, open(os.path.join(feature_save_path, object_name+"_reuse_result.json"), "w"))
        with open(os.path.join(time_path, object_item+"_tpl_fast_time.json"), "w") as ff:
            json.dump(cal_time, ff)  
            

def reuse_area_detection_one_annoy(func_path_list, tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs_dict, cdd_com_funcs_dict, area_save_path, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, tar_fcg_dict, cdd_fcg_dict, tpl_result):
    for object_item in tqdm(func_path_list):
        cal_time = {}
        reuse_result = {}

        object_name = object_item.split("_reuse_func_dict")[0]
        
        if object_name not in tpl_result or tpl_result[object_name] == []:
            continue
        
        
        
        # if "turbobench" in object_name:
        object_fcg = tar_fcg_dict[object_name]#get_isrd_fcg_new(tar_fcg_path, object_name)

        object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
        obj_com_funcs = obj_com_funcs_dict[object_name]
        
        
        # get_obj_afcg()
        # get_obj_subgraph()
        # get_cdd_afcg()
        # get_cdd_subgraph()
        
        area_result_dict = dict()
        start = time.time()
        # print("********* compare "+object_name+"**********")
        for candidate_name in tqdm(cdd_project_dict):
            # if "bzip2" in object_name:
            # if candidate_name == "liblzg" and object_name == "lzbench":
            if candidate_name not in tpl_result[object_name]:
                continue
            # obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
            # for obj_item in obj_com_funcs_file:
            #     if obj_item.split("|||")[0] == object_name:
            #         obj_com_funcs.append(obj_item.split("|||")[1])
            cdd_com_funcs = cdd_com_funcs_dict[candidate_name]
            # cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()
            # for cdd_item in cdd_com_funcs_file:
            #     if cdd_item.split("|||")[0] == candidate_name:
            #         cdd_com_funcs.append(cdd_item.split("|||")[1])
            #     print("warning")
            # if object_item.split("___")[0] == candidate_name.split("___")[0]:
            #     continue
            matched_func_list = cdd_project_dict[candidate_name]
            
            # json.dump(matched_func_list, open(os.path.join(sim_funcs_path, object_name+"___"+candidate_name+".json"), "w"))
            
            candidate_fcg = cdd_fcg_dict[candidate_name]#get_isrd_fcg_new(cdd_fcg_path, candidate_name)

        
            reuse_flag, reuse_dict = reuse_area_detection_core_annoy(object_name, candidate_name, object_fcg, matched_func_list, candidate_fcg, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num, tar_afcg_dict[object_name], cdd_afcg_dict[candidate_name], tar_subgraph_dict[object_name], cdd_subgraph_dict[candidate_name])

            if reuse_flag:
                print("find reuse------"+object_name+"-----"+candidate_name)
                if object_name not in reuse_result:
                    reuse_result[object_name] = []
                reuse_result[object_name].append(candidate_name)
                if len(area_result_dict) == 0:
                    area_result_dict = reuse_dict
                else:
                    area_result_dict = dict(area_result_dict, **reuse_dict)
                # with open(os.path.join(area_save_path, object_name+"----"+candidate_name+"_feature_result.json"), "w") as ff:
                #     json.dump(area_result_dict, ff)

        end = time.time()
        json.dump(reuse_result, open(os.path.join(feature_save_path, object_name+"_reuse_result.json"), "w"))
        with open(os.path.join(area_save_path, object_name+"_reuse_area.json"), "w") as ff:
            json.dump(area_result_dict, ff)
        cal_time[object_item] = end - start
        with open(os.path.join(time_path, object_item+"_tpl_fast_time.json"), "w") as ff:
            json.dump(cal_time, ff)  
            
def tpl_detection_fast_one_annoy_without_align(func_path_list, tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs_dict, cdd_com_funcs_dict, area_save_path, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, tar_fcg_dict, cdd_fcg_dict):
    for object_item in tqdm(func_path_list):
        
        cal_time = {}
        reuse_result = {}

        object_name = object_item.split("_reuse_func_dict")[0]
        # if "ASUS__AC68U_30043762048__smbd"  not in object_name and "ASUS__AC68U_30043762048__minidlna" not in object_name:
        # if "bzip2" in object_name:
        object_fcg = tar_fcg_dict[object_name]#get_isrd_fcg_new(tar_fcg_path, object_name)

        object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
        obj_com_funcs = obj_com_funcs_dict[object_name]
        
        
        # get_obj_afcg()
        # get_obj_subgraph()
        # get_cdd_afcg()
        # get_cdd_subgraph()
        
        
        start = time.time()
        # print("********* compare "+object_name+"**********")
        for candidate_name in tqdm(cdd_project_dict, desc = object_name+" ..."):
            # if "bzip2" in object_name:
            # if candidate_name == "libblosc":
            
            # obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
            # for obj_item in obj_com_funcs_file:
            #     if obj_item.split("|||")[0] == object_name:
            #         obj_com_funcs.append(obj_item.split("|||")[1])
            cdd_com_funcs = cdd_com_funcs_dict[candidate_name]
            # cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()
            # for cdd_item in cdd_com_funcs_file:
            #     if cdd_item.split("|||")[0] == candidate_name:
            #         cdd_com_funcs.append(cdd_item.split("|||")[1])
            #     print("warning")
            # if object_item.split("___")[0] == candidate_name.split("___")[0]:
            #     continue
            matched_func_list = cdd_project_dict[candidate_name]
            
            json.dump(matched_func_list, open(os.path.join(sim_funcs_path, object_name+"___"+candidate_name+".json"), "w"))
            
            

            if object_name in tar_afcg_dict and object_name in tar_subgraph_dict and candidate_name in cdd_afcg_dict and candidate_name in cdd_subgraph_dict:
                candidate_fcg = cdd_fcg_dict[candidate_name]#get_isrd_fcg_new(cdd_fcg_path, candidate_name)
                reuse_flag, reuse_dict = tpl_detection_fast_core_annoy_without_align(object_name, candidate_name, object_fcg, matched_func_list, candidate_fcg, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num, tar_afcg_dict[object_name], cdd_afcg_dict[candidate_name], tar_subgraph_dict[object_name], cdd_subgraph_dict[candidate_name])

                if reuse_flag:
                    print("find reuse------"+object_name+"-----"+candidate_name)
                    if object_name not in reuse_result:
                        reuse_result[object_name] = []
                    reuse_result[object_name].append(candidate_name)
                    with open(os.path.join(area_save_path, object_name+"----"+candidate_name+"_feature_result.json"), "w") as ff:
                        json.dump(reuse_dict, ff)

        end = time.time()
        json.dump(reuse_result, open(os.path.join(feature_save_path, object_name+"_reuse_result.json"), "w"))
        cal_time[object_item] = end - start
        with open(os.path.join(time_path, object_item+"_tpl_fast_time.json"), "w") as ff:
            json.dump(cal_time, ff)  
            

            
def tpl_detection_fast_one_annoy_without_gnn(func_path_list, tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs_dict, cdd_com_funcs_dict, area_save_path, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, tar_fcg_dict, cdd_fcg_dict):
    for object_item in tqdm(func_path_list):
        
        cal_time = {}
        reuse_result = {}

        object_name = object_item.split("_reuse_func_dict")[0]
        # if "ASUS__AC68U_30043762048__smbd"  not in object_name and "ASUS__AC68U_30043762048__minidlna" not in object_name:
        # if "bzip2" in object_name:
        object_fcg = tar_fcg_dict[object_name]#get_isrd_fcg_new(tar_fcg_path, object_name)

        object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
        obj_com_funcs = obj_com_funcs_dict[object_name]
        
        
        # get_obj_afcg()
        # get_obj_subgraph()
        # get_cdd_afcg()
        # get_cdd_subgraph()
        
        
        start = time.time()
        # print("********* compare "+object_name+"**********")
        for candidate_name in tqdm(cdd_project_dict, desc = object_name+" ..."):
            # if "bzip2" in object_name:
            # if candidate_name == "libblosc":
            
            # obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
            # for obj_item in obj_com_funcs_file:
            #     if obj_item.split("|||")[0] == object_name:
            #         obj_com_funcs.append(obj_item.split("|||")[1])
            cdd_com_funcs = cdd_com_funcs_dict[candidate_name]
            # cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()
            # for cdd_item in cdd_com_funcs_file:
            #     if cdd_item.split("|||")[0] == candidate_name:
            #         cdd_com_funcs.append(cdd_item.split("|||")[1])
            #     print("warning")
            # if object_item.split("___")[0] == candidate_name.split("___")[0]:
            #     continue
            matched_func_list = cdd_project_dict[candidate_name]
            
            json.dump(matched_func_list, open(os.path.join(sim_funcs_path, object_name+"___"+candidate_name+".json"), "w"))
            
            

            if object_name in tar_afcg_dict and object_name in tar_subgraph_dict and candidate_name in cdd_afcg_dict and candidate_name in cdd_subgraph_dict:
                candidate_fcg = cdd_fcg_dict[candidate_name]#get_isrd_fcg_new(cdd_fcg_path, candidate_name)
                reuse_flag, reuse_dict = tpl_detection_fast_core_annoy_without_gnn(object_name, candidate_name, object_fcg, matched_func_list, candidate_fcg, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num, tar_afcg_dict[object_name], cdd_afcg_dict[candidate_name], tar_subgraph_dict[object_name], cdd_subgraph_dict[candidate_name])

                if reuse_flag:
                    print("find reuse------"+object_name+"-----"+candidate_name)
                    if object_name not in reuse_result:
                        reuse_result[object_name] = []
                    reuse_result[object_name].append(candidate_name)
                    with open(os.path.join(area_save_path, object_name+"----"+candidate_name+"_feature_result.json"), "w") as ff:
                        json.dump(reuse_dict, ff)

        end = time.time()
        json.dump(reuse_result, open(os.path.join(feature_save_path, object_name+"_reuse_result.json"), "w"))
        cal_time[object_item] = end - start
        with open(os.path.join(time_path, object_item+"_tpl_fast_time.json"), "w") as ff:
            json.dump(cal_time, ff)  
            

def tpl_detection_fast_one_annoy_1_5(func_path_list, tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs_dict, cdd_com_funcs_dict, area_save_path, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, tar_fcg_dict, cdd_fcg_dict, alignment_tred):
    for object_item in tqdm(func_path_list):
        
        cal_time = {}
        reuse_result = {}

        object_name = object_item.split("_reuse_func_dict")[0]
        # if "ASUS__AC68U_30043762048__smbd"  not in object_name and "ASUS__AC68U_30043762048__minidlna" not in object_name:
        # if "turbobench" in object_name:
        object_fcg = tar_fcg_dict[object_name]#get_isrd_fcg_new(tar_fcg_path, object_name)

        object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
        obj_com_funcs = obj_com_funcs_dict[object_name]
        
        
        # get_obj_afcg()
        # get_obj_subgraph()
        # get_cdd_afcg()
        # get_cdd_subgraph()
        
        
        start = time.time()
        # print("********* compare "+object_name+"**********")
        for candidate_name in tqdm(cdd_project_dict, desc = object_name+" ..."):
            # if "bzip2" in object_name:
            # if candidate_name == "quicklz":
            
            # obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
            # for obj_item in obj_com_funcs_file:
            #     if obj_item.split("|||")[0] == object_name:
            #         obj_com_funcs.append(obj_item.split("|||")[1])
            cdd_com_funcs = cdd_com_funcs_dict[candidate_name]
            # cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()
            # for cdd_item in cdd_com_funcs_file:
            #     if cdd_item.split("|||")[0] == candidate_name:
            #         cdd_com_funcs.append(cdd_item.split("|||")[1])
            #     print("warning")
            # if object_item.split("___")[0] == candidate_name.split("___")[0]:
            #     continue
            matched_func_list = cdd_project_dict[candidate_name]
            
            json.dump(matched_func_list, open(os.path.join(sim_funcs_path, object_name+"___"+candidate_name+".json"), "w"))
            
            

            if object_name in tar_afcg_dict and object_name in tar_subgraph_dict and candidate_name in cdd_afcg_dict and candidate_name in cdd_subgraph_dict:
                candidate_fcg = cdd_fcg_dict[candidate_name]#get_isrd_fcg_new(cdd_fcg_path, candidate_name)
                reuse_flag, reuse_dict = tpl_detection_fast_core_annoy_1_5(object_name, candidate_name, object_fcg, matched_func_list, candidate_fcg, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num, tar_afcg_dict[object_name], cdd_afcg_dict[candidate_name], tar_subgraph_dict[object_name], cdd_subgraph_dict[candidate_name], alignment_tred)

                if reuse_flag:
                    print("find reuse------"+object_name+"-----"+candidate_name)
                    if object_name not in reuse_result:
                        reuse_result[object_name] = []
                    reuse_result[object_name].append(candidate_name)
                    with open(os.path.join(area_save_path, object_name+"----"+candidate_name+"_feature_result.json"), "w") as ff:
                        json.dump(reuse_dict, ff)

        end = time.time()
        json.dump(reuse_result, open(os.path.join(feature_save_path, object_name+"_reuse_result.json"), "w"))
        cal_time[object_item] = end - start
        with open(os.path.join(time_path, object_item+"_tpl_fast_time.json"), "w") as ff:
            json.dump(cal_time, ff)  


def tpl_detection_fast_one_annoy(func_path_list, tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs_dict, cdd_com_funcs_dict, area_save_path, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, tar_fcg_dict, cdd_fcg_dict):
    for object_item in tqdm(func_path_list):
        
        cal_time = {}
        reuse_result = {}

        object_name = object_item.split("_reuse_func_dict")[0]
        # if "ASUS__AC68U_30043762048__smbd"  not in object_name and "ASUS__AC68U_30043762048__minidlna" not in object_name:
        # if "bzip2_arm_O3" in object_name:
        object_fcg = tar_fcg_dict[object_name]#get_isrd_fcg_new(tar_fcg_path, object_name)

        object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
        obj_com_funcs = obj_com_funcs_dict[object_name]
        
        
        # get_obj_afcg()
        # get_obj_subgraph()
        # get_cdd_afcg()
        # get_cdd_subgraph()
        
        
        start = time.time()
        # print("********* compare "+object_name+"**********")
        for candidate_name in tqdm(cdd_project_dict, desc = object_name+" ..."):
            # if "bzip2" in object_name:
            # if candidate_name == "minizip":
            
            # obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
            # for obj_item in obj_com_funcs_file:
            #     if obj_item.split("|||")[0] == object_name:
            #         obj_com_funcs.append(obj_item.split("|||")[1])
            cdd_com_funcs = cdd_com_funcs_dict[candidate_name]
            # cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()
            # for cdd_item in cdd_com_funcs_file:
            #     if cdd_item.split("|||")[0] == candidate_name:
            #         cdd_com_funcs.append(cdd_item.split("|||")[1])
            #     print("warning")
            # if object_item.split("___")[0] == candidate_name.split("___")[0]:
            #     continue
            matched_func_list = cdd_project_dict[candidate_name]
            
            json.dump(matched_func_list, open(os.path.join(sim_funcs_path, object_name+"___"+candidate_name+".json"), "w"))
            
            

            if object_name in tar_afcg_dict and object_name in tar_subgraph_dict and candidate_name in cdd_afcg_dict and candidate_name in cdd_subgraph_dict:
                candidate_fcg = cdd_fcg_dict[candidate_name]#get_isrd_fcg_new(cdd_fcg_path, candidate_name)
                reuse_flag, reuse_dict = tpl_detection_fast_core_annoy(object_name, candidate_name, object_fcg, matched_func_list, candidate_fcg, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num, tar_afcg_dict[object_name], cdd_afcg_dict[candidate_name], tar_subgraph_dict[object_name], cdd_subgraph_dict[candidate_name])

                if reuse_flag:
                    print("find reuse------"+object_name+"-----"+candidate_name)
                    if object_name not in reuse_result:
                        reuse_result[object_name] = []
                    reuse_result[object_name].append(candidate_name)
                    with open(os.path.join(area_save_path, object_name+"----"+candidate_name+"_feature_result.json"), "w") as ff:
                        json.dump(reuse_dict, ff)

        end = time.time()
        json.dump(reuse_result, open(os.path.join(feature_save_path, object_name+"_reuse_result.json"), "w"))
        cal_time[object_item] = end - start
        with open(os.path.join(time_path, object_item+"_tpl_fast_time.json"), "w") as ff:
            json.dump(cal_time, ff)  
            



def tpl_detection_fast_one(func_path_list, tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs_dict, cdd_com_funcs_dict, area_save_path):
    for object_item in tqdm(func_path_list):
        
        cal_time = {}
        reuse_result = {}

        object_name = object_item.split("_reuse_func_dict")[0]
        # if "turbobench" in object_name:
        object_fcg = get_isrd_fcg_new(tar_fcg_path, object_name)

        object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
        obj_com_funcs = obj_com_funcs_dict[object_name]
        start = time.time()
        # print("********* compare "+object_name+"**********")
        for candidate_name in tqdm(cdd_project_dict):
            # if "bzip2" in object_name:
            # if candidate_name == "liblzg" and object_name == "lzbench":
            
            # obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
            # for obj_item in obj_com_funcs_file:
            #     if obj_item.split("|||")[0] == object_name:
            #         obj_com_funcs.append(obj_item.split("|||")[1])
            cdd_com_funcs = cdd_com_funcs_dict[candidate_name]
            # cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()
            # for cdd_item in cdd_com_funcs_file:
            #     if cdd_item.split("|||")[0] == candidate_name:
            #         cdd_com_funcs.append(cdd_item.split("|||")[1])
            #     print("warning")
            # if object_item.split("___")[0] == candidate_name.split("___")[0]:
            #     continue
            matched_func_list = cdd_project_dict[candidate_name]
            
            json.dump(matched_func_list, open(os.path.join(sim_funcs_path, object_name+"___"+candidate_name+".json"), "w"))
            
            candidate_fcg = get_isrd_fcg_new(cdd_fcg_path, candidate_name)

        
            reuse_flag, reuse_dict = tpl_detection_fast_core(object_name, candidate_name, object_fcg, matched_func_list, candidate_fcg, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, gnn, fcgs_num)

            if reuse_flag:
                print("find reuse------"+object_name+"-----"+candidate_name)
                if object_name not in reuse_result:
                    reuse_result[object_name] = []
                reuse_result[object_name].append(candidate_name)
                with open(os.path.join(area_save_path, object_name+"----"+candidate_name+"_feature_result.json"), "w") as ff:
                    json.dump(reuse_dict, ff)

        end = time.time()
        json.dump(reuse_result, open(os.path.join(feature_save_path, object_name+"_reuse_result.json"), "w"))
        cal_time[object_item] = end - start
        with open(os.path.join(time_path, object_item+"_tpl_fast_time.json"), "w") as ff:
            json.dump(cal_time, ff)  
            


def anchor_alignment_one_ransac(func_path_list, tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path):
    for object_item in tqdm(func_path_list):
        
        cal_time = {}

        object_name = object_item.split("_reuse_func_dict")[0]
        # if "turbobench" in object_name:
        object_fcg = get_isrd_fcg_new(tar_fcg_path, object_name)

        object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
        start = time.time()
        # print("********* compare "+object_name+"**********")
        for candidate_name in tqdm(cdd_project_dict):
            # if "bzip2" in object_name:
            # if candidate_name == "liblzg" and object_name == "lzbench":
            obj_com_funcs = []
            obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
            for obj_item in obj_com_funcs_file:
                if obj_item.split("|||")[0] == object_name:
                    obj_com_funcs.append(obj_item.split("|||")[1])
            cdd_com_funcs = []
            cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()
            for cdd_item in cdd_com_funcs_file:
                if cdd_item.split("|||")[0] == candidate_name:
                    cdd_com_funcs.append(cdd_item.split("|||")[1])
            #     print("warning")
            if object_item.split("_")[0] == candidate_name.split("_")[0]:
                continue
            matched_func_list = cdd_project_dict[candidate_name]
            
            json.dump(matched_func_list, open(os.path.join(sim_funcs_path, object_name+"___"+candidate_name+".json"), "w"))
            
            candidate_fcg = get_isrd_fcg_new(cdd_fcg_path, candidate_name)

        
            node_pair_feature, reuse_flag, max_alignment_num, obj_sim_funcs, cdd_sim_funcs = anchor_alignment_ransac(object_fcg, matched_func_list, candidate_fcg, obj_com_funcs, cdd_com_funcs)

            if reuse_flag:
                print("find reuse------"+object_name+"-----"+candidate_name)
                with open(os.path.join(feature_save_path, object_name+"----"+candidate_name+"_feature_result_"+str(max_alignment_num)+".json"), "w") as ff:
                    json.dump(node_pair_feature, ff)

        end = time.time()
        
        cal_time[object_item] = end - start
        with open(os.path.join(time_path, object_name+"_time.json"), "w") as ff:
            json.dump(cal_time, ff)  
            
            
            

# anchor alignment v1.0(ISSTA)
def get_reuse_area_multi(tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path):
    cal_time = {}
    object_item_list = os.listdir(func_path)
    
    if os.path.exists(feature_save_path) == False:
        os.makedirs(feature_save_path)
    if os.path.exists(time_path) == False:
        os.makedirs(time_path)
    
    p_list = []
    Process_num = 50
    for i in range(Process_num):
        p = Process(target=get_reuse_area_one_v2_0, args=(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path))
        p_list.append(p)
            #args_list.append([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
            # compare_one_cdd_bin([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()


def reuse_area_deal(area_result_path, save_path):
    # area_result_func_dict = {}
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    area_result_dict = {}
    for area_result_path_item in tqdm(os.listdir(area_result_path)):
        area_result = json.load(open(os.path.join(area_result_path, area_result_path_item), "r"))
        tar_bin = area_result_path_item.split("----")[0]
        if tar_bin not in area_result_dict:
            area_result_dict[tar_bin] = area_result
        else:
            area_result_dict[tar_bin] = dict(area_result_dict[tar_bin], **area_result)
    for tar_bin in tqdm(area_result_dict):
        json.dump(area_result_dict[tar_bin], open(os.path.join(save_path, tar_bin+"_reuse_area.json"), "w"))


def reuse_area_detection(tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, area_save_path, time_path, com_funcs_path, sim_funcs_path, obj_func_embeddings_path, cdd_func_embeddings_path, gnn_model_path, tpl_result_path):
    cal_time = {}
    # object_item_list_temp = []
    object_item_list = os.listdir(func_path)
    
    tpl_result = json.load(open(tpl_result_path, "r"))
    
    if len(tpl_result) == 0:
        return False    
    # for object_item_list_item in object_item_list:
    #     if "bzip2" in object_item_list_item:
    #         object_item_list_temp.append(object_item_list_item)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gnn = torch.load(gnn_model_path, weights_only=False)
    fcgs_num = {}
    torch.multiprocessing.set_start_method('spawn', force=True)
    for fcg_p in os.listdir(tar_fcg_path):
        with open(os.path.join(tar_fcg_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
    for fcg_p in os.listdir(cdd_fcg_path):
        with open(os.path.join(cdd_fcg_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
        
    if os.path.exists(feature_save_path) == False:
        os.makedirs(feature_save_path)
    if os.path.exists(area_save_path) == False:
        os.makedirs(area_save_path)
    if os.path.exists(sim_funcs_path) == False:
        os.makedirs(sim_funcs_path)
    if os.path.exists(time_path) == False:
        os.makedirs(time_path)
    
    with open(obj_func_embeddings_path, "r") as f:
        obj_func_embeddings = json.load(f)
    with open(cdd_func_embeddings_path, "r") as f:
        cdd_func_embeddings = json.load(f)
    for func, embed in obj_func_embeddings.items():
        if func not in cdd_func_embeddings:
            cdd_func_embeddings[func] = embed
    
    # object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
    # cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
    obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
    cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()
    
    obj_com_funcs = {}
        # obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
    for obj_item in obj_com_funcs_file:
        # if obj_item.split("|||")[0] == object_name:
        if obj_item.split("|||")[0] not in obj_com_funcs:
            obj_com_funcs[obj_item.split("|||")[0]] = []
        if obj_item.split("|||")[1] not in obj_com_funcs[obj_item.split("|||")[0]]:
            obj_com_funcs[obj_item.split("|||")[0]].append(obj_item.split("|||")[1])
        
    cdd_com_funcs = {}
    for cdd_item in cdd_com_funcs_file:
        # if cdd_item.split("|||")[0] == candidate_name:
        # cdd_com_funcs[cdd_item.split("|||")[0]] = cdd_item.split("|||")[1]
        if cdd_item.split("|||")[0] not in cdd_com_funcs:
            cdd_com_funcs[cdd_item.split("|||")[0]] = []
        if cdd_item.split("|||")[1] not in cdd_com_funcs[cdd_item.split("|||")[0]]:
            cdd_com_funcs[cdd_item.split("|||")[0]].append(cdd_item.split("|||")[1])
    #     print("warning")
    # if object_item.split("__")[2] == candidate_name.split("__")[1]:
    #     continue
    
    # for obj_item in obj_com_funcs_file:
    #     for cdd_item in cdd_com_funcs_file:
    #         if obj_item.split("__")[2] == cdd_item.split("__")[1]:
    #             continue
    #     matched_func_list = cdd_project_dict[candidate_name]
        
        # json.dump(matched_func_list, open(os.path.join(sim_funcs_path, object_name+"___"+candidate_name+".json"), "w"))
        
    
    p_list = []
    Process_num = 1
    for i in range(Process_num):
        # tpl_detection_fast_one(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs, cdd_com_funcs)
        p = Process(target=reuse_area_detection_one, args=(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs, cdd_com_funcs, tpl_result, area_save_path))
        p_list.append(p)
            #args_list.append([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
            # compare_one_cdd_bin([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()


def reuse_area_detection_annoy(tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, area_save_path, time_path, com_funcs_path, sim_funcs_path, obj_func_embeddings_path, cdd_func_embeddings_path, gnn_model_path, tar_afcg_path, cdd_afcg_path, tar_subgraph_path, cdd_subgraph_path, tpl_result_path):
    cal_time = {}
    # object_item_list_temp = []
    
    tpl_result = json.load(open(tpl_result_path, "r"))
    
    if len(tpl_result) == 0:
        return False    
    
    object_item_list = os.listdir(func_path)
    
    
    tar_afcg_dict = {}
    for tar_afcg_item in os.listdir(tar_afcg_path):
        tar_bin_name = tar_afcg_item.split("_afcg.json")[0]
        tar_afcg_dict[tar_bin_name] = json.load(open(os.path.join(tar_afcg_path, tar_afcg_item), "r"))
    cdd_afcg_dict = {}
    for cdd_afcg_item in os.listdir(cdd_afcg_path):
        cdd_bin_name = cdd_afcg_item.split("_afcg.json")[0]
        cdd_afcg_dict[cdd_bin_name] = json.load(open(os.path.join(cdd_afcg_path, cdd_afcg_item), "r"))
    tar_subgraph_dict = {}
    for tar_afcg_item in os.listdir(tar_subgraph_path):
        tar_bin_name = tar_afcg_item.split("_subgraph.json")[0]
        tar_subgraph_dict[tar_bin_name] = json.load(open(os.path.join(tar_subgraph_path, tar_afcg_item), "r"))
    cdd_subgraph_dict = {}
    for cdd_afcg_item in os.listdir(cdd_subgraph_path):
        cdd_bin_name = cdd_afcg_item.split("_subgraph.json")[0]
        cdd_subgraph_dict[cdd_bin_name] = json.load(open(os.path.join(cdd_subgraph_path, cdd_afcg_item), "r"))
    
    # for object_item_list_item in object_item_list:
    #     if "bzip2" in object_item_list_item:
    #         object_item_list_temp.append(object_item_list_item)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gnn = False#torch.load(gnn_model_path)
    fcgs_num = {}
    #torch.multiprocessing.set_start_method('spawn', force=True)
    tar_fcg_dict = {}
    cdd_fcg_dict = {}
    for fcg_p in os.listdir(tar_fcg_path):
        with open(os.path.join(tar_fcg_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        tar_fcg_dict[fcg_p.split("_fcg.pkl")[0]] = fcg
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
    for fcg_p in os.listdir(cdd_fcg_path):
        with open(os.path.join(cdd_fcg_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        cdd_fcg_dict[fcg_p.split("_fcg.pkl")[0]] = fcg
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
        
    if os.path.exists(feature_save_path) == False:
        os.makedirs(feature_save_path)
    if os.path.exists(area_save_path) == False:
        os.makedirs(area_save_path)
    if os.path.exists(sim_funcs_path) == False:
        os.makedirs(sim_funcs_path)
    if os.path.exists(time_path) == False:
        os.makedirs(time_path)
    
    with open(obj_func_embeddings_path, "r") as f:
        obj_func_embeddings = json.load(f)
    with open(cdd_func_embeddings_path, "r") as f:
        cdd_func_embeddings = json.load(f)
    for func, embed in obj_func_embeddings.items():
        if func not in cdd_func_embeddings:
            cdd_func_embeddings[func] = embed
    
    # object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
    # cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
    obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
    cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()
    
    obj_com_funcs = {}
        # obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
    for obj_item in obj_com_funcs_file:
        # if obj_item.split("|||")[0] == object_name:
        if obj_item.split("|||")[0] not in obj_com_funcs:
            obj_com_funcs[obj_item.split("|||")[0]] = []
        if obj_item.split("|||")[1] not in obj_com_funcs[obj_item.split("|||")[0]]:
            obj_com_funcs[obj_item.split("|||")[0]].append(obj_item.split("|||")[1])
        
    cdd_com_funcs = {}
    for cdd_item in cdd_com_funcs_file:
        # if cdd_item.split("|||")[0] == candidate_name:
        # cdd_com_funcs[cdd_item.split("|||")[0]] = cdd_item.split("|||")[1]
        if cdd_item.split("|||")[0] not in cdd_com_funcs:
            cdd_com_funcs[cdd_item.split("|||")[0]] = []
        if cdd_item.split("|||")[1] not in cdd_com_funcs[cdd_item.split("|||")[0]]:
            cdd_com_funcs[cdd_item.split("|||")[0]].append(cdd_item.split("|||")[1])
    #     print("warning")
    # if object_item.split("__")[2] == candidate_name.split("__")[1]:
    #     continue
    
    # for obj_item in obj_com_funcs_file:
    #     for cdd_item in cdd_com_funcs_file:
    #         if obj_item.split("__")[2] == cdd_item.split("__")[1]:
    #             continue
    #     matched_func_list = cdd_project_dict[candidate_name]
        
        # json.dump(matched_func_list, open(os.path.join(sim_funcs_path, object_name+"___"+candidate_name+".json"), "w"))
        
    
    p_list = []
    Process_num = 30
    for i in range(Process_num):
        # reuse_area_detection_one_annoy(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs, cdd_com_funcs, area_save_path, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, tar_fcg_dict, cdd_fcg_dict, tpl_result)
        p = Process(target=reuse_area_detection_one_annoy, args=(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs, cdd_com_funcs, area_save_path, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, tar_fcg_dict, cdd_fcg_dict, tpl_result))
        p_list.append(p)
            #args_list.append([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
            # compare_one_cdd_bin([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()

def tpl_detection_fast_annoy_without_align(tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, area_save_path, time_path, com_funcs_path, sim_funcs_path, obj_func_embeddings_path, cdd_func_embeddings_path, gnn_model_path, tar_afcg_path, cdd_afcg_path, tar_subgraph_path, cdd_subgraph_path):
    cal_time = {}
    # object_item_list_temp = []
    object_item_list = os.listdir(func_path)
    
    tar_afcg_dict = {}
    for tar_afcg_item in os.listdir(tar_afcg_path):
        tar_bin_name = tar_afcg_item.split("_afcg.json")[0]
        tar_afcg_dict[tar_bin_name] = json.load(open(os.path.join(tar_afcg_path, tar_afcg_item), "r"))
    cdd_afcg_dict = {}
    for cdd_afcg_item in os.listdir(cdd_afcg_path):
        cdd_bin_name = cdd_afcg_item.split("_afcg.json")[0]
        cdd_afcg_dict[cdd_bin_name] = json.load(open(os.path.join(cdd_afcg_path, cdd_afcg_item), "r"))
    tar_subgraph_dict = {}
    for tar_afcg_item in os.listdir(tar_subgraph_path):
        tar_bin_name = tar_afcg_item.split("_subgraph.json")[0]
        tar_subgraph_dict[tar_bin_name] = json.load(open(os.path.join(tar_subgraph_path, tar_afcg_item), "r"))
    cdd_subgraph_dict = {}
    for cdd_afcg_item in os.listdir(cdd_subgraph_path):
        cdd_bin_name = cdd_afcg_item.split("_subgraph.json")[0]
        cdd_subgraph_dict[cdd_bin_name] = json.load(open(os.path.join(cdd_subgraph_path, cdd_afcg_item), "r"))
    
    # for object_item_list_item in object_item_list:
    #     if "bzip2" in object_item_list_item:
    #         object_item_list_temp.append(object_item_list_item)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gnn = False#torch.load(gnn_model_path)
    fcgs_num = {}
    #torch.multiprocessing.set_start_method('spawn', force=True)
    tar_fcg_dict = {}
    cdd_fcg_dict = {}
    for fcg_p in os.listdir(tar_fcg_path):
        with open(os.path.join(tar_fcg_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        tar_fcg_dict[fcg_p.split("_fcg.pkl")[0]] = fcg
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
    for fcg_p in os.listdir(cdd_fcg_path):
        with open(os.path.join(cdd_fcg_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        cdd_fcg_dict[fcg_p.split("_fcg.pkl")[0]] = fcg
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
        
    if os.path.exists(feature_save_path) == False:
        os.makedirs(feature_save_path)
    if os.path.exists(area_save_path) == False:
        os.makedirs(area_save_path)
    if os.path.exists(sim_funcs_path) == False:
        os.makedirs(sim_funcs_path)
    if os.path.exists(time_path) == False:
        os.makedirs(time_path)
    
    with open(obj_func_embeddings_path, "r") as f:
        obj_func_embeddings = json.load(f)
    with open(cdd_func_embeddings_path, "r") as f:
        cdd_func_embeddings = json.load(f)
    for func, embed in obj_func_embeddings.items():
        if func not in cdd_func_embeddings:
            cdd_func_embeddings[func] = embed
    
    # object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
    # cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
    obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
    cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()
    
    obj_com_funcs = {}
        # obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
    for obj_item in obj_com_funcs_file:
        # if obj_item.split("|||")[0] == object_name:
        if obj_item.split("|||")[0] not in obj_com_funcs:
            obj_com_funcs[obj_item.split("|||")[0]] = []
        if obj_item.split("|||")[1] not in obj_com_funcs[obj_item.split("|||")[0]]:
            obj_com_funcs[obj_item.split("|||")[0]].append(obj_item.split("|||")[1])
        
    cdd_com_funcs = {}
    for cdd_item in cdd_com_funcs_file:
        # if cdd_item.split("|||")[0] == candidate_name:
        # cdd_com_funcs[cdd_item.split("|||")[0]] = cdd_item.split("|||")[1]
        if cdd_item.split("|||")[0] not in cdd_com_funcs:
            cdd_com_funcs[cdd_item.split("|||")[0]] = []
        if cdd_item.split("|||")[1] not in cdd_com_funcs[cdd_item.split("|||")[0]]:
            cdd_com_funcs[cdd_item.split("|||")[0]].append(cdd_item.split("|||")[1])
    #     print("warning")
    # if object_item.split("__")[2] == candidate_name.split("__")[1]:
    #     continue
    
    # for obj_item in obj_com_funcs_file:
    #     for cdd_item in cdd_com_funcs_file:
    #         if obj_item.split("__")[2] == cdd_item.split("__")[1]:
    #             continue
    #     matched_func_list = cdd_project_dict[candidate_name]
        
        # json.dump(matched_func_list, open(os.path.join(sim_funcs_path, object_name+"___"+candidate_name+".json"), "w"))
        
    
    p_list = []
    Process_num = 35
    for i in range(Process_num):
        # tpl_detection_fast_one_annoy(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs, cdd_com_funcs, area_save_path, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, tar_fcg_dict, cdd_fcg_dict)
        p = Process(target=tpl_detection_fast_one_annoy_without_align, args=(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs, cdd_com_funcs, area_save_path, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, tar_fcg_dict, cdd_fcg_dict))
        p_list.append(p)
            #args_list.append([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
            # compare_one_cdd_bin([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()



def tpl_detection_fast_annoy_without_gnn(tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, area_save_path, time_path, com_funcs_path, sim_funcs_path, obj_func_embeddings_path, cdd_func_embeddings_path, gnn_model_path, tar_afcg_path, cdd_afcg_path, tar_subgraph_path, cdd_subgraph_path):
    cal_time = {}
    # object_item_list_temp = []
    object_item_list = os.listdir(func_path)
    
    tar_afcg_dict = {}
    for tar_afcg_item in os.listdir(tar_afcg_path):
        tar_bin_name = tar_afcg_item.split("_afcg.json")[0]
        tar_afcg_dict[tar_bin_name] = json.load(open(os.path.join(tar_afcg_path, tar_afcg_item), "r"))
    cdd_afcg_dict = {}
    for cdd_afcg_item in os.listdir(cdd_afcg_path):
        cdd_bin_name = cdd_afcg_item.split("_afcg.json")[0]
        cdd_afcg_dict[cdd_bin_name] = json.load(open(os.path.join(cdd_afcg_path, cdd_afcg_item), "r"))
    tar_subgraph_dict = {}
    for tar_afcg_item in os.listdir(tar_subgraph_path):
        tar_bin_name = tar_afcg_item.split("_subgraph.json")[0]
        tar_subgraph_dict[tar_bin_name] = json.load(open(os.path.join(tar_subgraph_path, tar_afcg_item), "r"))
    cdd_subgraph_dict = {}
    for cdd_afcg_item in os.listdir(cdd_subgraph_path):
        cdd_bin_name = cdd_afcg_item.split("_subgraph.json")[0]
        cdd_subgraph_dict[cdd_bin_name] = json.load(open(os.path.join(cdd_subgraph_path, cdd_afcg_item), "r"))
    
    # for object_item_list_item in object_item_list:
    #     if "bzip2" in object_item_list_item:
    #         object_item_list_temp.append(object_item_list_item)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gnn = False#torch.load(gnn_model_path)
    fcgs_num = {}
    #torch.multiprocessing.set_start_method('spawn', force=True)
    tar_fcg_dict = {}
    cdd_fcg_dict = {}
    for fcg_p in os.listdir(tar_fcg_path):
        with open(os.path.join(tar_fcg_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        tar_fcg_dict[fcg_p.split("_fcg.pkl")[0]] = fcg
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
    for fcg_p in os.listdir(cdd_fcg_path):
        with open(os.path.join(cdd_fcg_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        cdd_fcg_dict[fcg_p.split("_fcg.pkl")[0]] = fcg
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
        
    if os.path.exists(feature_save_path) == False:
        os.makedirs(feature_save_path)
    if os.path.exists(area_save_path) == False:
        os.makedirs(area_save_path)
    if os.path.exists(sim_funcs_path) == False:
        os.makedirs(sim_funcs_path)
    if os.path.exists(time_path) == False:
        os.makedirs(time_path)
    
    with open(obj_func_embeddings_path, "r") as f:
        obj_func_embeddings = json.load(f)
    with open(cdd_func_embeddings_path, "r") as f:
        cdd_func_embeddings = json.load(f)
    for func, embed in obj_func_embeddings.items():
        if func not in cdd_func_embeddings:
            cdd_func_embeddings[func] = embed
    
    # object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
    # cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
    obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
    cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()
    
    obj_com_funcs = {}
        # obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
    for obj_item in obj_com_funcs_file:
        # if obj_item.split("|||")[0] == object_name:
        if obj_item.split("|||")[0] not in obj_com_funcs:
            obj_com_funcs[obj_item.split("|||")[0]] = []
        if obj_item.split("|||")[1] not in obj_com_funcs[obj_item.split("|||")[0]]:
            obj_com_funcs[obj_item.split("|||")[0]].append(obj_item.split("|||")[1])
        
    cdd_com_funcs = {}
    for cdd_item in cdd_com_funcs_file:
        # if cdd_item.split("|||")[0] == candidate_name:
        # cdd_com_funcs[cdd_item.split("|||")[0]] = cdd_item.split("|||")[1]
        if cdd_item.split("|||")[0] not in cdd_com_funcs:
            cdd_com_funcs[cdd_item.split("|||")[0]] = []
        if cdd_item.split("|||")[1] not in cdd_com_funcs[cdd_item.split("|||")[0]]:
            cdd_com_funcs[cdd_item.split("|||")[0]].append(cdd_item.split("|||")[1])
    #     print("warning")
    # if object_item.split("__")[2] == candidate_name.split("__")[1]:
    #     continue
    
    # for obj_item in obj_com_funcs_file:
    #     for cdd_item in cdd_com_funcs_file:
    #         if obj_item.split("__")[2] == cdd_item.split("__")[1]:
    #             continue
    #     matched_func_list = cdd_project_dict[candidate_name]
        
        # json.dump(matched_func_list, open(os.path.join(sim_funcs_path, object_name+"___"+candidate_name+".json"), "w"))
        
    
    p_list = []
    Process_num = 35
    for i in range(Process_num):
        # tpl_detection_fast_one_annoy(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs, cdd_com_funcs, area_save_path, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, tar_fcg_dict, cdd_fcg_dict)
        p = Process(target=tpl_detection_fast_one_annoy_without_gnn, args=(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs, cdd_com_funcs, area_save_path, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, tar_fcg_dict, cdd_fcg_dict))
        p_list.append(p)
            #args_list.append([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
            # compare_one_cdd_bin([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()

def tpl_detection_fast_annoy_1_5(tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, area_save_path, time_path, com_funcs_path, sim_funcs_path, obj_func_embeddings_path, cdd_func_embeddings_path, gnn_model_path, tar_afcg_path, cdd_afcg_path, tar_subgraph_path, cdd_subgraph_path, alignment_tred):
    cal_time = {}
    # object_item_list_temp = []
    object_item_list = os.listdir(func_path)
    
    tar_afcg_dict = {}
    for tar_afcg_item in os.listdir(tar_afcg_path):
        tar_bin_name = tar_afcg_item.split("_afcg.json")[0]
        tar_afcg_dict[tar_bin_name] = json.load(open(os.path.join(tar_afcg_path, tar_afcg_item), "r"))
    cdd_afcg_dict = {}
    for cdd_afcg_item in os.listdir(cdd_afcg_path):
        cdd_bin_name = cdd_afcg_item.split("_afcg.json")[0]
        cdd_afcg_dict[cdd_bin_name] = json.load(open(os.path.join(cdd_afcg_path, cdd_afcg_item), "r"))
    tar_subgraph_dict = {}
    for tar_afcg_item in os.listdir(tar_subgraph_path):
        tar_bin_name = tar_afcg_item.split("_subgraph.json")[0]
        tar_subgraph_dict[tar_bin_name] = json.load(open(os.path.join(tar_subgraph_path, tar_afcg_item), "r"))
    cdd_subgraph_dict = {}
    for cdd_afcg_item in os.listdir(cdd_subgraph_path):
        cdd_bin_name = cdd_afcg_item.split("_subgraph.json")[0]
        cdd_subgraph_dict[cdd_bin_name] = json.load(open(os.path.join(cdd_subgraph_path, cdd_afcg_item), "r"))
    
    # for object_item_list_item in object_item_list:
    #     if "bzip2" in object_item_list_item:
    #         object_item_list_temp.append(object_item_list_item)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gnn = False#torch.load(gnn_model_path)
    fcgs_num = {}
    #torch.multiprocessing.set_start_method('spawn', force=True)
    tar_fcg_dict = {}
    cdd_fcg_dict = {}
    for fcg_p in os.listdir(tar_fcg_path):
        with open(os.path.join(tar_fcg_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        tar_fcg_dict[fcg_p.split("_fcg.pkl")[0]] = fcg
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
    for fcg_p in os.listdir(cdd_fcg_path):
        with open(os.path.join(cdd_fcg_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        cdd_fcg_dict[fcg_p.split("_fcg.pkl")[0]] = fcg
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
        
    if os.path.exists(feature_save_path) == False:
        os.makedirs(feature_save_path)
    if os.path.exists(area_save_path) == False:
        os.makedirs(area_save_path)
    if os.path.exists(sim_funcs_path) == False:
        os.makedirs(sim_funcs_path)
    if os.path.exists(time_path) == False:
        os.makedirs(time_path)
    
    with open(obj_func_embeddings_path, "r") as f:
        obj_func_embeddings = json.load(f)
    with open(cdd_func_embeddings_path, "r") as f:
        cdd_func_embeddings = json.load(f)
    for func, embed in obj_func_embeddings.items():
        if func not in cdd_func_embeddings:
            cdd_func_embeddings[func] = embed
    
    # object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
    # cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
    obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
    cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()
    
    obj_com_funcs = {}
        # obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
    for obj_item in obj_com_funcs_file:
        # if obj_item.split("|||")[0] == object_name:
        if obj_item.split("|||")[0] not in obj_com_funcs:
            obj_com_funcs[obj_item.split("|||")[0]] = []
        if obj_item.split("|||")[1] not in obj_com_funcs[obj_item.split("|||")[0]]:
            obj_com_funcs[obj_item.split("|||")[0]].append(obj_item.split("|||")[1])
        
    cdd_com_funcs = {}
    for cdd_item in cdd_com_funcs_file:
        # if cdd_item.split("|||")[0] == candidate_name:
        # cdd_com_funcs[cdd_item.split("|||")[0]] = cdd_item.split("|||")[1]
        if cdd_item.split("|||")[0] not in cdd_com_funcs:
            cdd_com_funcs[cdd_item.split("|||")[0]] = []
        if cdd_item.split("|||")[1] not in cdd_com_funcs[cdd_item.split("|||")[0]]:
            cdd_com_funcs[cdd_item.split("|||")[0]].append(cdd_item.split("|||")[1])
    #     print("warning")
    # if object_item.split("__")[2] == candidate_name.split("__")[1]:
    #     continue
    
    # for obj_item in obj_com_funcs_file:
    #     for cdd_item in cdd_com_funcs_file:
    #         if obj_item.split("__")[2] == cdd_item.split("__")[1]:
    #             continue
    #     matched_func_list = cdd_project_dict[candidate_name]
        
        # json.dump(matched_func_list, open(os.path.join(sim_funcs_path, object_name+"___"+candidate_name+".json"), "w"))
        
    
    p_list = []
    Process_num = 35
    for i in range(Process_num):
        # tpl_detection_fast_one_annoy(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs, cdd_com_funcs, area_save_path, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, tar_fcg_dict, cdd_fcg_dict)
        p = Process(target=tpl_detection_fast_one_annoy_1_5, args=(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs, cdd_com_funcs, area_save_path, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, tar_fcg_dict, cdd_fcg_dict, alignment_tred))
        p_list.append(p)
            #args_list.append([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
            # compare_one_cdd_bin([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()

def tpl_detection_fast_one_annoy_simple_with_logging(func_path_list, tar_fcg_path, cdd_fcg_path, func_path, 
                                                      feature_save_path, time_path, com_funcs_path, sim_funcs_path, 
                                                      gnn, fcgs_num, obj_com_funcs_dict, 
                                                      cdd_com_funcs_dict, tar_afcg_dict, tar_subgraph_dict, 
                                                      cdd_afcg_dict, cdd_subgraph_dict, area_save_path,
                                                      cdd_func_embeddings_path, process_id, memory_log_path):
    """
    Worker with memory logging written to disk (no queue deadlock).
    """
    import psutil
    import os
    import sys
    
    process = psutil.Process(os.getpid())
    memory_log = []
    
    print(f"Worker {process_id} starting", flush=True)
    sys.stdout.flush()
    
    with open(cdd_func_embeddings_path, "r") as f:
        cdd_func_embeddings = json.load(f)
    
    for object_item in func_path_list:
        object_name = object_item.split("_reuse_func_dict")[0]
        reuse_result = {object_name: []}

        mem_start = process.memory_info().rss / 1024 / 1024
        
        try:
            with open(os.path.join(tar_fcg_path, f"{object_name}_fcg.pkl"), "rb") as f:
                object_fcg = pickle.load(f)
        except FileNotFoundError:
            continue
        
        mem_after_tar_load = process.memory_info().rss / 1024 / 1024
        
        try:
            object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        except (FileNotFoundError, json.JSONDecodeError):
            del object_fcg
            continue
        
        cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
        obj_com_funcs = obj_com_funcs_dict.get(object_name, [])
        
        for candidate_name in cdd_project_dict:
            if candidate_name not in cdd_com_funcs_dict:
                continue
            if object_name not in tar_afcg_dict or object_name not in tar_subgraph_dict:
                continue
            if candidate_name not in cdd_afcg_dict or candidate_name not in cdd_subgraph_dict:
                continue
            
            mem_before_cdd_load = process.memory_info().rss / 1024 / 1024
            
            cdd_com_funcs = cdd_com_funcs_dict.get(candidate_name, [])
            matched_func_list = cdd_project_dict[candidate_name]
            
            json.dump(matched_func_list, open(os.path.join(sim_funcs_path, 
                     f"{object_name}___{candidate_name}.json"), "w"))
            
            try:
                with open(os.path.join(cdd_fcg_path, f"{candidate_name}_fcg.pkl"), "rb") as f:
                    candidate_fcg = pickle.load(f)
            except FileNotFoundError:
                continue
            
            mem_after_cdd_load = process.memory_info().rss / 1024 / 1024
            
            mem_before_detection = process.memory_info().rss / 1024 / 1024
            
            reuse_flag, reuse_dict = tpl_detection_fast_core_annoy(
                object_name, candidate_name, object_fcg, matched_func_list, 
                candidate_fcg, obj_com_funcs, cdd_com_funcs, cdd_func_embeddings, 
                gnn, fcgs_num, tar_afcg_dict[object_name], cdd_afcg_dict[candidate_name], 
                tar_subgraph_dict[object_name], cdd_subgraph_dict[candidate_name])
            
            mem_after_detection = process.memory_info().rss / 1024 / 1024
            
            if reuse_flag:
                reuse_result[object_name].append(candidate_name)
                with open(os.path.join(area_save_path, 
                         f"{object_name}___{candidate_name}_feature_result.json"), "w") as ff:
                    json.dump(reuse_dict, ff)
            
            mem_after_save = process.memory_info().rss / 1024 / 1024
            
            # Log this iteration
            memory_log.append({
                "process_id": process_id,
                "object_name": object_name,
                "candidate_name": candidate_name,
                "delta_cdd_load": mem_after_cdd_load - mem_before_cdd_load,
                "delta_detection": mem_after_detection - mem_before_detection,
                "delta_save": mem_after_save - mem_after_detection,
                "peak_mem": mem_after_detection,
            })
            
            del candidate_fcg
            del reuse_dict
            gc.collect()

        result_path = os.path.join(feature_save_path, f"{object_name}_reuse_result.json")

        # Merge with any results already written by a previous phase
        if os.path.exists(result_path):
            try:
                with open(result_path, "r") as f:
                    existing = json.load(f)
                for k, v in existing.items():
                    if k not in reuse_result:
                        reuse_result[k] = v
                    else:
                        # Merge candidate lists without duplicates
                        reuse_result[k] = list(set(reuse_result[k]) | set(v))
            except (json.JSONDecodeError, KeyError):
                pass  # corrupt or empty file, just overwrite

        json.dump(reuse_result, open(result_path, "w"))
        
        del object_fcg
        del object_cdd_func_dict
        del cdd_project_dict
        gc.collect()
    
    # Write memory log to disk (no queue)
    print(f"Worker {process_id} writing memory log to {memory_log_path}_{process_id}.json", flush=True)
    sys.stdout.flush()
    with open(f"{memory_log_path}_{process_id}.json", "w") as f:
        json.dump(memory_log, f, indent=2)
    
    print(f"Worker {process_id} done", flush=True)
    sys.stdout.flush()

def tpl_detection_fast_annoy_simple_with_logging(tar_fcg_path, cdd_fcg_path, func_path,
                                                  feature_save_path, area_save_path, time_path,
                                                  com_funcs_path, sim_funcs_path,
                                                  obj_func_embeddings_path, cdd_func_embeddings_path,
                                                  gnn_model_path, tar_afcg_path, cdd_afcg_path,
                                                  tar_subgraph_path, cdd_subgraph_path):
    """
    Phase-based processing with memory diagnostics.
    """
    import gc
    import psutil
    import sys

    main_process = psutil.Process(os.getpid())

    object_item_list = os.listdir(func_path)

    print("Loading target metadata...", flush=True)
    tar_afcg_dict = {}
    for tar_afcg_item in os.listdir(tar_afcg_path):
        tar_bin_name = tar_afcg_item.split("_afcg.json")[0]
        tar_afcg_dict[tar_bin_name] = json.load(open(os.path.join(tar_afcg_path, tar_afcg_item), "r"))

    tar_subgraph_dict = {}
    for tar_subgraph_item in os.listdir(tar_subgraph_path):
        tar_bin_name = tar_subgraph_item.split("_subgraph.json")[0]
        tar_subgraph_dict[tar_bin_name] = json.load(open(os.path.join(tar_subgraph_path, tar_subgraph_item), "r"))

    cdd_subgraph_files = sorted(os.listdir(cdd_subgraph_path))
    print(f"Found {len(cdd_subgraph_files)} candidates to process")

    for path in [feature_save_path, area_save_path, sim_funcs_path, time_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    print("Pre-computing FCG node counts...", flush=True)
    fcgs_num = {}
    for fcg_p in os.listdir(tar_fcg_path):
        try:
            with open(os.path.join(tar_fcg_path, fcg_p), "rb") as f:
                fcg = pickle.load(f)
            fcg_name = fcg_p.split("_fcg.pkl")[0]
            fcgs_num[fcg_name] = len(list(fcg.nodes()))
            del fcg
        except (FileNotFoundError, pickle.PickleError):
            continue

    for fcg_p in os.listdir(cdd_fcg_path):
        try:
            with open(os.path.join(cdd_fcg_path, fcg_p), "rb") as f:
                fcg = pickle.load(f)
            fcg_name = fcg_p.split("_fcg.pkl")[0]
            fcgs_num[fcg_name] = len(list(fcg.nodes()))
            del fcg
        except (FileNotFoundError, pickle.PickleError):
            continue

    gc.collect()

    print("Loading metadata...", flush=True)
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

    CANDIDATES_PER_PHASE = 3
    num_phases = (len(cdd_subgraph_files) + CANDIDATES_PER_PHASE - 1) // CANDIDATES_PER_PHASE

    print(f"\nProcessing {num_phases} phases with memory diagnostics", flush=True)
    print("="*70, flush=True)

    memory_log_base = os.path.join(time_path, "worker_memory")
    phase_memory_log = []

    for phase in range(num_phases):
        print(f"\n[Phase {phase + 1}/{num_phases}]", flush=True)

        phase_start = phase * CANDIDATES_PER_PHASE
        phase_end = min((phase + 1) * CANDIDATES_PER_PHASE, len(cdd_subgraph_files))
        phase_subgraph_files = cdd_subgraph_files[phase_start:phase_end]

        print(f"Loading {len(phase_subgraph_files)} candidate subgraphs...", flush=True)
        mem_before_load = main_process.memory_info().rss / 1024 / 1024

        cdd_afcg_dict = {}
        cdd_afcg_total_size = 0
        for cdd_subgraph_item in phase_subgraph_files:
            cdd_bin_name = cdd_subgraph_item.split("_subgraph.json")[0]
            afcg_file = f"{cdd_bin_name}_afcg.json"
            try:
                data = json.load(open(os.path.join(cdd_afcg_path, afcg_file), "r"))
                cdd_afcg_dict[cdd_bin_name] = data
                # Estimate size
                cdd_afcg_total_size += len(str(data)) / 1024 / 1024
            except (FileNotFoundError, json.JSONDecodeError):
                continue

        cdd_subgraph_dict = {}
        cdd_subgraph_total_size = 0
        for cdd_subgraph_item in phase_subgraph_files:
            cdd_bin_name = cdd_subgraph_item.split("_subgraph.json")[0]
            try:
                data = json.load(open(os.path.join(cdd_subgraph_path, cdd_subgraph_item), "r"))
                cdd_subgraph_dict[cdd_bin_name] = data
                # Estimate size
                cdd_subgraph_total_size += len(str(data)) / 1024 / 1024
            except (FileNotFoundError, json.JSONDecodeError):
                continue

        mem_after_load = main_process.memory_info().rss / 1024 / 1024
        mem_delta_load = mem_after_load - mem_before_load

        print(f"  Loaded {len(cdd_afcg_dict)} AFCG (est. {cdd_afcg_total_size:.2f} MB)")
        print(f"  Loaded {len(cdd_subgraph_dict)} subgraph (est. {cdd_subgraph_total_size:.2f} MB)")
        print(f"  Memory delta: {mem_delta_load:.2f} MB (before: {mem_before_load:.2f}, after: {mem_after_load:.2f})")

        # Find largest entries
        if cdd_subgraph_dict:
            subgraph_sizes = [(k, len(str(v)) / 1024 / 1024) for k, v in cdd_subgraph_dict.items()]
            subgraph_sizes.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top 3 largest subgraphs:")
            for name, size in subgraph_sizes[:3]:
                print(f"    {name}: {size:.2f} MB")

        if cdd_afcg_dict:
            afcg_sizes = [(k, len(str(v)) / 1024 / 1024) for k, v in cdd_afcg_dict.items()]
            afcg_sizes.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top 3 largest AFCG:")
            for name, size in afcg_sizes[:3]:
                print(f"    {name}: {size:.2f} MB")

        phase_memory_log.append({
            "phase": phase + 1,
            "num_candidates": len(phase_subgraph_files),
            "mem_before_mb": mem_before_load,
            "mem_after_mb": mem_after_load,
            "delta_mb": mem_delta_load,
            "estimated_afcg_mb": cdd_afcg_total_size,
            "estimated_subgraph_mb": cdd_subgraph_total_size,
            "estimated_total_mb": cdd_afcg_total_size + cdd_subgraph_total_size,
        })

        print(f"Starting workers for phase {phase + 1}...", flush=True)

        p_list = []
        Process_num = 2 

        ctx = multiprocessing.get_context('fork')

        for i in range(Process_num):
            start_idx = int((i / Process_num) * len(object_item_list))
            end_idx = int(((i + 1) / Process_num) * len(object_item_list))
            func_path_slice = object_item_list[start_idx:end_idx]

            if len(func_path_slice) == 0:
                continue

            p = ctx.Process(target=tpl_detection_fast_one_annoy_simple_with_logging,
                           args=(func_path_slice, tar_fcg_path, cdd_fcg_path, func_path,
                                 feature_save_path, time_path, com_funcs_path, sim_funcs_path,
                                 gnn, fcgs_num, obj_com_funcs_dict,
                                 cdd_com_funcs_dict, tar_afcg_dict, tar_subgraph_dict,
                                 cdd_afcg_dict, cdd_subgraph_dict, area_save_path,
                                 cdd_func_embeddings_path, i, memory_log_base))
            p_list.append(p)
            p.start()

        for idx, p in enumerate(p_list):
            p.join()

        for p in p_list:
            p.close()

        p_list.clear()

        print(f"Phase {phase + 1} complete.", flush=True)
        mem_before_cleanup = main_process.memory_info().rss / 1024 / 1024

        del cdd_afcg_dict
        del cdd_subgraph_dict
        gc.collect()

        mem_after_cleanup = main_process.memory_info().rss / 1024 / 1024
        print(f"Memory after cleanup: {mem_after_cleanup:.2f} MB (freed {mem_before_cleanup - mem_after_cleanup:.2f} MB)")

    # Write phase memory log
    with open(os.path.join(time_path, "phase_memory_profile.json"), "w") as f:
        json.dump(phase_memory_log, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("PHASE MEMORY SUMMARY")
    print("="*70)
    print(f"{'Phase':6s} {'Candidates':12s} {'Est. Size':12s} {'Actual Delta':12s} {'Ratio':10s}")
    print("-" * 60)

    for log in phase_memory_log:
        est = log["estimated_total_mb"]
        actual = log["delta_mb"]
        ratio = actual / est if est > 0 else 0
        print(f"{log['phase']:6d} {log['num_candidates']:12d} {est:12.2f} MB {actual:12.2f} MB {ratio:10.2f}x")

import multiprocessing
import gc


def tpl_detection_fast_annoy(tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, area_save_path, time_path, com_funcs_path, sim_funcs_path, obj_func_embeddings_path, cdd_func_embeddings_path, gnn_model_path, tar_afcg_path, cdd_afcg_path, tar_subgraph_path, cdd_subgraph_path):
    cal_time = {}
    # object_item_list_temp = []
    object_item_list = os.listdir(func_path)
    
    tar_afcg_dict = {}
    for tar_afcg_item in os.listdir(tar_afcg_path):
        tar_bin_name = tar_afcg_item.split("_afcg.json")[0]
        tar_afcg_dict[tar_bin_name] = json.load(open(os.path.join(tar_afcg_path, tar_afcg_item), "r"))
    cdd_afcg_dict = {}
    for cdd_afcg_item in os.listdir(cdd_afcg_path):
        cdd_bin_name = cdd_afcg_item.split("_afcg.json")[0]
        cdd_afcg_dict[cdd_bin_name] = json.load(open(os.path.join(cdd_afcg_path, cdd_afcg_item), "r"))
    tar_subgraph_dict = {}
    for tar_afcg_item in os.listdir(tar_subgraph_path):
        tar_bin_name = tar_afcg_item.split("_subgraph.json")[0]
        tar_subgraph_dict[tar_bin_name] = json.load(open(os.path.join(tar_subgraph_path, tar_afcg_item), "r"))
    cdd_subgraph_dict = {}
    for cdd_afcg_item in os.listdir(cdd_subgraph_path):
        cdd_bin_name = cdd_afcg_item.split("_subgraph.json")[0]
        cdd_subgraph_dict[cdd_bin_name] = json.load(open(os.path.join(cdd_subgraph_path, cdd_afcg_item), "r"))
    
    # for object_item_list_item in object_item_list:
    #     if "bzip2" in object_item_list_item:
    #         object_item_list_temp.append(object_item_list_item)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gnn = False#torch.load(gnn_model_path)
    fcgs_num = {}
    #torch.multiprocessing.set_start_method('spawn', force=True)
    tar_fcg_dict = {}
    cdd_fcg_dict = {}
    for fcg_p in os.listdir(tar_fcg_path):
        with open(os.path.join(tar_fcg_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        tar_fcg_dict[fcg_p.split("_fcg.pkl")[0]] = fcg
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
    for fcg_p in os.listdir(cdd_fcg_path):
        with open(os.path.join(cdd_fcg_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        cdd_fcg_dict[fcg_p.split("_fcg.pkl")[0]] = fcg
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
        
    if os.path.exists(feature_save_path) == False:
        os.makedirs(feature_save_path)
    if os.path.exists(area_save_path) == False:
        os.makedirs(area_save_path)
    if os.path.exists(sim_funcs_path) == False:
        os.makedirs(sim_funcs_path)
    if os.path.exists(time_path) == False:
        os.makedirs(time_path)
    
    with open(obj_func_embeddings_path, "r") as f:
        obj_func_embeddings = json.load(f)
    with open(cdd_func_embeddings_path, "r") as f:
        cdd_func_embeddings = json.load(f)
    for func, embed in obj_func_embeddings.items():
        if func not in cdd_func_embeddings:
            cdd_func_embeddings[func] = embed
    
    # object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
    # cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
    obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
    cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()
    
    obj_com_funcs = {}
        # obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
    for obj_item in obj_com_funcs_file:
        # if obj_item.split("|||")[0] == object_name:
        if obj_item.split("|||")[0] not in obj_com_funcs:
            obj_com_funcs[obj_item.split("|||")[0]] = []
        if obj_item.split("|||")[1] not in obj_com_funcs[obj_item.split("|||")[0]]:
            obj_com_funcs[obj_item.split("|||")[0]].append(obj_item.split("|||")[1])
        
    cdd_com_funcs = {}
    for cdd_item in cdd_com_funcs_file:
        # if cdd_item.split("|||")[0] == candidate_name:
        # cdd_com_funcs[cdd_item.split("|||")[0]] = cdd_item.split("|||")[1]
        if cdd_item.split("|||")[0] not in cdd_com_funcs:
            cdd_com_funcs[cdd_item.split("|||")[0]] = []
        if cdd_item.split("|||")[1] not in cdd_com_funcs[cdd_item.split("|||")[0]]:
            cdd_com_funcs[cdd_item.split("|||")[0]].append(cdd_item.split("|||")[1])
    #     print("warning")
    # if object_item.split("__")[2] == candidate_name.split("__")[1]:
    #     continue
    
    # for obj_item in obj_com_funcs_file:
    #     for cdd_item in cdd_com_funcs_file:
    #         if obj_item.split("__")[2] == cdd_item.split("__")[1]:
    #             continue
    #     matched_func_list = cdd_project_dict[candidate_name]
        
        # json.dump(matched_func_list, open(os.path.join(sim_funcs_path, object_name+"___"+candidate_name+".json"), "w"))
        
    
    p_list = []
    Process_num = 35
    for i in range(Process_num):
        # tpl_detection_fast_one_annoy(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs, cdd_com_funcs, area_save_path, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, tar_fcg_dict, cdd_fcg_dict)
        p = Process(target=tpl_detection_fast_one_annoy, args=(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs, cdd_com_funcs, area_save_path, tar_afcg_dict, cdd_afcg_dict, tar_subgraph_dict, cdd_subgraph_dict, tar_fcg_dict, cdd_fcg_dict))
        p_list.append(p)
            #args_list.append([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
            # compare_one_cdd_bin([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()





def tpl_detection_fast(tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, area_save_path, time_path, com_funcs_path, sim_funcs_path, obj_func_embeddings_path, cdd_func_embeddings_path, gnn_model_path):
    cal_time = {}
    # object_item_list_temp = []
    object_item_list = os.listdir(func_path)
    
    # for object_item_list_item in object_item_list:
    #     if "bzip2" in object_item_list_item:
    #         object_item_list_temp.append(object_item_list_item)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gnn = torch.load(gnn_model_path, weights_only=False)
    fcgs_num = {}
    torch.multiprocessing.set_start_method('spawn', force=True)
    for fcg_p in os.listdir(tar_fcg_path):
        with open(os.path.join(tar_fcg_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
    for fcg_p in os.listdir(cdd_fcg_path):
        with open(os.path.join(cdd_fcg_path, fcg_p), "rb") as f:
            fcg = pickle.load(f)
        fcgs_num[fcg_p.split("_fcg.pkl")[0]] = len(list(fcg.nodes()))
        
    if os.path.exists(feature_save_path) == False:
        os.makedirs(feature_save_path)
    if os.path.exists(sim_funcs_path) == False:
        os.makedirs(sim_funcs_path)
    if os.path.exists(time_path) == False:
        os.makedirs(time_path)
    
    with open(obj_func_embeddings_path, "r") as f:
        obj_func_embeddings = json.load(f)
    with open(cdd_func_embeddings_path, "r") as f:
        cdd_func_embeddings = json.load(f)
    for func, embed in obj_func_embeddings.items():
        if func not in cdd_func_embeddings:
            cdd_func_embeddings[func] = embed
    
    # object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
    # cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
    obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
    cdd_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "candidate_in9_embedding.json"))).keys()
    
    obj_com_funcs = {}
        # obj_com_funcs_file = json.load(open(os.path.join(com_funcs_path, "target_in9_embedding.json"))).keys()
    for obj_item in obj_com_funcs_file:
        # if obj_item.split("|||")[0] == object_name:
        if obj_item.split("|||")[0] not in obj_com_funcs:
            obj_com_funcs[obj_item.split("|||")[0]] = []
        if obj_item.split("|||")[1] not in obj_com_funcs[obj_item.split("|||")[0]]:
            obj_com_funcs[obj_item.split("|||")[0]].append(obj_item.split("|||")[1])
        
    cdd_com_funcs = {}
    for cdd_item in cdd_com_funcs_file:
        # if cdd_item.split("|||")[0] == candidate_name:
        # cdd_com_funcs[cdd_item.split("|||")[0]] = cdd_item.split("|||")[1]
        if cdd_item.split("|||")[0] not in cdd_com_funcs:
            cdd_com_funcs[cdd_item.split("|||")[0]] = []
        if cdd_item.split("|||")[1] not in cdd_com_funcs[cdd_item.split("|||")[0]]:
            cdd_com_funcs[cdd_item.split("|||")[0]].append(cdd_item.split("|||")[1])
    #     print("warning")
    # if object_item.split("__")[2] == candidate_name.split("__")[1]:
    #     continue
    
    # for obj_item in obj_com_funcs_file:
    #     for cdd_item in cdd_com_funcs_file:
    #         if obj_item.split("__")[2] == cdd_item.split("__")[1]:
    #             continue
    #     matched_func_list = cdd_project_dict[candidate_name]
        
        # json.dump(matched_func_list, open(os.path.join(sim_funcs_path, object_name+"___"+candidate_name+".json"), "w"))
        
    
    p_list = []
    Process_num = 15
    for i in range(Process_num):
        # tpl_detection_fast_one(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs, cdd_com_funcs)
        p = Process(target=tpl_detection_fast_one, args=(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path, cdd_func_embeddings, gnn, fcgs_num, obj_com_funcs, cdd_com_funcs, area_save_path))
        p_list.append(p)
            #args_list.append([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
            # compare_one_cdd_bin([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()



# anchor alignment v2.0(ICSME)
def anchor_alignment_ransac_multi(tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path):
    cal_time = {}
    # object_item_list_temp = []
    object_item_list = os.listdir(func_path)
    
    # for object_item_list_item in object_item_list:
    #     if "bzip2" in object_item_list_item:
    #         object_item_list_temp.append(object_item_list_item)
    
    if os.path.exists(feature_save_path) == False:
        os.makedirs(feature_save_path)
    if os.path.exists(sim_funcs_path) == False:
        os.makedirs(sim_funcs_path)
    if os.path.exists(time_path) == False:
        os.makedirs(time_path)
    
    p_list = []
    Process_num = 50
    for i in range(Process_num):
        # anchor_alignment_one_ransac(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path)
        p = Process(target=anchor_alignment_one_ransac, args=(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], tar_fcg_path, cdd_fcg_path, func_path, feature_save_path, time_path, com_funcs_path, sim_funcs_path))
        p_list.append(p)
            #args_list.append([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
            # compare_one_cdd_bin([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()





def get_achor_align_graph(fcg_path, func_path, taint_fcg_path):
    for object_item in os.listdir(func_path):
        object_name = object_item.split("_reuse_func_dict.json")[0]
        # if object_name == "precomp":
        object_fcg = get_fcg(fcg_path, object_name)
        object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
        
        for candidate_name in cdd_project_dict:
            
            if object_item.split("_")[0] == candidate_name.split("_")[0]:
                continue
            matched_func_list = cdd_project_dict[candidate_name]
            candidate_fcg = get_fcg(fcg_path, candidate_name)
            print("********* compare "+object_name+" and "+candidate_name+"**********")

            # (object_graph, candidate_graph, taint_flag, obj_tainted_func_set,cdd_tainted_func_set) = anchor_align_v1(object_fcg, matched_func_list, candidate_fcg)
            # if taint_flag == True:
            #     object_pydot_graph = nx.nx_pydot.to_pydot(object_graph)
            #     cdd_pydot_graph = nx.nx_pydot.to_pydot(candidate_graph)
            #     tainted_object_pydot_graph = get_taint_graph(object_pydot_graph, obj_tainted_func_set)
            #     tainted_cdd_pydot_graph = get_taint_graph(cdd_pydot_graph, cdd_tainted_func_set)
            #     tainted_object_pydot_graph.write_png(os.path.join(taint_fcg_path, object_name+"-_"+candidate_name+'_tainted_fcg.png'))
            #     tainted_cdd_pydot_graph.write_png(os.path.join(taint_fcg_path, object_name+"-_"+candidate_name+'_reverse_tainted_fcg.png'))
            
            tainted_graph_list = anchor_align_v3(object_fcg, matched_func_list, candidate_fcg)
            
            if tainted_graph_list != []:
                reuse_num = 0
                print("-- save tainted graph......")
                for graph_pair in tqdm(tainted_graph_list):
                    if graph_pair[2] == 0:
                        graph_pair[0].write_png(os.path.join(taint_fcg_path, object_name+"-_"+candidate_name+'_tainted_fcg_'+str(reuse_num)+'_all.png'))
                        graph_pair[1].write_png(os.path.join(taint_fcg_path, object_name+"-_"+candidate_name+'_reverse_tainted_fcg_'+str(reuse_num)+'_all.png'))
                        continue
                    reuse_num += 1
                    graph_pair[0].write_png(os.path.join(taint_fcg_path, object_name+"-_"+candidate_name+'_tainted_fcg_'+str(reuse_num)+'_'+str(graph_pair[2])+'.png'))
                    graph_pair[1].write_png(os.path.join(taint_fcg_path, object_name+"-_"+candidate_name+'_reverse_tainted_fcg_'+str(reuse_num)+'_'+str(graph_pair[2])+'.png'))
                print("analysis done!")
        

def get_libdb_version(fcg_path, func_path, result_path):

    obj_cdd_dict = {}
    reuse_detect_time = {}
    for object_item in tqdm(os.listdir(func_path)):
        start = time.time()
        object_name = object_item.split("_reuse_func_dict.json")[0]
        object_fcg = get_fcg(fcg_path, object_name)
        object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
        
        candidate_score = 0
        candidate_best = "None"
        for candidate_name in cdd_project_dict:
            if object_item.split("_")[0] == candidate_name.split("_")[0]:
                continue
            matched_func_list = cdd_project_dict[candidate_name]
            candidate_fcg = get_fcg(fcg_path, candidate_name)
            print("********* compare "+object_name+" and "+candidate_name+"**********")
 
            # tainted_graph_list = anchor_align_v3(object_fcg, matched_func_list, candidate_fcg)
            (common, afcg_rate) = libdb_fcg_filter(object_fcg, matched_func_list, candidate_fcg)
            
            if common > candidate_score:
                candidate_score = common
                candidate_best = candidate_name

        obj_cdd_dict[object_name] = candidate_best
        end = time.time()
        running_time = end-start
        reuse_detect_time[object_name] = running_time
        
        json.dump(obj_cdd_dict, open(result_path,"w"))
        json.dump(reuse_detect_time, open(result_path, "w"))
    
def get_libdb_reuse(fcg_path, func_path, result_path):
    obj_cdd_dict = {}
    reuse_detect_time = {}
    for object_item in tqdm(os.listdir(func_path)):
        start = time.time()
        object_name = object_item.split("_reuse_func_dict.json")[0]
        object_fcg = get_isrd_fcg(fcg_path, object_name)
        object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
        obj_cdd_dict[object_name] = []
        print("********* compare "+object_name+"**********")
 
        for candidate_name in cdd_project_dict:
            if object_item.split("_")[0] == candidate_name.split("_")[0]:
                continue
            matched_func_list = cdd_project_dict[candidate_name]
            candidate_fcg = get_isrd_fcg(fcg_path, candidate_name)
            # print("********* compare "+object_name+" and "+candidate_name+"**********")
 
            # tainted_graph_list = anchor_align_v3(object_fcg, matched_func_list, candidate_fcg)
            (common, afcg_rate) = libdb_fcg_filter(object_fcg, matched_func_list, candidate_fcg)
            
            #现在的结果还没加0.1，可以加上>0.1试一下能不能提高点
            if common > 3 and afcg_rate > 0.1:
                obj_cdd_dict[object_name].append(candidate_name)
        end = time.time()
        running_time = end-start
        reuse_detect_time[object_name] = running_time
    json.dump(obj_cdd_dict, open(result_path+"libdb_reuse_result.json","w"))
    json.dump(reuse_detect_time, open(result_path+"libdb_reuse_time.json","w"))
    


def get_libdb_reuse_one(func_path_list, fcg_path, func_path, result_path, i):
    obj_cdd_dict = {}
    obj_cdd_func_dict = {}
    reuse_detect_time = {}
    for object_item in tqdm(func_path_list):
        start = time.time()
        object_name = object_item.split("_reuse_func_dict")[0]
        object_fcg = get_isrd_fcg(fcg_path, object_name)
        object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        cdd_project_dict = get_cdd_func_dict(object_cdd_func_dict)
        obj_cdd_dict[object_name] = []
        obj_cdd_func_dict[object_name] = {}
        # print("********* compare "+object_name+"**********")
 
        for candidate_name in tqdm(cdd_project_dict):
            if object_item.split("_")[0] == candidate_name.split("_")[0]:
                continue
            matched_func_list = cdd_project_dict[candidate_name]
            candidate_fcg = get_isrd_fcg(fcg_path, candidate_name)
            # print("********* compare "+object_name+" and "+candidate_name+"**********")
 
            # tainted_graph_list = anchor_align_v3(object_fcg, matched_func_list, candidate_fcg)
            (common, afcg_rate, matched_func) = libdb_fcg_filter(object_fcg, matched_func_list, candidate_fcg)
            
            #现在的结果还没加0.1，可以加上>0.1试一下能不能提高点
            if common > 3 and afcg_rate > 0.1:
                obj_cdd_dict[object_name].append(candidate_name)
                obj_cdd_func_dict[object_name][candidate_name] = matched_func
        end = time.time()
        running_time = end-start
        reuse_detect_time[object_name] = running_time
    json.dump(obj_cdd_dict, open(result_path+"libdb_reuse_result"+i+".json","w"))
    json.dump(reuse_detect_time, open(result_path+"libdb_reuse_time"+i+".json","w"))
    json.dump(obj_cdd_func_dict, open(result_path+"libdb_func_result"+i+".json","w"))




def get_libdb_reuse_multi(fcg_path, func_path, result_path):
    cal_time = {}
    object_item_list = os.listdir(func_path)
    
    p_list = []
    Process_num = 35
    for i in range(Process_num):
        p = Process(target=get_libdb_reuse_one, args=(object_item_list[int((i/Process_num)*len(object_item_list)):int(((i+1)/Process_num)*len(object_item_list))], fcg_path, func_path, result_path, str(i)))
        p_list.append(p)
            #args_list.append([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
            # compare_one_cdd_bin([candidate_software, object_funcs, object_software, candidate_funcs, object_matrix, sims_list_opath])
    for p in p_list:
        p.start()
    for p in p_list:
        p.join()


  
def get_gemini_result(fcg_path, func_path, result_path):
    reuse_result = {}
    reuse_func_result = {}
    reuse_detect_time = {}
    for object_item in tqdm(os.listdir(func_path)):
        start = time.time()
        object_name = object_item.split("_reuse_func_dict.json")[0]
        reuse_result[object_name] = []
        reuse_func_result[object_name] = {}
        # object_fcg = get_fcg(fcg_path, object_name)
        object_cdd_func_dict = json.load(open(os.path.join(func_path, object_item), "r"))
        cdd_project_dict = get_gemini_dict(object_cdd_func_dict)
        for candidate_name in cdd_project_dict:
            if object_item.split("_")[0] == candidate_name.split("_")[0]:
                continue
            matched_func_list = cdd_project_dict[candidate_name]
            candidate_fcg = get_fcg(fcg_path, candidate_name)
            
            if isrd_method(matched_func_list, candidate_fcg):
                reuse_result[object_name].append(candidate_name)
                reuse_func_result[object_name][candidate_name] = matched_func_list
        end = time.time()
        running_time = end-start
        reuse_detect_time[object_name] = running_time
        json.dump(reuse_result, open(result_path+"gemini_isrd_reuse_result.json","w"))
        json.dump(reuse_detect_time, open(result_path+"gemini_isrd_reuse_time.json","w"))
        json.dump(reuse_func_result, open(result_path+"gemini_isrd_reuse_func.json","w"))
    
if __name__ == "__main__":
    # fcg_path = "/data/lisiyuan/libAE/data-new/fcg"#"isrd_default_fcg"
    # func_path = "/data/lisiyuan/code/new_code_20221015/DFcon_20221027/score/paper_dataset/new_isrd_triple_loss_score_top50/"#"reuse_result_deal_pruning"
    # # taint_fcg_path = "F:\\mypaper\\data\\cross_isrd_tainted_fcg"#"tainted_fcg"
    # result_path = "result/1124_gemini_top50/"
    # feature_save_path = "result/1124_1_libae_libae_top50/"
    # time_path = "time_result/1124_1_libdb_paper_time.json"

    # #get_achor_align_graph(fcg_path, func_path, taint_fcg_path)
    # get_reuse_area_multi(fcg_path, func_path, feature_save_path, time_path)
    get_reuse_area_multi(os.path.join(DATA_PATH, "3_fcg/target"),
                        os.path.join(DATA_PATH, "3_fcg/candidate"), 
                        os.path.join(DATA_PATH, "5_func_compare_result/score_top50"), 
                        os.path.join(DATA_PATH, "6_alignment_result/alignment_result"), 
                        os.path.join(DATA_PATH, "6_alignment_result/"))
    # get_libdb_reuse_multi(fcg_path, func_path, result_path)
    
    # fcg_path = "F:\\mypaper\\data\\openssl_result\\openssl_fcg"#"isrd_default_fcg"
    # func_path = "F:\\mypaper\\data\\openssl_result\\reuse_result_openssl"#"reuse_result_deal_pruning"
    # result_path = "F:\\mypaper\\data\\version_result.json"#"tainted_fcg"
    
    # get_libdb_version(fcg_path, func_path, result_path)
    
    # get_gemini_result(fcg_path, func_path, result_path)
            
            
                
    print("all done")
            