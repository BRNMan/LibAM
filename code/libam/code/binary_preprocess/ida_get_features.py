import os.path
import os
from getAcfg import *
from idc import *
from idautils import *
from idaapi import *
import json
import idc
import ida_auto
import ida_pro
from time import gmtime, strftime

def extract_features():
    times = {}
    time_begin = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    analysis_flags = idc.get_inf_attr(idc.INF_AF)
    analysis_flags &= ~idc.AF_IMMOFF
    idc.set_inf_attr(idc.INF_AF, analysis_flags)

    savePath = idc.ARGV[1]
    print("itss:", savePath)


    ida_auto.auto_wait()
    cfgs = get_func_cfgs_c(idc.get_first_seg())
    binaryName = idc.get_input_file_path()
    for nodes in cfgs.func_acfg_list:
        dict = {}
        dict["src"] = binaryName
        # dict["succs"] = list(nodes.g.edges())
        succs = []
        for node in range(len(nodes.g)):
            succs.append([])
        for edge in nodes.g.edges():
            succs[edge[0]].append(edge[1])
        dict["succs"] = succs
        dict["n_num"] = len(nodes.g)
        features = []
        for node in nodes.g.nodes():
            features.append(nodes.g.nodes[node]['vec'])
            pass
        dict["features"] = list(features)
        dict["fname"] = nodes.funcName
        dict["calls"] = nodes.calls
        saveJsonDocument(dict, savePath)
    time_end = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    times[binaryName] = time_begin + "||" + time_end
    ida_pro.qexit(0)
    return cfgs

def saveJsonDocument(dicts, fileName):
    with open(fileName, 'a') as out:
        json.dump(dicts, out, ensure_ascii=False)
        out.write("\n")
    out.close()
    pass

if __name__ == '__main__':
    print("check point 1")
    extract_features()
    print("check point 2")