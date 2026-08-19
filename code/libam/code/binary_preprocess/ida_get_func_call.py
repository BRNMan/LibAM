import sys, os

from settings import *
sys.path.insert(0, PACKAGE_PATH)
sys.path.insert(0, PACKAGE_PATH2)
# from settings import *
from idaapi import *
from idc import *
from idautils import *
import idc
import ida_auto
import ida_pro
import ida_funcs

# import matplotlib.pyplot as plt
print("hello fcg")
import networkx as nx
import pickle


def get_func_len(func_ea):
    func_context_len = 0
    for head in FuncItems(func_ea):
        func_context_len += 1
    return func_context_len


def is_func_in_plt(function_ea):
    return False
    segm_name = idc.get_segm_name(function_ea).lower()
    if ".got" not in segm_name and ".plt" not in segm_name:
        return False
    else:
        return True


def auto_analysis():
    inputName = idc.get_idb_path()
    callees = dict()
    func_addr_dict = dict()

    for function_ea in Functions():  # SegStart(ea), SegEnd(ea)):
        if False == is_func_in_plt(function_ea):
            f_name = idc.get_name(function_ea)
            func_addr_dict[f_name] = function_ea
            find_func = False

            for ref_ea in CodeRefsTo(function_ea, 0):
                # Find function associated with the call
                ref_func = ida_funcs.get_func(ref_ea)
                if not ref_func:
                    # Functions in the PLT will be like this
                    continue
                ref_start_ea = ref_func.start_ea
                if False == is_func_in_plt(ref_start_ea):
                    find_func = True

                    caller_name = idc.get_name(ref_start_ea)
                    callees[caller_name] = callees.get(caller_name, set())

                    callees[caller_name].add(f_name)


    g = nx.DiGraph()


    functions = set(callees.keys())

    for f in functions:
        if f in callees and f != "":
            g.add_node(f, func_addr=func_addr_dict[f])
            for f2 in callees[f]:
                g.add_node(f2, func_addr=func_addr_dict[f2])
                g.add_edge(f, f2)


    savePath = idc.ARGV[1]

    with open(savePath, 'wb') as f:
        pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)

ida_auto.auto_wait()
auto_analysis()
ida_pro.qexit(0)
