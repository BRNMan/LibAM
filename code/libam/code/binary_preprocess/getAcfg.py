# -*- encoding: utf-8 -*-
'''
@File    :   getAcfg.py
@Time    :   2022/11/25 13:10:41
@Author  :   WangYongpan 
'''
from idautils import *
from idc import *
import ida_segment
import sys
from settings import *
sys.path.insert(0, PACKAGE_PATH)
sys.path.insert(0, PACKAGE_PATH2)
import buildAcfg as cfg
from acfg import *
from discovRe import *

# 获取程序的所有函数的acfg
def get_func_cfgs_c(ea):
    binary_path = idc.get_input_file_path()
    binary_name = binary_path.split("/")[-1]
    project_name = binary_path.split("/")[-2]
    raw_cfgs = funcs_acfg(binary_name)
    externs_eas, ea_externs = processpltSegs()
    i = 0
    for funcea in Functions(idc.get_segm_start(ea)):
        funcname = get_unified_funcname(funcea)
        func = get_func(funcea)
        # 获取该函数调用的函数名
        calls = []
        startEa = func.start_ea
        while startEa < func.end_ea:
            for call in CodeRefsFrom(startEa, 1):
                call_name = get_unified_funcname(call)
                if call_name not in calls and call_name != funcname and call_name != "":
                    call_name = project_name + '|||' + binary_name.split('.')[0] + '|||' + call_name
                    calls.append(call_name)
                    pass
            startEa = idc.next_head(startEa)
        i += 1
        icfg = cfg.buildCfg(func, externs_eas, ea_externs)
        func_f = get_discoverRe_feature(func, icfg[0])
        # funcname = project_name + '|||' + binary_name.split('.')[0] + '|||' + funcname
        raw_g = func_acfg(funcname, icfg, func_f, calls)
        raw_cfgs.append(raw_g)
    return raw_cfgs
    pass

def processpltSegs():
    funcdata = {}
    datafunc = {}
    for n in range(ida_segment.get_segm_qty()):
        seg = ida_segment.getnseg(n)
        ea = seg.start_ea
        segname = idc.get_segm_name(ea)
        if segname in ['.plt', 'extern', '.MIPS.stubs']:
            start = seg.start_ea
            end = seg.end_ea
            cur = start
            while cur < end:
                name = get_unified_funcname(cur)
                funcdata[name] = hex(cur)
                datafunc[cur] = name
                cur = idc.next_head(cur)
    return funcdata, datafunc

# 获取指定地址的函数名
def get_unified_funcname(ea):
    funcname = idc.get_func_name(ea)
    if len(funcname) > 0:
        if '.' == funcname[0]:
            funcname = funcname[1:]
    return funcname
