# -*- encoding: utf-8 -*-
'''
@File    :   extract_statistical_features.py
@Time    :   2022/11/25 13:10:47
@Author  :   WangYongpan 
'''
from idautils import *
from idaapi import *
import idaapi
import ida_idaapi
from idc import *

# 获取基本块内每条指令的字符串常量和数值常量
import ida_ua
import ida_bytes
import ida_segment
import ida_nalt

def getConst(ea, offset):
    """
    COPILOT HELPED HERE
    Extract strings and constants from an address in a binary.
    Updated for IDA Pro 9.3+ from the original IDA 6.8 version.
    
    :param ea: Linear address of the instruction
    :param offset: Operand offset (0 for first operand, 1 for second, etc.)
    :return: Tuple of (strings_list, consts_list)
    """
    strings = []
    consts = []
    
    # Get operand type - use modern API
    opType1 = get_operand_type(ea, offset)
    
    if opType1 == ida_ua.o_imm:  # Immediate value operand
        imm_value = get_operand_value(ea, offset)
        
        # Filter out small constants (0-10)
        if 0 <= imm_value <= 10:
            consts.append(imm_value)
        else:
            # Check if the value points to a loaded memory address
            if ida_bytes.is_loaded(imm_value):
                seg = ida_segment.getseg(imm_value)
                if seg:
                    # Try to read string from the address
                    str_value = get_string_from_address(imm_value)
                    
                    # If not found, try offset by 0x40000 (common relocation offset)
                    if str_value is None:
                        str_value = get_string_from_address(imm_value + 0x40000)
                        if str_value is None:
                            consts.append(imm_value)
                        else:
                            # Validate string contains printable ASCII
                            if is_valid_string(str_value):
                                strings.append(str_value)
                            else:
                                consts.append(imm_value)
                    else:
                        if is_valid_string(str_value):
                            strings.append(str_value)
                        else:
                            consts.append(imm_value)
                else:
                    consts.append(imm_value)
            else:
                consts.append(imm_value)
    
    return strings, consts


def get_operand_type(ea, n):
    """
    Get type of instruction operand using modern IDA API.
    
    :param ea: Linear address of instruction
    :param n: Operand number (0 or 1)
    :return: Operand type constant
    """
    insn = ida_ua.insn_t()
    inslen = ida_ua.decode_insn(insn, ea)
    
    if inslen == 0:
        return -1
    
    return insn.ops[n].type


def get_operand_value(ea, n):
    """
    Get value of instruction operand using modern IDA API.
    
    :param ea: Linear address of instruction
    :param n: Operand number (0 or 1)
    :return: Operand value
    """
    insn = ida_ua.insn_t()
    inslen = ida_ua.decode_insn(insn, ea)
    
    if inslen == 0:
        return -1
    
    op = insn.ops[n]
    
    if op.type == ida_ua.o_imm:
        return op.value
    elif op.type == ida_ua.o_mem:
        return op.addr
    elif op.type == ida_ua.o_far or op.type == ida_ua.o_near:
        return op.addr
    elif op.type == ida_ua.o_displ:
        return op.addr
    else:
        return -1


def get_string_from_address(ea, max_length=256):
    """
    Extract string from address using modern IDA API.
    
    :param ea: Linear address where string is located
    :param max_length: Maximum string length to read
    :return: String if found, None otherwise
    """
    try:
        # Check if address is loaded
        if not ida_bytes.is_loaded(ea):
            return None
        
        # Try to detect string type at this address
        flags = ida_bytes.get_flags(ea)
        
        # If it's already marked as a string, get its type
        if ida_bytes.is_strlit(flags):
            strtype = get_str_type(ea)
            if strtype is None:
                strtype = ida_nalt.STRTYPE_C  # Default to C-string
        else:
            strtype = ida_nalt.STRTYPE_C
        
        # Get string contents
        str_contents = ida_bytes.get_strlit_contents(
            ea, 
            max_length, 
            strtype
        )
        
        if str_contents:
            # Convert bytes to string if necessary
            if isinstance(str_contents, bytes):
                return str_contents.decode('utf-8', errors='ignore')
            return str_contents
        
        return None
        
    except Exception as e:
        print(f"Error reading string from {hex(ea)}: {e}")
        return None


def get_str_type(ea):
    """
    Get string type at address.
    
    :param ea: Linear address
    :return: String type constant or None
    """
    flags = ida_bytes.get_flags(ea)
    
    if ida_bytes.is_strlit(flags):
        oi = ida_nalt.opinfo_t()
        if ida_bytes.get_opinfo(oi, ea, 0, flags):
            return oi.strtype
    
    return None


def is_valid_string(s):
    """
    Validate if string contains printable ASCII characters.
    
    :param s: String to validate
    :return: True if valid, False otherwise
    """
    if not s:
        return False
    
    # Check if all characters are in printable ASCII range (40-127)
    # This matches your original validation: 40 <= ord(c) < 128
    try:
        if isinstance(s, bytes):
            return all(40 <= b < 128 for b in s)
        else:
            return all(40 <= ord(c) < 128 for c in s)
    except (TypeError, ValueError):
        return False

# 获取给定基本块的所有字符串常量和数值常量
def getBBconsts(bl):
    strings = []
    consts = []
    start = bl[0]
    end = bl[1]
    inst_addr = start
    while inst_addr < end:
        opcode = idc.print_insn_mnem(inst_addr)
        if opcode in ['la', 'jalr', 'call', 'jal']:
            inst_addr = idc.next_head(inst_addr)
            continue
        strings_src, consts_src = getConst(inst_addr, 0)
        strings_dst, consts_dst = getConst(inst_addr, 1)
        strings += strings_src
        strings += strings_dst
        consts += consts_src
        consts += consts_dst
        try:
            strings_dst, consts_dst = getConst(inst_addr, 2)
            consts += consts_dst
            strings += strings_dst
        except:
            pass
        inst_addr = idc.next_head(inst_addr)
    return strings, consts
    pass

# 获取每个汇编函数中的字符串常量和数值常量
def getfunc_consts(func):
    strings = []
    consts = []
    blocks = [(v.start_ea, v.end_ea) for v in FlowChart(func)]
    for bl in blocks:
        strs, conts = getBBconsts(bl)
        strings += strs
        consts += conts
    return strings, consts

# 计算基本块内转移指令的数量
def calTransferIns(bl):
    # x86_TI = {'jmp': 1, 'jz': 1, 'jnz': 1, 'js': 1, 'je': 1, 'jne': 1, 'jg': 1, 'jle': 1, 'jge': 1, 'ja': 1, 'jnc': 1, 'call': 1}
    # mips_TI = {'beq': 1, 'bne': 1, 'bgtz': 1, "bltz": 1, "bgez": 1, "blez": 1, 'j': 1, 'jal': 1, 'jr': 1, 'jalr': 1}
    # arm_TI = {'MVN': 1, "MOV": 1}
    mips_TI = {"beqz": 1, "beq": 1, "bne": 1, "bgez": 1, "b": 1, "bnez": 1, "bgtz": 1, "bltz": 1, "blez": 1, "bgt": 1,  "bge": 1, "blt": 1, "ble": 1, "bgtu": 1, "bgeu": 1, "bltu": 1, "bleu": 1}
    x86_TI = {"jz": 1, "jnb": 1, "jne": 1, "je": 1, "jg": 1, "jle": 1, "jl": 1, "jge": 1, "ja": 1, "jae": 1, "jb": 1,  "jbe": 1, "jo": 1, "jno": 1, "js": 1, "jns": 1, "jr": 1}
    arm_TI = {"B": 1, "BL": 1, "BAL": 1, "BNE": 1, "BEQ": 1, "BPL": 1, "BMI": 1, "BCC": 1, "BLO": 1, "BCS": 1, "BHS": 1, "BVC": 1, "BVS": 1, "BGT": 1, "BGE": 1, "BLT": 1, "BLE": 1, "BHI": 1, "BLS": 1}
    calls = {}
    calls.update(x86_TI)
    calls.update(mips_TI)
    calls.update(arm_TI)
    start = bl[0]
    end = bl[1]
    invoke_num = 0
    inst_addr = start
    while inst_addr < end:
        opcode = idc.print_insn_mnem(inst_addr)
        re = [v for v in calls if opcode.lower() in v.lower()]
        if len(re) > 0:
            invoke_num += 1
        inst_addr = idc.next_head(inst_addr)
    return invoke_num

# 计算函数内转移指令的数量
def getTransferInsts(func):
    blocks = [(v.start_ea, v.end_ea) for v in FlowChart(func)]
    sumcalls = 0
    for bl in blocks:
        callnum = calTransferIns(bl)
        sumcalls += callnum
    return sumcalls

# 计算基本块内调用的数量
def calCalls(bl):
    calls = {'call': 1, 'jal': 1, 'jalr': 1}
    start = bl[0]
    end = bl[1]
    invoke_num = 0
    inst_addr = start
    while inst_addr < end:
        opcode = idc.print_insn_mnem(inst_addr)
        if opcode.lower() in calls:
            invoke_num += 1
        inst_addr = idc.next_head(inst_addr)
    return invoke_num
    pass

# 计算函数内调用的数量
def getFuncCalls(func):
    blocks = [(v.start_ea, v.end_ea) for v in FlowChart(func)]
    sumcalls = 0
    for bl in blocks:
        callnum = calCalls(bl)
        sumcalls += callnum
    return sumcalls

# 计算基本块内指令的数量
def calInstrs(bl):
    start = bl[0]
    end = bl[1]
    inst_addr = start
    invoke_num = 0
    while inst_addr < end:
        invoke_num += 1
        inst_addr = idc.next_head(inst_addr)
    return invoke_num

# 计算函数的指令数量
def getFuncInstrs(func):
    blocks = [(v.start_ea, v.end_ea) for v in FlowChart(func)]
    sumInstr = 0
    for bl in blocks:
        instr = calInstrs(bl)
        sumInstr += instr
    return sumInstr

# 计算基本块内算数指令的数量
def calArithmeticInstr(bl):
    x86_AI = {'add': 1, 'sub': 1, 'div': 1, 'imul': 1, 'idiv': 1, 'mul': 1, 'shl': 1, 'dec': 1, 'inc': 1}
    mips_AI = {'add': 1, 'addu': 1, 'addi': 1, 'addiu': 1, 'mult': 1, 'multu': 1, 'div': 1, 'divu': 1}
    arm_AI = {'ADD': 1, 'ADC': 1, 'SUB': 1, 'RSB': 1, 'SBC': 1, 'RSC': 1, 'MUL': 1, 'MLA': 1, 'SMULL': 1, 'SMLAL': 1, 'UMULL': 1, 'UMLAL': 1}
    calls = {}
    calls.update(x86_AI)
    calls.update(mips_AI)
    calls.update(arm_AI)
    start = bl[0]
    end = bl[1]
    invoke_num = 0
    inst_addr = start
    while inst_addr < end:
        opcode = idc.print_insn_mnem(inst_addr)
        re = [v for v in calls if opcode.lower() in v.lower()]
        if len(re) > 0:
            invoke_num += 1
        inst_addr = idc.next_head(inst_addr)
    return invoke_num

# 计算基本块内逻辑指令的数量
def calLogicInstructions(bl):
    x86_LI = {'and': 1, 'andn': 1, 'andnpd': 1, 'andpd': 1, 'andps': 1, 'andnps': 1, 'test': 1, 'xor': 1, 'xorpd': 1, 'pslld': 1}
    mips_LI = {'and': 1, 'andi': 1, 'or': 1, 'ori': 1, 'xor': 1, 'nor': 1, 'slt': 1, 'slti': 1, 'sltu': 1}
    arm_LI = {'AND': 1, 'ORR': 1, 'EOR': 1, 'BIC': 1, 'TEQ': 1, 'TST': 1}
    calls = {}
    calls.update(x86_LI)
    calls.update(mips_LI)
    calls.update(arm_LI)
    start = bl[0]
    end = bl[1]
    invoke_num = 0
    inst_addr = start
    while inst_addr < end:
        opcode = idc.print_insn_mnem(inst_addr)
        re = [v for v in calls if opcode.lower() in v.lower()]
        if len(re) > 0:
            invoke_num += 1
        inst_addr = idc.next_head(inst_addr)
    return invoke_num

# 计算函数内的逻辑指令的数量
def getLogicInsts(func):
    blocks = [(v.start_ea, v.end_ea) for v in FlowChart(func)]
    sumcalls = 0
    for bl in blocks:
        callnum = calLogicInstructions(bl)
        sumcalls += callnum
    return sumcalls

# 获取该基本块调用的基本块地址
def retrieveExterns(bl, ea_externs):
    externs = []
    start = bl[0]
    end = bl[1]
    inst_addr = start
    while inst_addr < end:
        refs = CodeRefsFrom(inst_addr, 1)
        try:
            ea = [v for v in refs if v in ea_externs][0]
            externs.append(ea_externs[ea])
        except:
            pass
        inst_addr = idc.next_head(inst_addr)
    return externs


def getLocalVariables(func):
    args_num = get_stackVariables(func.start_ea)
    return args_num

# 获取存储的本地变量
import ida_funcs
import ida_frame
import ida_typeinf

def get_stackVariables(func_addr):
    """
    Get the number of stack variables in a function.
    Updated for IDA Pro 9.3+ from the original IDA 6.8 version.

    :param func_addr: Address of the function
    :return: Number of stack variables (local variables)
    """
    args = []

    # Get the function object
    func = ida_funcs.get_func(func_addr)
    if not func:
        return 0

    # Get the frame (stack frame) ID
    frame_id = func.frame
    if frame_id == ida_idaapi.BADADDR:
        return 0

    # Get the frame structure
    tif = ida_typeinf.tinfo_t()
    if not tif.get_type_by_tid(frame_id):
        return 0

    if not tif.is_udt():
        return 0

    # Get member count
    member_count = tif.get_udt_nmembers()
    if member_count <= 0:
        return 0

    # Iterate through members
    for idx in range(member_count):
        try:
            # Get member name
            mName = get_member_name_by_idx(frame_id, idx)

            # Filter: only include variables with 'var_' in the name
            if mName and 'var_' in mName and mName not in args:
                args.append(mName)

        except Exception as e:
            print(f"Error processing member {idx}: {e}")
            continue

    return len(args)


def get_member_name_by_idx(frame_id, idx):
    """
    Get member name by index from a structure/frame.

    :param frame_id: Structure/Frame type ID
    :param idx: Member index
    :return: Member name or None
    """
    tif = ida_typeinf.tinfo_t()
    if not tif.get_type_by_tid(frame_id):
        return None

    if not tif.is_udt():
        return None

    try:
        if idx < 0 or idx >= tif.get_udt_nmembers():
            return None

        # Get member TID by index
        tidx = tif.get_udm_tid(idx)
        if tidx == ida_idaapi.BADADDR:
            return None

        # Get the member details
        udm = ida_typeinf.udm_t()
        if tif.get_udm_by_tid(udm, tidx):
            return udm.name

        return None
    except Exception as e:
        print(f"Error getting member name at index {idx}: {e}")
        return None
# 获取基本块的数量
def getBasicBlocks(func):
    blocks = [(v.start_ea, v.end_ea) for v in FlowChart(func)]
    return len(blocks)

# 获取调用该函数的地址数量
def getIncommingCalls(func):
    refs = CodeRefsTo(func.start_ea, 0)
    re = len([v for v in refs])
    return re
