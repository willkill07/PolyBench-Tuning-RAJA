#!/usr/bin/env python3
import re
from collections import OrderedDict
from collections import defaultdict

def generateWPCFG(asmFile):
    
    code=OrderedDict()
    calls=OrderedDict()
    links=defaultdict(set)
    
    functions=set()
    breakpoints=set()
    returns=set()
    
    with open(asmFile) as file:
        # previous instruction is used for conditional branching to add a link to if branch not taken
        prev = None
        
        newFunctionBlock = True
        for line in file:
            line = line.rstrip('\n')
            split = re.split("[\t :,]+", line)
            # check if we are encountering a new block (function)
            if newFunctionBlock:
                newFunctionBlock = False
                functions.add(split[0])
            elif len(line) == 0:
                # encounter empty line -- expect a new function block
                newFunctionBlock = True
                continue
                
            # rename accesses to make sense
            address, instr, args = split[0], split[1], split[2:]
            
            # insert into code lookup
            code[address] = (instr, args)
            # if we have a previous instruction set from a conditional jump, add the link from prev->address
            if prev:
                links[prev].add(address)
            prev = None
            if instr.startswith("ret"): # handle returns
                returns.add(address)
            elif instr.startswith('jmp'): # and unconditional jumps
                links[address].add(args[0])
            elif instr[0] == 'j': # and conditional jumps
                links[address].add(args[0])
                prev = address
            elif instr.startswith('call'): # be intelligent with calls
                calls[address] = args
    
    for address in calls:
        if calls[address][0] in functions:
            calls[address] = calls[address][0]
            continue
        # Resolve GOMP_parallel call to function
        if "GOMP_parallel" in calls[address][1]:
            if not address in list(code.keys()):
                calls[address] = None
                continue
            index = list(code.keys()).index(address) - 1
            reversedInstr = reversed(list(code.items())[0:index])
            found = False
            for backAddress,args in reversedInstr:
                # if we reach another call give up on resolving
                if 'call' in args[0]:
                    calls[address] = None
                    break
                if 'mov' in args[0]:
                    # trim leading $, 0x, or $0x if exists
                    reference = re.sub('^\$?(0x)?', '', args[1][0])
                    # if we encounter the address, resolve the function reference
                    if reference in functions:
                        calls[address] = reference
                        found = True
                        break
            if found:
                continue
        calls[address] = None
    
    # establish a list of all possible divergent locations
    targetBreaks = {address for targets in links.values() for address in targets}
    jumpBreaks = {addr[0] for addr in links}

    # add all calls to resolved links
    for call in calls:
        if calls[call]:
            links[call].add(calls[call])
    
    # construct basic blocks
    currBlock=[]
    basicBlocks={}
    prevLast = None
    # iterate over each instruction in our code
    for c in code:
        # functions get their own basic block start
        if c in functions:
            if len(currBlock) > 0:
                basicBlocks[currBlock[0]] = currBlock
            currBlock = []
        currBlock.append(c)
        # so do any observed breakpoints (jump targets)
        if c in jumpBreaks or c in returns:
            basicBlocks[currBlock[0]] = currBlock
            currBlock = []
        # and so do targets of jumps!
        if c in targetBreaks:
            inst = code[prevLast][0]
            if not 'call' in inst and not 'ret' in inst and 'j' != inst[0]:
                links[prevLast].add(c)
            basicBlocks[currBlock[0]] = currBlock
            currBlock = []
        prevLast = c
    if len(currBlock) > 0:
        basicBlocks[currBlock[0]] = currBlock
        
        
    return {"links":links,"bb":basicBlocks,"code":code}
