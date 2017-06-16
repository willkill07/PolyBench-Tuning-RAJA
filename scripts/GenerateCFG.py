#!/usr/bin/env python3
import re
import sys
import pprint

from collections import OrderedDict
from collections import defaultdict

def generateWPCFG(asmFile):

    code=OrderedDict()
    calls=OrderedDict()
    links=defaultdict(set)

    functions=set()
    returns=set()


    with open(asmFile) as asm_file:
        new_function_block = True
        prev_address_cjump = None
        for line in asm_file:

            line = line.rstrip('\n')
            split_line = re.split("[\t :,]+", line)

            if new_function_block:
                new_function_block = False
                functions.add(split_line[0])
            elif len(line) == 0:
                new_function_block = True
                continue
            address, instr, args = split_line[0], split_line[1], split_line[2:]

            code[address] = (instr, args)

            # if prior instruction was conditional jump, make the Branch-Not-Taken link
            if prev_address_cjump:
                links[prev_address_cjump].add(address)
            prev_address_cjump = None

            # handle ret
            if instr.startswith("ret"):
                returns.add(address)
            # and the elusive rep ret
            elif instr == "rep" and args[0] == "ret":
                returns.add(address)
            # unconditional jumps
            elif instr.startswith('jmp'):
                links[address].add(args[0])
            # and conditional jumps, too
            elif instr.startswith('j'):
                links[address].add(args[0])
                prev_address_cjump = address
            # finally, calls
            elif instr.startswith('call'):
                calls[address] = args

    for address in calls:
        if calls[address][0] in functions:
            calls[address] = calls[address][0]
            continue
        print(calls[address])
        if len(calls[address]) < 2:
            continue
        # Resolve GOMP_parallel call to function
        if "GOMP_parallel" in calls[address][1] or "__kmpc_fork_call" in calls[address][1]:
            if not address in list(code.keys()):
                calls[address] = None
                continue
            index = list(code.keys()).index(address) - 1
            reversed_instr = reversed(list(code.items())[0:index])
            found = False
            for _,args in reversed_instr:
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
    target_breaks = {address for targets in links.values() for address in targets}
    jump_breaks = {addr[0] for addr in links}

    # add all calls to resolved links
    for call in calls:
        if calls[call] and type(calls[call]) is not list:
            links[call].add(calls[call])

    # construct basic blocks
    curr_block=[]
    basic_blocks={}
    prev_last = None
    # iterate over each instruction in our code
    for curr in code:
        # functions get their own basic block start
        if curr in functions:
            if len(curr_block) > 0:
                basic_blocks[curr_block[0]] = curr_block
            curr_block = []
        curr_block.append(curr)
        # so do any observed breakpoints (jump targets)
        if curr in jump_breaks or curr in returns:
            basic_blocks[curr_block[0]] = curr_block
            curr_block = []
        # and so do targets of jumps!
        elif curr in target_breaks:
            # this should not be necessary
            #    inst = code[prev_last][0]
            #    if not 'call' in inst and not 'ret' in inst and 'j' != inst[0]:
            links[prev_last].add(curr)
            basic_blocks[curr_block[0]] = curr_block
            curr_block = []
        prev_last = curr
    if len(curr_block) > 0:
        basic_blocks[curr_block[0]] = curr_block

    return {"links" : links, "bb" : basic_blocks, "code" : code}

if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(generateWPCFG(sys.argv[1]))
