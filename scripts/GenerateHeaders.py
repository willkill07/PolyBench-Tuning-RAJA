#!/usr/bin/env python3
import sys
import os
import json
import itertools
import pprint

with open("data/autotuning.json") as fp:
    tuning = json.load(fp)['data']

def checkPolicyValidity(policyTuple, perm=None):
    if perm:
        lol = [i for i,v in enumerate(policyTuple) if v == 'RAJA::omp_collapse_nowait_exec']
        print(perm)
    # more than one parallel for? gonna have a bad time
    if policyTuple.count('RAJA::omp_parallel_for_exec') > 1:
        return False
    if policyTuple.count('RAJA::omp_for_nowait_exec') > 1:
        return False
    # only one collapse? gonna have a bad time
    if policyTuple.count('RAJA::omp_collapse_nowait_exec') == 1:
        return False
    # mixing collapse with parallel for? gonna have a bad time
    if policyTuple.count('RAJA::omp_for_nowait_exec') > 0 and policyTuple.count('RAJA::omp_collapse_nowait_exec') > 0:
        return False
    return True

def generatePolicies(loopSize):
    expansion = itertools.product(tuning[loopSize]['policies'], repeat=loopSize);
    policies = [x for x in expansion if checkPolicyValidity(x)]
    return policies

def generatePermutations(loopSize):
    return [x for x in itertools.permutations(range(loopSize))]

def generateTiles(loopSize):
    if loopSize == 1:
        return [None]
    return [x for x in itertools.product(tuning[loopSize]['tiles'], repeat=loopSize)]

def generateAll(loopSize, tag):
    return [(tag,x) for x in itertools.product(generatePolicies(loopSize),generateTiles(loopSize),generatePermutations(loopSize))]

def traverse(loopNest, node):
    while loopNest[node]['parent'] != None:
        yield node
        node = loopNest[node]['parent']
    yield node

def findConstraints(loopNest):
    # set difference between the loop nests and what's in parent
    data = set(range(len(loopNest)))-set(v['parent'] for i,v in enumerate(loopNest))
    return [[y for y in traverse(loopNest, x)] for x in data]

def expandVersionsGivenConstraints(versions, constraints):
    # blow up constraints with the actual policies
    for arr in constraints:
        data = [versions[i] for i in arr]
        yield [x for x in itertools.product(*data)]

def prettyPrintPolicy(pol):
    if len(pol[0]) == 1:
        return pol[0][0]
    execList = 'RAJA::ExecList<'+','.join(pol[0])+'>'
    tileList = 'RAJA::TileList<'+','.join(pol[1])+'>'
    perm = 'RAJA::Permute<VarOps::index_sequence<' + ','.join(str(x) for x in pol[2]) + '>>'
    outerTile = 'RAJA::Tile<' + tileList + ',' + perm + '>'
    if "omp" in str(pol[0]):
        outerTile = 'RAJA::OMP_Parallel<' + outerTile + '>'
    nested = 'RAJA::NestedPolicy<' + execList + ',' + outerTile + '>'
    return nested

def prettyPrint(version, loopNest):
    for i,pol in enumerate(version):
        parent = loopNest[i]['parent']
        if parent == None:
            parent = "null"
        policy = prettyPrintPolicy(pol)
        yield "using Pol_Id_{}_Size_{}_Parent_{} = {};".format(i, loopNest[i]['size'], parent, policy)

def createFrom(jsonFilename):
    with open(jsonFilename) as curr:
        loopNest = json.load(curr)['data']
    constraints = findConstraints(loopNest)
    versions = [generateAll(v['size'], i) for i,v in enumerate(loopNest)]
    # cross product of each constraint -- may have duplicates
    for x in itertools.product(*expandVersionsGivenConstraints(versions, constraints)):
        checked = [None] * len(loopNest)
        omp = []
        valid = True
        for i,group in enumerate(x):
            for j,t in enumerate(group):
                # we can only have one loop with an openmp policy per loop nest group
                if "omp_" in str(t[1]):
                    omp.append(j)
                if checked[t[0]] == None:
                    checked[t[0]] = t[1]
                # remove non-matching combinations
                elif not (checked[t[0]] == t[1]):
                    valid = False
        if valid and len(omp) < 2:
            yield '\n'.join(x for x in prettyPrint(checked, loopNest))

def dumpHeadersFor(filename):
    gendir = 'gen' + os.sep + os.path.basename(filename).split('.')[0]
    os.makedirs(gendir, exist_ok=True)
    count = 0
    for i,v in enumerate(createFrom(filename)):
        outdir = gendir + os.sep + "{num:06d}".format(num=i)
        os.makedirs(outdir, exist_ok=True)
        with open(outdir + os.sep + "config.hpp", "w") as fp:
            fp.write(v)
        count += 1
    return [count, gendir]

def main():
    for i in range(1,len(sys.argv)):
        print("Processing " + sys.argv[i])
        print("{} files dumped to {}".format(*dumpHeadersFor(sys.argv[i])))

if __name__ == "__main__":
    main()
