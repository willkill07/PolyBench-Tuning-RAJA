
## ExtractJSON

creates `json/<bench>.json` (autotuning loop configuration) from source code analysis

## GenerateHeaders

creates `gen/<bench>/[0-9]*/config.hpp` (autotuning headers) from json and `data/autotuning.json` (search space)

## CompileVersions

creates `out/<bench>/[0-9]*` (executables) from each header file corresponding in `gen/<bench>/[0-9]`

## CompileAll

convenience wrapper for `CompileVersions`

## ObjDumpKernels

dumps out x86-assembly-like representation of any function matching `kernel_.*`

## DumpAssembly

convenience wrapper for `ObjDumpKernels`
