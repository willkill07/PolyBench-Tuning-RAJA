#!/usr/bin/env bash

objdump -d $@ | \
    awk '/^[[:xdigit:]]+ <[^>]*kernel[^>]*>:$/{flag=1;print;next}/^[[:xdigit:]]+ <.*>:$/{flag=0}flag' | \
    sed -r 's/^0+([0-9a-f]+)/\1/;s/ +#.*$//;s/^ +//;s/:\s+([0-9a-f][0-9a-f]\s)+\s*/:\t/;s/nop[wldb]\b.*$/nop/;/^[0-9a-f]+:\s+$/d'

# awk -- consume matching "paragraphs"

# sed -- remove leading zeros
#     -- remove trailing comments
#     -- remove extra whitespace
#     -- remove bytes
#     -- rename nop consistently
#     -- delete lines with no instruction
