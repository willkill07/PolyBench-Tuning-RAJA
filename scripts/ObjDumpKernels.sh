#!/usr/bin/env bash

objdump -d $@ \
| awk '
/^[[:xdigit:]]+ <[^>]*kernel[^>]*>:$/{
  flag=1
  print
  next
}
/^[[:xdigit:]]+ <.*>:$/{
  flag=0
}
flag' \
| sed -r '
s/^0+([0-9a-f]+)/\1/
s/ +#.*$//;s/^ +//
s/:\s+([0-9a-f][0-9a-f]\s)+\s*/:\t/
s/nop[wldb]\b.*$/nop/
/^[0-9a-f]+:\s+$/d
s/%(r[0-9]+b|[sd]il|[a-d][lh])\b/Reg8/g
s/%(r[0-9]+w|[sd]i|[bs]p|[a-d]x)\b/Reg16/g
s/%(r[0-9]+d|e([sd]i|[bs]p|[a-d]x))\b/Reg32/g
s/%(r[0-9]+|r([sd]i|[bs]p|[a-d]x))\b/Reg64/g
s/%xmm[0-9]+/Vec128/g
s/%ymm[0-9]+/Vec256/g
s/%zmm[0-9]+/Vec512/g
s/(-?0x[0-9a-f]+)?\([^)]+\)/Mem/g'

# awk -- consume matching "paragraphs"

# - remove leading zeros
# - remove trailing comments
# - remove extra whitespace
# - remove bytes
# - rename nop consistently
# - delete lines with no instruction
# - Reg8
# - Reg16
# - Reg32
# - Reg64
# - Vec128
# - Vec256
# - Vec512
# - num?(....) -> Mem
