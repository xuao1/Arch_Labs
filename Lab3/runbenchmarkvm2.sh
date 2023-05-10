#!/bin/bash

outfile="mm"
cmd="/home/xxa/Desktop/Arch_Labs/Lab2/cs251a-microbench-master/mm"

build/X86/gem5.opt \
    configs/example/se1.py \
    --cpu-type=O3CPU \
    --mem-type=DDR3_1600_8x8 \
    --caches \
    --l1d_size=1kB \
    --l1i_size=64kB \
    --l2cache \
    --l2_size=2MB \
    --sys-clock=2GHz \
    --cpu-clock=2GHz \
    --l1d_repl=RandomRP \
    --l2_repl=RandomRP \
    --l1d_assoc=4 \
    --cmd=$cmd
cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}0"
echo -e "\n\n\n"

build/X86/gem5.opt \
    configs/example/se1.py \
    --cpu-type=O3CPU \
    --mem-type=DDR3_1600_8x8 \
    --caches \
    --l1d_size=1kB \
    --l1i_size=64kB \
    --l2cache \
    --l2_size=2MB \
    --sys-clock=2GHz \
    --cpu-clock=2GHz \
    --l1d_repl=RandomRP \
    --l2_repl=RandomRP \
    --l1d_assoc=8 \
    --cmd=$cmd
cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}1"
echo -e "\n\n\n"

build/X86/gem5.opt \
    configs/example/se1.py \
    --cpu-type=O3CPU \
    --mem-type=DDR3_1600_8x8 \
    --caches \
    --l1d_size=1kB \
    --l1i_size=64kB \
    --l2cache \
    --l2_size=2MB \
    --sys-clock=2GHz \
    --cpu-clock=2GHz \
    --l1d_repl=RandomRP \
    --l2_repl=RandomRP \
    --l1d_assoc=16 \
    --cmd=$cmd
cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}2"
echo -e "\n\n\n"

build/X86/gem5.opt \
    configs/example/se1.py \
    --cpu-type=O3CPU \
    --mem-type=DDR3_1600_8x8 \
    --caches \
    --l1d_size=1kB \
    --l1i_size=64kB \
    --l2cache \
    --l2_size=2MB \
    --sys-clock=2GHz \
    --cpu-clock=2GHz \
    --l1d_repl=NMRURP \
    --l2_repl=NMRURP \
    --l1d_assoc=4 \
    --cmd=$cmd
cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}3"
echo -e "\n\n\n"

build/X86/gem5.opt \
    configs/example/se1.py \
    --cpu-type=O3CPU \
    --mem-type=DDR3_1600_8x8 \
    --caches \
    --l1d_size=1kB \
    --l1i_size=64kB \
    --l2cache \
    --l2_size=2MB \
    --sys-clock=2GHz \
    --cpu-clock=2GHz \
    --l1d_repl=NMRURP \
    --l2_repl=NMRURP \
    --l1d_assoc=8 \
    --cmd=$cmd
cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}4"
echo -e "\n\n\n"

build/X86/gem5.opt \
    configs/example/se1.py \
    --cpu-type=O3CPU \
    --mem-type=DDR3_1600_8x8 \
    --caches \
    --l1d_size=1kB \
    --l1i_size=64kB \
    --l2cache \
    --l2_size=2MB \
    --sys-clock=2GHz \
    --cpu-clock=2GHz \
    --l1d_repl=NMRURP \
    --l2_repl=NMRURP \
    --l1d_assoc=16 \
    --cmd=$cmd
cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}5"
echo -e "\n\n\n"

build/X86/gem5.opt \
    configs/example/se1.py \
    --cpu-type=O3CPU \
    --mem-type=DDR3_1600_8x8 \
    --caches \
    --l1d_size=1kB \
    --l1i_size=64kB \
    --l2cache \
    --l2_size=2MB \
    --sys-clock=2GHz \
    --cpu-clock=2GHz \
    --l1d_repl=LIPRP \
    --l2_repl=LIPRP \
    --l1d_assoc=4 \
    --cmd=$cmd
cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}6"
echo -e "\n\n\n"

build/X86/gem5.opt \
    configs/example/se1.py \
    --cpu-type=O3CPU \
    --mem-type=DDR3_1600_8x8 \
    --caches \
    --l1d_size=1kB \
    --l1i_size=64kB \
    --l2cache \
    --l2_size=2MB \
    --sys-clock=2GHz \
    --cpu-clock=2GHz \
    --l1d_repl=LIPRP \
    --l2_repl=LIPRP \
    --l1d_assoc=8 \
    --cmd=$cmd
cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}7"
echo -e "\n\n\n"

build/X86/gem5.opt \
    configs/example/se1.py \
    --cpu-type=O3CPU \
    --mem-type=DDR3_1600_8x8 \
    --caches \
    --l1d_size=1kB \
    --l1i_size=64kB \
    --l2cache \
    --l2_size=2MB \
    --sys-clock=2GHz \
    --cpu-clock=2GHz \
    --l1d_repl=LIPRP \
    --l2_repl=LIPRP \
    --l1d_assoc=16 \
    --cmd=$cmd
cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}8"
echo -e "\n\n\n"

