#!/bin/bash

for i in {1..5}
do
    case $i in
    1)
        outfile="lfsr"
        cmd="/home/xxa/Desktop/Arch_Labs/Lab2/cs251a-microbench-master/lfsr"
        ;;
    2)
        outfile="merge"
        cmd="/home/xxa/Desktop/Arch_Labs/Lab2/cs251a-microbench-master/merge"
        ;;
    3)
        outfile="mm"
        cmd="/home/xxa/Desktop/Arch_Labs/Lab2/cs251a-microbench-master/mm"
        ;;
    4)
        outfile="sieve"
        cmd="/home/xxa/Desktop/Arch_Labs/Lab2/cs251a-microbench-master/sieve"
        ;;
    5)
        outfile="spmv"
        cmd="/home/xxa/Desktop/Arch_Labs/Lab2/cs251a-microbench-master/spmv"
        ;;
    esac


    build/X86/gem5.opt \
        configs/example/se1.py \
        --cpu-type=DerivO3CPU \
        --mem-type=DDR3_1600_8x8 \
        --caches \
        --l1d_size=64kB \
        --l1i_size=64kB \
        --cpu-clock=1GHz \
        --cmd=$cmd \
        --l1d-hwp-type=MultiPrefetcher
    cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}0"
    echo -e "\n\n\n"

    build/X86/gem5.opt \
        configs/example/se.py \
        --cpu-type=X86MinorCPU \
        --mem-type=DDR3_1600_8x8 \
        --caches \
        --l1d_size=64kB \
        --l1i_size=64kB \
        --cpu-clock=1GHz \
        --cmd=$cmd \
        --l1d-hwp-type=MultiPrefetcher
    cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}1"
    echo -e "\n\n\n"

    build/X86/gem5.opt \
        configs/example/se2.py \
        --cpu-type=DerivO3CPU \
        --mem-type=DDR3_1600_8x8 \
        --caches \
        --l1d_size=64kB \
        --l1i_size=64kB \
        --cpu-clock=1GHz \
        --cmd=$cmd \
        --l1d-hwp-type=MultiPrefetcher
    cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}2"
    echo -e "\n\n\n"

    build/X86/gem5.opt \
        configs/example/se1.py \
        --cpu-type=DerivO3CPU \
        --mem-type=DDR3_1600_8x8 \
        --caches \
        --l1d_size=64kB \
        --l1i_size=64kB \
        --cpu-clock=4GHz \
        --cmd=$cmd \
        --l1d-hwp-type=MultiPrefetcher
    cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}3"
    echo -e "\n\n\n"

    build/X86/gem5.opt \
        configs/example/se1.py \
        --cpu-type=DerivO3CPU \
        --mem-type=DDR3_1600_8x8 \
        --caches \
        --l1d_size=64kB \
        --l1i_size=64kB \
        --l2cache \
        --l2_size=256kB \
        --cpu-clock=1GHz \
        --cmd=$cmd \
        --l1d-hwp-type=MultiPrefetcher \
        --l2-hwp-type=MultiPrefetcher 
    cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}4"
    echo -e "\n\n\n"

    build/X86/gem5.opt \
        configs/example/se1.py \
        --cpu-type=DerivO3CPU \
        --mem-type=DDR3_1600_8x8 \
        --caches \
        --l1d_size=64kB \
        --l1i_size=64kB \
        --l2cache \
        --l2_size=2MB \
        --cpu-clock=1GHz \
        --cmd=$cmd \
        --l1d-hwp-type=MultiPrefetcher \
        --l2-hwp-type=MultiPrefetcher
    cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}5"
    echo -e "\n\n\n"

    build/X86/gem5.opt \
        configs/example/se1.py \
        --cpu-type=DerivO3CPU \
        --mem-type=DDR3_1600_8x8 \
        --caches \
        --l1d_size=64kB \
        --l1i_size=64kB \
        --l2cache \
        --l2_size=16MB \
        --cpu-clock=1GHz \
        --cmd=$cmd \
        --l1d-hwp-type=MultiPrefetcher \
        --l2-hwp-type=MultiPrefetcher
    cp -r m5out "/home/xxa/Desktop/tmpfile/${outfile}6"
    echo -e "\n\n\n"
    echo -e "\n\n\n"

done
