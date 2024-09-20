#!/bin/bash
set -x
heptc -c mymath.epi
heptc -target c resnet.ept
gcc -O3 -I /home/hpompougnac/.opam/default/lib/heptagon/c -I./ -c _main.c
gcc -O3 -I /home/hpompougnac/.opam/default/lib/heptagon/c -I./ -c timestamp.c
gcc -O3 -I /home/hpompougnac/.opam/default/lib/heptagon/c -I./ -c resnet_c/resnet.c
gcc -O3 _main.o resnet.o timestamp.o -lm -o resnet
rm *.o
rm *.epci
rm *.log
rm *.mls
rm -rf resnet_c
