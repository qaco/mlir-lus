#!/bin/bash

heptc mymath.epi
heptc -target c model.ept
# gcc -O3 -I /home/hpompougnac/.opam/default/lib/heptagon/c -I./ -c _main.c
# gcc -O3 -I /home/hpompougnac/.opam/default/lib/heptagon/c -I./ -c model_c/model.c
gcc -O3 -I /home/hpompougnac/.opam/default/lib/heptagon/c -I./ -c _main.c
gcc -O3 -I /home/hpompougnac/.opam/default/lib/heptagon/c -I./ -c model_c/model.c
gcc  _main.o model.o -lm -o model
rm *.o
rm *.epci
rm *.log
rm *.mls
rm -rf model_c
