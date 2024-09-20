#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
BLUE='\033[1;34m'

print_wrong() {
    printf "  ${RED}$1${NC}\n"
}

print_ok() {
    printf "  ${GREEN}$1${NC}\n"
}

check_ok () {
    echo "* $1"
    if output="$(mlirlus $1 $2 > /dev/null 2> /dev/null)";
    then
	print_ok "mlirlus $2"
    else
	print_wrong "mlirlus $2"
    fi
    if [ -n "$3" ]; then
	if output="$(mlirlus $1 $3 > /dev/null 2> /dev/null)";
	then
	    print_ok "mirlus $3"
	else
	    print_wrong "mlirlus $3"
	fi
    fi
}

check_wrong () {
    echo "* $1"
    if output="$(mlirlus $1 $2 > /dev/null 2> /dev/null)";
    then
	print_wrong "mlirlus $2"
    else
	print_ok "mlirlus $2"
    fi
    if [ -n "$3" ]; then
	if output="$(mlirlus $1 $3 > /dev/null 2> /dev/null)";
	then
	    print_wrong "mlirlus $3"
	else
	    print_ok "mlirlus $3"
	fi
    fi
}

printf "\n${BLUE}CLASSIC CLOCK CALCULUS${NC}\n\n"
for file in ok*;do
    check_ok ${file} "--classic-clock-calculus"
done

for file in wrong*;do
    check_wrong ${file} "--classic-clock-calculus"
done

printf "\n${BLUE}NORMALIZATION${NC}\n\n"
for file in ok*;do
    check_ok ${file} "--normalize"
done

for file in wrong*;do
    check_wrong ${file} "--normalize"
done

printf "\n${BLUE}COMPILATION TO SYNC${NC}\n\n"
for file in ok*;do
    check_ok ${file} "--normalize --to-sync-automata" "--normalize --invert-control"
done

for file in wrong*;do
    check_wrong ${file} "--normalize --to-sync-automata" "--normalize --invert-control"
done

printf "\n${BLUE}COMPILATION TO STD${NC}\n\n"
for file in ok*;do
    check_ok ${file} "--normalize --to-sync-automata --sync-to-std" "--normalize --invert-control --sync-to-std"
done

for file in wrong*;do
    check_wrong ${file} "--normalize --to-sync-automata --sync-to-std" "--normalize --invert-control --sync-to-std"
done
