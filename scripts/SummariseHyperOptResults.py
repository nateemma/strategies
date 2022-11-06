
# Script to process hyperopt log and summarise results. Useful for multiple hyperopts in one file (e.g. from hyp_exchange.sh)


import sys
import os
from re import search
import statistics
import pandas
import numpy
from tabulate import tabulate

infile = None
curr_line = ""

# routine to skip to requested pattern
def skipto(pattern) -> bool:
    global curr_line
    global infile

    if infile is None:
        print("ERR: file not open")
        return False

    curr_line = infile.readline()

    while curr_line:
        if pattern in curr_line:
            break
        curr_line = infile.readline()

    if curr_line:
        return True
    else:
        return False

# copies the input file and prints each line until pattern is found.
# Note: prints current line but not the final line
def copyto(pattern):
    global curr_line
    global infile

    while curr_line:
        if pattern in curr_line:
            break
        print(curr_line.rstrip())
        curr_line = infile.readline()

    if curr_line:
        return True
    else:
        return False


def main():
    global curr_line
    global infile

    args = sys.argv[1:]

    file_name = args[0]
    if not os.path.isfile(file_name):
        print("File {} does not exist. Exiting...".format(file_name))
        sys.exit()

    infile = open(file_name)

    # repeatedly scan file and find header of new run, then print results
    while skipto("----------------------"):
        print("")
        copyto('freqtrade hyperopt')
        print(curr_line.rstrip())
        # skip anything between header & results
        skipto('+--------+')
        # print everything up to end of results (assuming we don't need anything past ROI table)
        copyto('# ROI table:')

if __name__ == '__main__':
    main()