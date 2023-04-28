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
strat_results = {}
strat_summary = {}


# routine to skip to requested pattern
def skipto(pattern, anywhere=False) -> bool:
    global curr_line
    global infile

    if infile is None:
        print("ERR: file not open")
        return False

    curr_line = infile.readline()

    while curr_line:
        if anywhere:
            found = (pattern in curr_line) # pattern is anywhere in the string
        else:
            found = curr_line.lstrip().startswith(pattern) # string starts with pattern (ignoring whitespace)
        if found:
            break
        curr_line = infile.readline()

    if curr_line:
        return True
    else:
        return False

# copies the input file and prints each line until pattern is found.
# Note: prints current line but not the final line
def copyto(pattern, anywhere=False):
    global curr_line
    global infile

    while curr_line:
        if anywhere:
            found = (pattern in curr_line) # pattern is anywhere in the string
        else:
            found = curr_line.lstrip().startswith(pattern) # string starts with pattern (ignoring whitespace)
        if found:
            break
        print(curr_line.rstrip())
        curr_line = infile.readline()

    if curr_line:
        return True
    else:
        return False


def rank_simple(vector):
    return sorted(range(len(vector)), key=vector.__getitem__)


def rankdata(a):
    n = len(a)
    ivec = rank_simple(a)
    svec = [a[rank] for rank in ivec]
    sumranks = 0
    dupcount = 0
    newarray = [0] * n
    for i in range(n):
        sumranks += i
        dupcount += 1
        if i == n - 1 or svec[i] != svec[i + 1]:
            averank = sumranks / float(dupcount) + 1
            for j in range(i - dupcount + 1, i + 1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray


def process_totals(strat, line):
    global strat_summary
    global strat_results

    # format of line:
    # | TOTAL | Entries | Avg Profit % | Cum Profit % |  Tot Profit USD | Tot Profit % | Avg Duration | Win  Draw  Loss  Win% |
    cols = line.strip().split("|")

    cols.pop(0)
    cols.pop(len(cols) - 1)

    entry = {}
    entry['entries'] = int(cols[1])
    entry['ave_profit'] = float(cols[2])
    entry['tot_profit'] = float(cols[5])
    entry['win_pct'] = float(cols[7].strip().split(" ")[-1])
    entry['expectancy'] = 0 # updated later

    strat_summary[strat] = entry

    return

def process_expectancy(strat, line):
    global strat_summary
    global strat_results

    # format of line:
    # | Expectancy                  | -0.05               |
    cols = line.strip().split("|")
    cols.pop(0)
    cols.pop(len(cols) - 1)

    strat_summary[strat]['expectancy'] = float(cols[-1])

    return


def print_results():
    global strat_summary
    global strat_results

    print("")
    # print("Summary:")

    # convert associative array into 'plain' array
    strat_stats = []
    if strat_summary:
        # calculate stats for each strategy
        for strategy in strat_summary:
            strat_stats.append([strategy,
                                strat_summary[strategy]['entries'], strat_summary[strategy]['ave_profit'],
                                strat_summary[strategy]['tot_profit'], strat_summary[strategy]['win_pct'],
                                strat_summary[strategy]['expectancy'],
                                0.0])

        # create dataframe
        df = pandas.DataFrame(strat_stats,
                              columns=["Strategy", "Trades", "Average%",
                                       "Total%", "Win%", "Expectancy", "Rank"])

        df["Rank"] = df["Total%"].rank(ascending=False, method='min')

        pandas.set_option('display.precision', 2)
        print("")
        hdrs = df.columns.values
        print(tabulate(df.sort_values(by=['Rank'], ascending=True), showindex="never", headers=hdrs, tablefmt='psql'))

    return


def main():
    global curr_line
    global infile
    global strat_results

    args = sys.argv[1:]

    file_name = args[0]
    if not os.path.isfile(file_name):
        print("File {} does not exist. Exiting...".format(file_name))
        sys.exit()

    infile = open(file_name)

    # repeatedly scan file and find header of new run, then print results
    while skipto("Result for strategy ", anywhere=True):
        strat = curr_line.rstrip().split(" ")[-1]
        # print("")
        # print("------------")
        # print(strat)
        # print("------------")
        # print("")
        # copyto('TOTAL', anywhere=True)
        skipto('TOTAL', anywhere=True)
        process_totals(strat, curr_line.rstrip())
        # copyto('================== SUMMARY METRICS', anywhere=True)
        skipto('================== SUMMARY METRICS', anywhere=True)
        if strat_summary[strat]['entries'] > 0:
            # copyto('Expectancy', anywhere=True)
            skipto('Expectancy', anywhere=True)
            # print(curr_line.rstrip())
            process_expectancy(strat, curr_line.rstrip())
        # copyto('===============================')
        skipto('===============================')
        # print(curr_line.rstrip())
        # print("")

    print_results()


if __name__ == '__main__':
    main()
