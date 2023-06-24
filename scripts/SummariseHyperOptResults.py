
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
def copyto(pattern, anywhere=False, include_str=""):
    global curr_line
    global infile

    while curr_line:
        if anywhere:
            found = (pattern in curr_line) # pattern is anywhere in the string
        else:
            found = curr_line.lstrip().startswith(pattern) # string starts with pattern (ignoring whitespace)
        if found:
            break

        if (len(include_str) == 0) or (include_str in curr_line):
            print(curr_line.rstrip())
        curr_line = infile.readline()

    if curr_line:
        return True
    else:
        return False

def process_totals(strat, line):
    global strat_summary
    global strat_results

    # format of line:
    # 97/100:     94 trades. 63/0/31 Wins/Draws/Losses. Avg profit   0.59%. Median profit   1.15%. Total profit 1658.03907912 USD (  16.58%). Avg duration 22:16:00 min. Objective: -30.83651
    cols = " ".join(line[1:].split()) # get rid of multiple spaces (and 1st character, which could be '*')
    cols = cols.strip().split(" ")

    # print("cols: ", cols)

    entry = {}
    entry['entries'] = int(cols[1])
    wins, draws, losses = cols[3].strip().split("/")
    entry['ave_profit'] = float(cols[7].split('%')[0])
    entry['tot_profit'] = float(cols[16].split('%')[0])
    entry['win_pct'] = 100.0 * float(wins) / float(entry['entries'])

    strat_summary[strat] = entry

    return

def print_results():

    global strat_summary
    global strat_results

    print("")
    print("Summary:")

    if len(strat_summary) > 0:
        # convert associative array into 'plain' array
        strat_stats = []
        # calculate stats for each strategy
        for strategy in strat_summary:
            strat_stats.append([strategy,
                                strat_summary[strategy]['entries'], strat_summary[strategy]['ave_profit'],
                                strat_summary[strategy]['tot_profit'], strat_summary[strategy]['win_pct'],
                               0.0])

        # create dataframe
        df = pandas.DataFrame(strat_stats,
                              columns=["Strategy", "Trades", "Average(%)", "Total(%)", "Win%", "Rank"])

        df["Rank"] = df["Total(%)"].rank(ascending=False, method='min')

        pandas.set_option('display.precision', 2)
        print("")
        hdrs = df.columns.values
        print(tabulate(df.sort_values(by=['Rank'], ascending=True),
                       showindex="never", headers=hdrs,
                       colalign=("left", "center", "decimal", "decimal", "decimal", "center"),
                       floatfmt=('.0f', '.0f', '.2f', '.2f', '.2f', '.0f'),
                       numalign="center", tablefmt='psql')
              )
    else:
        print("No results found")

    return

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
    while skipto("------------------"):
        print("")
        print(curr_line.rstrip())
        curr_line = infile.readline()
        strategy = curr_line.strip()

        copyto(("------------------"))
        print(curr_line.rstrip())
        # skip anything between header & results
        if skipto('+--------'):
            print(curr_line.rstrip())
            # get the best results line
            if copyto('Wins/Draws/Losses', anywhere=True, include_str="|"):
                process_totals(strategy, curr_line.strip())

                # copy everything up to end of results (assuming we don't need anything past ROI table)
                copyto('# ROI table:')

    print_results()

if __name__ == '__main__':
    main()