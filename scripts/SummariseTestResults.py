# Script to process hyperopt log and summarise results. Useful for multiple hyperopts in one file (e.g. from hyp_exchange.sh)
from datetime import datetime
import sys
import os
from re import search
import statistics
import pandas
import numpy as np
import scipy
from tabulate import tabulate

import json

infile = None
curr_line = ""
strat_results = {}
market_change = 0.0
test_date = None
num_test_days = 0
exchange = ""


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
        # print(f"*** Target not found: {pattern}")
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


def process_exchange(line):
    global exchange

    # line format:
    # Testing strategy list for exchange: binanceus...

    exchange = line.split(":")[-1]
    exchange = exchange.strip().replace(".","")
    return

def process_test_date(line):
    global test_date

    # line format:
    # Date/time: Wed May 31 08:42:11 PDT 2023
    date_string = line.strip().split(": ")[-1]
    input_format = "%a %b %d %H:%M:%S %Z %Y"
    date_object = datetime.strptime(date_string, input_format)
    output_format = "%Y %b %d"
    test_date = date_object.strftime(output_format)

    print("")
    print(f'Test Date:\t{test_date}')
    return

def process_time_range(line):
    global num_test_days

    # line format:
    # Time range: 20220605-20230531
    date_string = line.strip().split(":")[-1]
    start_date, end_date = date_string.split("-")
    start_date = datetime.strptime(start_date.strip(), "%Y%m%d")
    end_date = datetime.strptime(end_date.strip(), "%Y%m%d")
    date_diff = end_date - start_date
    num_test_days = date_diff.days

    print(f"No. Test Days:\t{num_test_days}")

    return
def process_totals(strat, line):
    global strat_results
    global strat_results
    global test_date
    global num_test_days

    # format of line:
    # | TOTAL | Entries | Avg Profit % | Cum Profit % |  Tot Profit USD | Tot Profit % | Avg Duration | Win  Draw  Loss  Win% |
    cols = line.strip().split("|")

    cols.pop(0)
    cols.pop(len(cols) - 1)

    entry = {}
    entry['test_date'] = str(test_date)
    entry['num_test_days'] = int(num_test_days)
    entry['entries'] = int(cols[1])
    entry['daily_trades'] = float(cols[1]) / float(num_test_days)
    entry['ave_profit'] = float(cols[2])
    entry['tot_profit'] = float(cols[5])
    entry['win_pct'] = float(cols[7].strip().split(" ")[-1])
    entry['expectancy'] = 0 # updated later
    entry['daily_profit'] = 0 # updated later
    entry['vs_market'] = 0 # updated later

    strat_results[strat] = entry

    return

def process_expectancy(strat, line):
    global strat_results
    global strat_results

    # format of line:
    # | Expectancy                  | -0.05               |
    cols = line.strip().split("|")
    cols.pop(0)
    cols.pop(len(cols) - 1)

    cols = cols[-1]

    if "(" in cols:
        cols = cols.strip().split("(")[0]

    # print(f"cols:{cols}")

    strat_results[strat]['expectancy'] = float(cols)

    return

def process_daily_profit(strat, line):
    global strat_results
    global strat_results

    # # format of line:
    # # | Avg. daily profit %         | -0.01%              |
    # cols = line.strip().split("|")
    # cols.pop(0)
    # cols.pop(len(cols) - 1)
    #
    # strat_results[strat]['daily_profit'] = float(cols[-1].replace("%",""))

    # entry in test output is not very accurate, so just calculate
    strat_results[strat]['daily_profit'] = round(float(strat_results[strat]['tot_profit']/num_test_days), 3)

    return

def process_market_change(strat, line):
    global strat_results
    global strat_results
    global market_change

    # format of line:
    # | Market change                   | -16.55%                |
    cols = line.strip().split("|")
    cols.pop(0)
    cols.pop(len(cols) - 1)

    mkt_change = str(cols[-1]).strip()
    market_change = float(mkt_change.strip("%"))

    return


def print_results(test_results):
    global market_change

    print("")
    # print("Summary:")

    # convert associative array into 'plain' array
    strat_stats = []
    if test_results:
            
        # calculate stats for each strategy
        for strategy in test_results:
            test_results[strategy]['vs_market'] = test_results[strategy]['tot_profit'] - market_change
            strat_stats.append([strategy,
                                test_results[strategy]['entries'],
                                test_results[strategy]['daily_trades'],
                                test_results[strategy]['ave_profit'],
                                test_results[strategy]['tot_profit'],
                                test_results[strategy]['vs_market'],
                                test_results[strategy]['win_pct'],
                                test_results[strategy]['expectancy'],
                                test_results[strategy]['daily_profit'],
                                0])

        # create dataframe
        df = pandas.DataFrame(strat_stats,
                              columns=["Strategy", "Trades", "Tr/day", "Average%",
                                       "Total%", "vs Mkt%", "Win%", "Expectancy", "Daily%", "Rank"])

        rank1 = df["Tr/day"].rank(ascending=False, method='min', pct=False)
        rank2 = df["Average%"].rank(ascending=False, method='min', pct=False)
        rank3 = df["Win%"].rank(ascending=False, method='min', pct=False)
        rank4 = df["Expectancy"].rank(ascending=False, method='min', pct=False)
        rank5 = df["Daily%"].rank(ascending=False, method='min', pct=False)
        # rank_mean = np.mean([rank1, rank2, rank3, rank4, rank5], axis=0)
        # rank_mean = np.mean([rank1, rank2, rank4, rank5], axis=0)
        rank_mean = np.mean([rank1, rank3, rank4, rank5], axis=0)
        # print(f'rank_mean: {rank_mean}')
        df["Rank"] = scipy.stats.rankdata(rank_mean)

        pandas.set_option('display.precision', 2)
        print("")
        hdrs = df.columns.values
        print(tabulate(df.sort_values(by=['Rank', "Expectancy"], ascending=[True, False]),
                       floatfmt=["", "d", ".2f", ".2f", ".1f", ".1f", ".1f", ".2f", ".2f", ".0f"],
                       showindex="never", headers=hdrs, tablefmt='psql'))

    return


def update_saved_results(curr_results):

    global exchange

    results_file = f"./user_data/strategies/{exchange}/test_results.json"


    # if file exists, load it
    if os.path.isfile(results_file):
        print(f"Loading prior results from {results_file}")
        with open(results_file, "r") as rf:
            results = json.load(rf)
    else:
        results = {}

    # add the current results
    for strat in curr_results:
        results[strat] = curr_results[strat]

    # Save to the file
    with open(results_file, "w") as rf:
        print(f"Saving updated results to {results_file}")
        json.dump(results, rf, indent=4)

    return

def main():
    global curr_line
    global infile
    global strat_results
    global market_change

    args = sys.argv[1:]

    file_name = args[0]
    if not os.path.isfile(file_name):
        print("File {} does not exist. Exiting...".format(file_name))
        sys.exit()

    infile = open(file_name)

    # get header data
    if skipto('exchange:', anywhere=True):
        process_exchange(curr_line.rstrip())
    else:
        infile.close()
        infile = open(file_name)

    if skipto('Date/time:'):
        process_test_date(curr_line.rstrip())

        if skipto('Time range'):
            process_time_range(curr_line.rstrip())

    # repeatedly scan file and find header of new run, then print results
    while skipto("Result for strategy ", anywhere=True):
        strat = curr_line.rstrip().split(" ")[-1]
        # print("")
        # print("------------")
        # print(strat)
        # print("------------")
        # print("")
        # copyto('TOTAL', anywhere=True)
        if skipto('TOTAL', anywhere=True):
            process_totals(strat, curr_line.rstrip())
            # copyto('================== SUMMARY METRICS', anywhere=True)
            if skipto('================== SUMMARY METRICS', anywhere=True):
                if strat_results[strat]['entries'] > 0:
                    if skipto('Expectancy', anywhere=True):
                        process_expectancy(strat, curr_line.rstrip())

                        if skipto('daily profit %', anywhere=True):
                            process_daily_profit(strat, curr_line.rstrip())

                        if market_change <= 0.0:
                            if skipto('Market change', anywhere=True):
                                process_market_change(strat, curr_line.rstrip())
                                print(f"Market Change:\t{market_change}")
                            else:
                                market_change = 0.0

                        # copyto('===============================')
                        skipto('===============================')
                        # print(curr_line.rstrip())
                        # print("")

    print_results(strat_results)
    print("")

    update_saved_results(strat_results)

if __name__ == '__main__':
    main()
