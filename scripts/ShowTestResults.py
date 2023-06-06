# Script to show the contents of saved test results file in table format
from datetime import datetime
import sys
import os
from re import search
import statistics
import pandas
import numpy
from tabulate import tabulate

import json



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


def print_results(test_results):
    global market_change

    print("")
    # print("Summary:")

    # convert associative array into 'plain' array
    strat_stats = []
    if test_results:
        # calculate stats for each strategy
        for strategy in test_results:
            strat_stats.append([strategy,
                                test_results[strategy]['test_date'],
                                test_results[strategy]['num_test_days'],
                                test_results[strategy]['entries'],
                                test_results[strategy]['ave_profit'],
                                test_results[strategy]['tot_profit'],
                                test_results[strategy]['win_pct'],
                                test_results[strategy]['expectancy'],
                                test_results[strategy]['daily_profit'],
                                0.0])

        # create dataframe
        df = pandas.DataFrame(strat_stats,
                              columns=["Strategy", "Date", "Days", "Trades", "Average%",
                                       "Total%", "Win%", "Expectancy", "Daily%", "Rank"])

        df["Rank"] = df["Daily%"].rank(ascending=False, method='min')

        pandas.set_option('display.precision', 2)
        print("")
        hdrs = df.columns.values
        print(tabulate(df.sort_values(by=['Rank', "Expectancy"], ascending=[True, False]),
                       floatfmt=["", "", "d", "d", ".2f", ".2f", ".2f", ".2f", ".3f", ".0f"],
                       showindex="never", headers=hdrs, tablefmt='psql'))

    return


def main():
    global curr_line
    global infile
    global strat_results

    args = sys.argv[1:]

    num_args = len(args)

    if num_args == 0:
        print("Please specify the location of the test results file")
        sys.exit()

    file_name = args[0]
    if not os.path.isfile(file_name):
        print(f"File {file_name} does not exist. Exiting...")
        sys.exit()

    with open(file_name, "r") as rf:
        results = json.load(rf)
        print_results(results)
        print("")

if __name__ == '__main__':
    main()
