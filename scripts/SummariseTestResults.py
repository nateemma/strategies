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
issues_found = False


# routine to skip to requested pattern.
# `stop`: if given, abort (return False) as soon as a line containing `stop` is
# seen, LEAVING curr_line on that boundary line (so a caller scanning for the
# next strategy block isn't consumed past it). This keeps an optional per-block
# lookahead (e.g. drawdown/Calmar) from running to EOF when the line is absent.
def skipto(pattern, anywhere=False, stop=None) -> bool:
    global curr_line
    global infile

    if infile is None:
        print("ERR: file not open")
        return False

    curr_line = infile.readline()

    while curr_line:
        if anywhere:
            found = pattern in curr_line  # pattern is anywhere in the string
        else:
            found = curr_line.lstrip().startswith(
                pattern
            )  # string starts with pattern (ignoring whitespace)
        if found:
            return True
        if stop is not None and stop in curr_line:
            return False  # hit the boundary; leave curr_line here for the caller
        curr_line = infile.readline()

    # EOF
    return False


# copies the input file and prints each line until pattern is found.
# Note: prints current line but not the final line
def copyto(pattern, anywhere=False):
    global curr_line
    global infile

    while curr_line:
        if anywhere:
            found = pattern in curr_line  # pattern is anywhere in the string
        else:
            found = curr_line.lstrip().startswith(
                pattern
            )  # string starts with pattern (ignoring whitespace)
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
    exchange = exchange.strip().replace(".", "")
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
    print(f"Test Date:\t{test_date}")
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


def get_empty_strat_result():
    entry = {}
    entry["test_date"] = ""
    entry["num_test_days"] = int(num_test_days)
    entry["entries"] = 0
    entry["daily_trades"] = 0
    entry["ave_profit"] = 0
    entry["tot_profit"] = 0
    entry["win_pct"] = 0
    entry["expectancy"] = 0
    entry["daily_profit"] = 0
    entry["vs_market"] = 0
    entry["drawdown"] = 0.0
    entry["calmar"] = 0.0
    return entry


def process_totals(strat, line):
    global strat_results
    global strat_results
    global test_date
    global num_test_days

    # format of line:

    # ┃      TOTAL ┃ Trades ┃ Avg Profit % ┃ Tot Profit USDT ┃ Tot Profit % ┃ Avg Duration ┃  Win  Draw  Loss  Win% ┃

    # if "|" not in line and "│" not in line:
    #     # Unexpected line format - skip to avoid crash
    #     return
    sep = "|" if "|" in line else "│"
    cols = [c.strip() for c in line.strip().split(sep)]
    cols = [c for c in cols if c]

    # print(f'cols: {cols}')

    if len(cols) < 7 or cols[0] != "TOTAL":
        # Not enough columns for totals parsing
        return

    entry = {}
    entry["test_date"] = str(test_date)
    entry["num_test_days"] = int(num_test_days)
    entry["entries"] = int(cols[1])
    entry["daily_trades"] = float(cols[1]) / float(num_test_days)
    entry["ave_profit"] = float(cols[2])
    entry["tot_profit"] = float(cols[4])
    entry["win_pct"] = float(cols[6].strip().split(" ")[-1])
    entry["expectancy"] = 0  # updated later
    entry["daily_profit"] = 0  # updated later
    entry["vs_market"] = 0  # updated later
    entry["drawdown"] = 0.0  # updated later
    entry["calmar"] = 0.0  # updated later

    strat_results[strat] = entry

    # print(f'entry: {entry}')

    return


def process_expectancy(strat, line):
    global strat_results
    global strat_results
    global issues_found

    # format of line:
    # │ Expectancy (Ratio)            │ 54.00 (100.00)                 │
    if "|" not in line and "│" not in line:
        return
    sep = "|" if "|" in line else "│"
    cols = [c.strip() for c in line.strip().split(sep)]
    cols = [c for c in cols if c]
    if len(cols) < 2:
        return

    value = cols[-1]
    # Expectancy (Ratio) value looks like "54.00 (100.00)"
    if "(" in value:
        value = value.split("(")[0].strip()

    strat_results[strat]["expectancy"] = float(value)

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
    strat_results[strat]["daily_profit"] = round(
        float(strat_results[strat]["tot_profit"] / num_test_days), 3
    )

    return


def process_market_change(strat, line):
    global strat_results
    global strat_results
    global market_change

    # format of line:
    # | Market change                   | -16.55%                |
    if "|" not in line and "│" not in line:
        return
    sep = "|" if "|" in line else "│"
    cols = [c.strip() for c in line.strip().split(sep)]
    cols = [c for c in cols if c]
    if len(cols) < 2:
        return

    mkt_change = str(cols[-1]).strip()
    market_change = float(mkt_change.strip("%"))

    return


def process_drawdown(strat, line):
    global strat_results

    # format of line (wallet-based = TRUE mark-to-market drawdown):
    # │ Max % of account underwater (balance)  │ 28.50%                 │
    if "|" not in line and "│" not in line:
        return
    sep = "|" if "|" in line else "│"
    cols = [c.strip() for c in line.strip().split(sep)]
    cols = [c for c in cols if c]
    if len(cols) < 2:
        return
    try:
        strat_results[strat]["drawdown"] = float(str(cols[-1]).strip().strip("%"))
    except ValueError:
        pass
    return


def process_calmar(strat, line):
    global strat_results

    # format of line:
    # │ Calmar (daily wallet balance)          │ 4.59                   │
    if "|" not in line and "│" not in line:
        return
    sep = "|" if "|" in line else "│"
    cols = [c.strip() for c in line.strip().split(sep)]
    cols = [c for c in cols if c]
    if len(cols) < 2:
        return
    try:
        strat_results[strat]["calmar"] = float(str(cols[-1]).strip())
    except ValueError:
        pass
    return


def print_results(test_results):
    global market_change
    global issues_found

    print(f"Market Change(%): {market_change}")
    print("")
    # print("Summary:")

    # convert associative array into 'plain' array
    strat_stats = []
    if test_results:

        # calculate stats for each strategy
        for strategy in test_results:
            test_results[strategy]["vs_market"] = (
                test_results[strategy]["tot_profit"] - market_change
            )
            strat_stats.append(
                [
                    strategy,
                    test_results[strategy]["entries"],
                    test_results[strategy]["daily_trades"],
                    test_results[strategy]["ave_profit"],
                    test_results[strategy]["tot_profit"],
                    test_results[strategy]["vs_market"],
                    test_results[strategy]["drawdown"],
                    test_results[strategy]["calmar"],
                    test_results[strategy]["win_pct"],
                    test_results[strategy]["expectancy"],
                    test_results[strategy]["daily_profit"],
                    0,
                ]
            )

        # create dataframe
        df = pandas.DataFrame(
            strat_stats,
            columns=[
                "Strategy",
                "Trades",
                "Tr/day",
                "Average%",
                "Total%",
                "vs Mkt%",
                "MaxDD%",
                "Calmar",
                "Win%",
                "Expectancy",
                "Daily%",
                "Rank",
            ],
        )

        rank1 = df["Tr/day"].rank(ascending=False, method="min", pct=False)
        # rank1 = df["Total%"].rank(ascending=False, method='min', pct=False)
        rank2 = df["Daily%"].rank(ascending=False, method="min", pct=False)
        rank3 = df["Win%"].rank(ascending=False, method="min", pct=False)
        rank4 = df["Expectancy"].rank(ascending=False, method="min", pct=False)
        rank5 = df["Total%"].rank(ascending=False, method="min", pct=False)
        # risk terms: lower drawdown = better (ascending); higher Calmar
        # (risk-adjusted return) = better (descending).
        rank6 = df["MaxDD%"].rank(ascending=True, method="min", pct=False)
        rank7 = df["Calmar"].rank(ascending=False, method="min", pct=False)
        # rank_mean = np.mean([rank1, rank2, rank3, rank4, rank5], axis=0)
        # NOTE: Daily% == Total%/num_days, i.e. the SAME ranking within a run, so
        # include only Daily% (drop Total%) to avoid double-weighting aggregate
        # return. Rank = aggregate return (Daily%) + frequency (Win%, immune to
        # a single big gain) + per-trade edge (Expectancy) + risk (drawdown) +
        # risk-adjusted return (Calmar).
        rank_mean = np.mean([rank2, rank3, rank4, rank6, rank7], axis=0)
        # 0-trade strategies (a failed model, or the inert base) otherwise rank
        # WELL on the 0-default DD/Calmar terms — force them to the bottom.
        rank_mean = np.where(df["Trades"].values == 0, len(df) + 1, rank_mean)
        # print(f'rank_mean: {rank_mean}')
        df["Rank"] = scipy.stats.rankdata(rank_mean)

        pandas.set_option("display.precision", 2)
        print("")
        hdrs = df.columns.tolist()
        # print(tabulate(df.sort_values(by=['Rank', "Expectancy"], ascending=[True, False]),

        print(
            tabulate(
                df.sort_values(by=["Rank"], ascending=[True]),
                floatfmt=[
                    "",
                    "d",
                    ".2f",
                    ".2f",
                    ".1f",
                    ".1f",
                    ".1f",
                    ".2f",
                    ".1f",
                    ".2f",
                    ".2f",
                    ".0f",
                ],
                showindex="never",
                headers=hdrs,
                tablefmt="psql",
            )
        )

        if issues_found:
            print()
            print("*** Suspicious results found. Ave. Results set to -100")
            print()

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
    if skipto("exchange:", anywhere=True):
        process_exchange(curr_line.rstrip())
    else:
        infile.close()
        infile = open(file_name)

    if skipto("Date/time:"):
        process_test_date(curr_line.rstrip())

        if skipto("Time range"):
            process_time_range(curr_line.rstrip())

    # repeatedly scan file and find header of new run, then print results
    found_marker = skipto("Result for strategy ", anywhere=True)
    while found_marker:
        strat = curr_line.rstrip().split(" ")[-1]
        strat_results[strat] = get_empty_strat_result()

        # print("")
        # print("------------")
        # print(strat)
        # print("------------")
        # print("")
        # Bound TOTAL / SUMMARY METRICS to this strategy's block: a 0-trade
        # strategy has NO data tables ("No trades made"), so an unbounded
        # forward scan would consume the NEXT strategy's block and drop it.
        if skipto("TOTAL", anywhere=True, stop="Result for strategy"):
            process_totals(strat, curr_line.rstrip())
            if skipto(" SUMMARY METRICS", anywhere=True, stop="Result for strategy"):
                if strat_results[strat]["entries"] > 0:
                    if skipto("Expectancy", anywhere=True):
                        process_expectancy(strat, curr_line.rstrip())

                        if skipto("daily profit", anywhere=True):
                            process_daily_profit(strat, curr_line.rstrip())

                        if market_change <= 0.0:
                            if skipto("Market change", anywhere=True):
                                process_market_change(strat, curr_line.rstrip())
                                # print(f"Market Change:\t{market_change}")
                            else:
                                market_change = 0.0

                        # Risk metrics — wallet-based (the TRUE mark-to-market
                        # drawdown / Calmar, correct for hold and fast strategies
                        # alike). Standard freqtrade output, so group-agnostic.
                        # Bounded by stop="Result for strategy" so that a log
                        # WITHOUT these lines (older runs) doesn't consume the
                        # rest of the file — it just leaves the defaults (0.0).
                        if skipto(
                            "Max % of account underwater (balance)",
                            anywhere=True,
                            stop="Result for strategy",
                        ):
                            process_drawdown(strat, curr_line.rstrip())
                            if skipto(
                                "Calmar (daily wallet",
                                anywhere=True,
                                stop="Result for strategy",
                            ):
                                process_calmar(strat, curr_line.rstrip())

                        # copyto('===============================')
                        # skipto('===============================')
                        # print(curr_line.rstrip())
                        # print("")

        # A bounded DD/Calmar lookahead may have stopped ON the next strategy
        # marker; if so, process it rather than skipping past it.
        if "Result for strategy " in curr_line:
            found_marker = True
        else:
            found_marker = skipto("Result for strategy ", anywhere=True)

    print_results(strat_results)
    print("")

    update_saved_results(strat_results)


if __name__ == "__main__":
    main()
