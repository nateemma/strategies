
# Script to process monthly test results and summarise the statistics for each strategy


import sys
import os
from re import search
import statistics
import pandas
import numpy
from tabulate import tabulate


# parser states
FIND_HEADER = 0
SKIP_HEADER = 1
READ_DATA = 2

def rank_simple(vector):
    return sorted(range(len(vector)), key=vector.__getitem__)

def rankdata(a):
    n = len(a)
    ivec=rank_simple(a)
    svec=[a[rank] for rank in ivec]
    sumranks = 0
    dupcount = 0
    newarray = [0]*n
    for i in xrange(n):
        sumranks += i
        dupcount += 1
        if i==n-1 or svec[i] != svec[i+1]:
            averank = sumranks / float(dupcount) + 1
            for j in xrange(i-dupcount+1,i+1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray

def main():
    args = sys.argv[1:]

    infile = args[0]
    if not os.path.isfile(infile):
        print("File {} does not exist. Exiting...".format(infile))
        sys.exit()

    state = FIND_HEADER
    count = 1

    stratParams = {}

    # scan the file and read in the test data
    with open(infile) as f:
        line = f.readline()
        while line:
            line = f.readline()

            if state == FIND_HEADER:
                if search("STRATEGY SUMMARY", line):
                    state = SKIP_HEADER
            elif state == SKIP_HEADER:
                    state = READ_DATA
            elif state == READ_DATA:
                if search("===================", line):
                    state = FIND_HEADER
                else:
                    items = line.split("|")
                    strategy = items[1].strip()
                    if not (strategy in stratParams.keys()):
                        stratParams[strategy] = {"profit":[], "winPct":[], "draw":[]}

                    # print("items: ", items)
                    profitPct = float(items[6].strip())
                    tmp = items[8].split()
                    winPct = float(tmp[3].strip())
                    tmp = items[9].split()
                    draw = float(tmp[2].strip().replace("%", ""))

                    stratParams[strategy]["profit"].append(profitPct)
                    stratParams[strategy]["winPct"].append(winPct)
                    stratParams[strategy]["draw"].append(draw)

            else:
                print ("Invalid state: ", state)
                sys.exit()

    # print("stratParams: ", stratParams)

    if stratParams:
        # print("stratParams: ")
        # print(stratParams)

        #header
        divider = "-" * 90
        # print(divider)
        # print("| {:16s} | {:33s} | {:19s} | {:9s} |".format("Strategy".center(16, " "), "Profit".center(33, " "), "Win%".center(19, " "), "Rank".center(9, " ")))
        # print("| {:16s} | {:8s}{:8s}{:8s}{:8s} | {:8s}{:8s}{:8s}{:8s} |".format(" ".center(16, " "),
        #                                                                             "Min".center(8, " "), "Max".center(8, " "), "Ave".center(8, " "), "Med".center(8, " "),
        #                                                                             "Min".center(8, " "), "Max".center(8, " "), "Ave".center(8, " "), "Med".center(8, " ")
        # ))
        # print(divider)

        stratStats=[]
        # calculate stats for each strategy
        for strategy in stratParams:
            ptot = sum(stratParams[strategy]["profit"])
            pmin = min(stratParams[strategy]["profit"])
            pmax = max(stratParams[strategy]["profit"])
            pave = statistics.mean(stratParams[strategy]["profit"])
            pmed = statistics.median(stratParams[strategy]["profit"])
            wmin = min(stratParams[strategy]["winPct"])
            wmax = max(stratParams[strategy]["winPct"])
            wave = statistics.mean(stratParams[strategy]["winPct"])
            wmed = statistics.median(stratParams[strategy]["winPct"])
            dmin = min(stratParams[strategy]["draw"])
            dmax = max(stratParams[strategy]["draw"])
            dave = statistics.mean(stratParams[strategy]["draw"])
            dmed = statistics.median(stratParams[strategy]["draw"])
            empty = ""
            stratStats.append([strategy, ptot, pmin, pmax, pave, pmed, empty,
                               wmin, wmax, wave, wmed,  empty,
                               dmin, dmax, dave, dmed,  empty,
                               0.0, 0.0])
            # print("| {:<18s} | {:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f} | {:>8.2f}{:>8.2f}{:>8.2f}{:>8.2f} |".format(strategy,
            #                                                                         pmin, pmax, pave, pmed,
            #                                                                         wmin, wmax, wave, wmed
            #                                                                         ))
            # print(strategy, ": Profit min:", pmin, " max:", pmax, " Ave:", pave, " med:", pmed)
            # print(strategy, ": Win min:", wmin, " max:", wmax, " Ave:", wave, " med:", wmed)


        # create dataframe
        df = pandas.DataFrame(stratStats, columns=["Strategy",
                                                   "ptot", "pmin", "pmax", "pave", "pmed", "",
                                                   "wmin", "wmax", "wave", "wmed", "",
                                                   "dmin", "dmax", "dave", "dmed", "",
                                                   "Score", "Rank"])

        # calculate score. Weight profit higher, and median scores
        df["Score"] = 2.00 * ( df["ptot"].rank(pct=True) + df["pmin"].rank(pct=True) + df["pmax"].rank(pct=True) +
                               df["pave"].rank(pct=True) + 1.5*df["pmed"].rank(pct=True) ) + \
                      0.50 * ( df["wmin"].rank(pct=True) + df["wmax"].rank(pct=True) + df["wave"].rank(pct=True) +
                               1.5*df["wmed"].rank(pct=True) ) + \
                      0.25 * ( df["dmin"].rank(ascending=False, pct=True) + df["dmax"].rank(ascending=False, pct=True) +
                               df["dave"].rank(ascending=False, pct=True) + 1.5 * df["dmed"].rank(ascending=False, pct=True) )

        df["Rank"] = df["Score"].rank(ascending=False, method='min')

        pandas.set_option('display.precision', 2)
        print ("")
        print("                                         Profit                         ",
              "                 Win%                                 Drawdown                              Rank")
        hdrs=["Strategy", "PTot", "PMin", "PMax", "PAve", "PMed", "",
              "WMin", "WMax", "WAve", "WMed", ""
            , "DMin", "DMax", "DAve", "DMed", ""
            , "Score", "Rank"]
        print(tabulate(df, showindex="never", headers=hdrs, tablefmt='psql'))
        print ("")
        # print(tabulate(data, headers=["Name", "User ID", "Roll. No."]))
        # print(divider)


if __name__ == '__main__':
    main()