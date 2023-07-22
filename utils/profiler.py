# utility functions to help with profiling

# Usage:
#    import profiler
#
#    # to start tracing call:
#    profiler.start(10) # 10 is arbitrary but recommended
#
#    periodically take snapshots of memory usage:
#    profiler.snapshot()
#
#    when you want to display results:
#    profiler.display_stats()
#    profiler.compare()
#    profiler.print_trace()

import tracemalloc

# list to store memory snapshots
snaps = []

def start(num_frames: int):
    clear()
    tracemalloc.start(num_frames)

def stop():
    tracemalloc.stop()
    clear()

def clear():
    global snaps

    if tracemalloc.is_tracing():
        compare()
    tracemalloc.clear_traces()
    snaps = []

def snapshot():
    global snaps

    snaps.append(tracemalloc.take_snapshot())


def display_stats():
    global snaps

    stats = snaps[0].statistics('filename')
    print("\n*** top 5 stats grouped by filename ***")
    for s in stats[:5]:
        print(s)


def compare():
    global snaps

    first = snaps[0]
    for snapshot in snaps[1:]:
        stats = snapshot.compare_to(first, 'lineno')
        print("\n*** top 10 stats ***")
        for s in stats[:10]:
            print(s)


def print_trace():
    global snaps

    # pick the last saved snapshot, filter noise
    snapshot = snaps[-1].filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    largest = snapshot.statistics("traceback")[0]

    print(f"\n*** Trace for largest memory block - ({largest.count} blocks, {largest.size / 1024} Kb) ***")
    for l in largest.traceback.format():
        print(l)