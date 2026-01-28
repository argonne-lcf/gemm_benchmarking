import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from collections.abc import Iterable
import sys
import scipy.stats as ss
from pathlib import Path
import string
import pprint
from sklearn.cluster import KMeans
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List
from itertools import chain

CLUSTER_NUMBER = SOW_target = {
    "cpu_DGEMM": 1,
    "cpu_SGEMM-FP32": 1,
    "cpu_HGEMM-FP16": 4,
    "cpu_HGEMM-BF16": 1,
    "cpu_IGEMM": 1,
    "gpu_DGEMM": 1,
    "gpu_SGEMM-FP32": 3,
    "gpu_SGEMM-TF32": 1,
    "gpu_HGEMM-FP16": 5,
    "gpu_HGEMM-BF16": 1,
    "gpu_IGEMM": 1,
}

name_remap = {
    "cpu_DGEMM": "cpu_GEMM-FP64",
    "cpu_SGEMM-FP32": "cpu_GEMM-FP32",
    "cpu_HGEMM-FP16": "cpu_GEMM-FP16",
    "cpu_HGEMM-BF16": "cpu_GEMM-BF16",
    "cpu_IGEMM": "cpu_GEMM-I8",
    "gpu_DGEMM": "gpu_GEMM-FP64",
    "gpu_SGEMM-FP32": "gpu_GEMM-FP32",
    "gpu_SGEMM-TF32": "gpu_GEMM-TF32",
    "gpu_HGEMM-FP16": "gpu_GEMM-FP16",
    "gpu_HGEMM-BF16": "gpu_GEMM-BF16",
    "gpu_IGEMM": "gpu_GEMM-I8",
}


def outline_thr(x):
    sample_size = len(x)
    counts, bins = np.histogram(x, bins=int(np.sqrt(sample_size)))
    # Always add a `0` to the end to avoid trailing loop
    counts = np.append(counts, 0)

    # Find the beginning and the size of the contiguous regions of non empty bins
    runs = []
    i_begin, size = None, 0
    for i, v in enumerate(counts):
        if v == 0:
            if i_begin != None:
                runs.append([i_begin, size])
            i_begin, size = None, 0
        else:
            if i_begin == None:
                i_begin = i
            size += 1

    max_begin, size = max(runs, key=lambda x: x[1])
    return bins[max_begin], bins[max_begin + size]


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def add_gaussian(ax, benchmark_array, counts, bin_edges):
    # Get avg and std for normal distribution
    mu, std = ss.norm.fit(benchmark_array)
    # counts, bin_edges = np.histogram(benchmark_array, bins=int(np.sqrt(sample_size)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # number of samples * size of each bin (i.e. reimann sum integration)
    scale = len(benchmark_array) * (bin_edges[1] - bin_edges[0])
    # pdf with mu and std but using the bin centers as the x
    gaussian = ss.norm.pdf(bin_centers, mu, std) * scale
    ax.plot(bin_centers, gaussian, "k", linewidth=2)
    rmse_val = rmse(gaussian, counts)
    print(f"   -Gaussian RMSE {rmse_val:.0f}")


def plot_hist_with_clustering(ax, data, n):
    # determine which K-Means cluster each point belongs to
    cluster_id = KMeans(n, random_state=0).fit_predict(data.reshape(-1, 1))

    # determine densities by cluster assignment and plot
    sample_size = len(data)
    bins = np.linspace(data.min(), data.max(), int(np.sqrt(sample_size)))

    for ii in np.unique(cluster_id):
        subset = data[cluster_id == ii]
        counts, _ = np.histogram(subset, bins=bins)
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html#matplotlib-pyplot-hist
        # Alternatively, plot pre-computed bins and counts using hist() by treating each bin as a single point with a weight equal to its count
        ax.hist(bins[:-1], bins, alpha=0.5, weights=counts, label=f"Cluster {ii}")
        ax.legend(fontsize=16)
        add_gaussian(ax, subset, counts, bins)


def plot_subfigs(benchmark_array, name, u, subfig, label, clustering=False):
    q1 = np.quantile(benchmark_array, 0.25)
    q3 = np.quantile(benchmark_array, 0.75)
    q2 = np.quantile(benchmark_array, 0.5)
    max_val = max(benchmark_array)
    min_val = min(benchmark_array)
    med_val = np.median(benchmark_array)
    iqr = q3 - q1
    percent_between_q3_q1 = iqr / med_val
    sample_size = len(benchmark_array)

    print(f"   -Min {min_val:.1f}")
    print(f"   -Q1 {q1:.1f}")
    print(f"   -Median {med_val:.1f}")
    print(f"   -Q3 {q3:.1f}")
    print(f"   -Max {max_val:.1f}")
    print(f"   -Sample Size {sample_size}")
    print(f"   -(Q3-Q1)/median {percent_between_q3_q1:.2%}")

    subfig.suptitle(
        f"{name_remap[name].upper()} (N: {sample_size}, Median: {med_val:.1f} {u})",
        fontsize=20,
    )
    axs = subfig.subplots(nrows=1, ncols=2, gridspec_kw={"width_ratios": [3, 1]})

    if clustering:
        plot_hist_with_clustering(axs[0], benchmark_array, CLUSTER_NUMBER[name])
    else:
        # to match what is done in the clustering
        bins = np.linspace(
            benchmark_array.min(), benchmark_array.max(), int(np.sqrt(sample_size))
        )
        counts, _ = np.histogram(benchmark_array, bins=bins)
        axs[0].hist(bins[:-1], bins, alpha=0.5, weights=counts)

    def add_box_letter(ax, text):
        ax.annotate(
            text,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(5, -5),
            textcoords="offset points",
            bbox=dict(boxstyle="square, pad=0.25", fc="none", ec="black"),
            ha="left",
            va="top",
            fontsize=20,
        )

    axs[0].set_xlabel(u, fontsize=18)
    hardware = name.split("_")[0].upper()
    axs[0].set_ylabel(f"# {hardware}s", fontsize=18)
    if clustering:
        axs[0].set_title(
            "Histogram (with K-means Clustering and fitted Gaussian)", fontsize=18
        )
    else:
        axs[0].set_title("Histogram", fontsize=18)
    add_box_letter(axs[0], f"{label}-h)")

    axs[1].boxplot(benchmark_array)
    axs[1].set_title(f"Boxplot (IQR/median: {percent_between_q3_q1:.2%})", fontsize=18)
    axs[1].set_ylabel(u, fontsize=18)
    add_box_letter(axs[1], f"{label}-b)")


def parse_file_our_format(file_path) -> Iterable:
    gemm_type = file_path.stem.split(".")[0]
    with file_path.open() as f:
        for line in f:
            # 19787.7,GFlop/s,gpu0,x4102c5s5b0n0
            measurement, unit, device_id, hostname = line.strip().split(",")

            measurement = float(measurement)
            if unit == "GFlop/s" and device_id.startswith("gpu"):
                unit = "TFLOP/s"
                measurement /= 1e3

            if device_id.startswith("gpu"):
                gemm_type_ = f"gpu_{gemm_type}"
            elif device_id.startswith("cpu"):
                gemm_type_ = f"cpu_{gemm_type}"
            else:
                raise (f"Fail to parse #{line}")
            yield ((gemm_type_, unit), (float(measurement), hostname))


def parse_file_reframe(file_path):
    with file_path.open() as f:
        for line in f:
            # Skip first line
            if "stagedir" in line:
                continue
            token = line.strip().split("|")
            # ...|cpu1_IGEMM=2954.48|ref=3000 (l=-0.1, u=null)|GFLOPS
            k, measurement = token[-3].split("=")

            device_id, gemm_type = k.split("_")
            measurement = float(measurement)

            unit = token[-1]

            # |FFTTest %$nodes=x4118c7s5b0n0 /4ad10f8c @aurora:compute+PrgEnv-intel
            if "$nodes" in token[3]:
                hostname = token[3].split()[1].split("=")[1]
                if "," in hostname:
                    raise ("Multiple hostname not supported")
            else:
                hostname = "unknown"

            if unit == "GFLOPS" and device_id.startswith("gpu"):
                unit = "TFLOP/s"
                measurement /= 1e3

            if device_id.startswith("gpu"):
                gemm_type_ = f"gpu_{gemm_type}"
            elif device_id.startswith("cpu"):
                gemm_type_ = f"cpu_{gemm_type}"
            else:
                raise (f"Fail to parse #{line}")

            yield ((gemm_type_, unit), (measurement, hostname))


# Ugly, they are Ndarray in reality...
@dataclass
class Point:
    flops: List[float] = field(default_factory=list)
    hostnames: List = field(default_factory=list)


def parse(path, use_directory):  # -> Return Dict [ name_test, unit ] = Point
    benchmarks_tests_results = defaultdict(Point)

    # 1. Iterator for all type
    if use_directory:
        # 1. Map the parser to the files
        # 2. Chain flattens the resulting iterators into one stream
        it = chain.from_iterable(map(parse_file_our_format, sorted(path.glob("*.txt"))))
    else:
        it = parse_file_reframe(path)

    # 2. Collect data (List append is actually optimal for unknown lengths)
    for k, (measurement, hostname) in it:
        benchmarks_tests_results[k].flops.append(measurement)
        benchmarks_tests_results[k].hostnames.append(hostname)

    # 3. Batch convert to numpy one
    for v in benchmarks_tests_results.values():
        v.flops = np.array(v.flops, dtype=float)
        v.hostnames = np.array(v.hostnames, dtype=object)

    return benchmarks_tests_results


def remove_outlier(d):
    print("# Removing Outliners")
    benchmarks_tests_results = defaultdict(Point)
    tests_failure = defaultdict(set)

    for k, point in d.items():
        name, unit = k
        min_thr, max_thr = outline_thr(point.flops)

        # 1. Create a boolean mask
        mask = (point.flops >= min_thr) & (point.flops < max_thr)

        # 2. Apply mask to both arrays
        valid_flops = point.flops[mask]
        valid_hostnames = point.hostnames[mask]

        # 3. Identify outliers using the inverse mask (~)
        outlier_hostnames = point.hostnames[~mask]

        # Store results
        benchmarks_tests_results[k] = Point(
            flops=valid_flops, hostnames=valid_hostnames
        )
        count = outlier_hostnames.size

        if count > 0:
            tests_failure[name].update(outlier_hostnames)
            count = outlier_hostnames.size
            print(f"  - {name}: {count} ∉ [{min_thr:.1f}, {max_thr:.1f}) {unit}")

    # Create unique keys
    hostname_failures = defaultdict(list)
    for name, hostnames in tests_failure.items():
        for hostname in hostnames:
            hostname_failures[hostname].append(name)
    # Now count oh many keys
    exclusif_count_per_types = Counter(tuple(v) for v in hostname_failures.values())
    print("## Hostname Outliner Removed Exclusif per Failure Group:")
    pprint.pprint(exclusif_count_per_types)

    return benchmarks_tests_results


def plot(output_name, benchmarks_results, clustering=False):
    num_plots = len(benchmarks_results)

    plt.rc("xtick", labelsize=18)
    plt.rc("ytick", labelsize=18)

    fig = plt.figure(layout="constrained", figsize=(20, num_plots * 4))

    subfigs_initial = fig.subfigures(nrows=num_plots, ncols=1)
    if not isinstance(subfigs_initial, Iterable):
        subfigs = np.array([subfigs_initial])
    else:
        subfigs = subfigs_initial

    it = zip(string.ascii_lowercase, subfigs, benchmarks_results.items())
    for label, subfig, ((name, unit), v) in it:
        print(f"## {name} ({unit})")
        plot_subfigs(v.flops, name, unit, subfig, label, clustering)
    plt.savefig(output_name)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEMM Miner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", type=Path, help="Path to the input file")
    group.add_argument(
        "-d", "--directory", type=Path, help="Path to the directory with input files"
    )
    parser.add_argument("-o", "--output", type=Path, help="Path to the output file")

    parser.add_argument(
        "--no-post-process",
        action="store_true",
        help="No outlier removal, no clustering",
    )

    parser.add_argument(
        "--print-outlier",
        action="store_true",
        help="No outlier removal, no clustering",
    )

    args = parser.parse_args()

    path = args.directory if args.directory else args.file
    use_directory = True if args.directory else False

    output_name = args.output if args.output else path.stem + ".png"
    print(f"Graph will be saved in `{output_name}`")

    d = parse(path, use_directory)
    if args.no_post_process:
        d = remove_outlier(d)
    print("# Plot and Statistic")

    plot(output_name, d, args.no_post_process)
