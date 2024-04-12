# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: plot_state_bench_temp
# @Author: Wei Zhou
# @Time: 2023/9/22 20:23

import os
import re
import json
import pickle

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("/data1/wz/index/code_utils")

from result_plot.fig_plot_all import FigurePlot
from result_plot.fig_plot import ErrorBarPlot, LineRangePlot

benchmarks = ["TPC-H", "TPC-DS", "JOB", "TPC-H Skew", "DSB"]
bench_id = {"tpch": "tpch_1gb_template_18",
            "tpcds": "tpcds_1gb_template_79",
            "job": "job_template_33",
            "tpch_skew": "tpch_skew_1gb_template_18",
            "dsb": "dsb_1gb_template_53"}

bench_name = ["(a) TPC-H", "(b) TPC-DS", "(c) JOB", "(d) TPC-H Skew", "(e) DSB"]

heu_algos = ["extend", "db2advis", "relaxation", "anytime", "auto_admin", "drop"]
rl_agents = {"swirl": list(), "drlinda": list(), "dqn": list()}

heu_algo_alias = {"extend": "Extend", "db2advis": "DB2Advis", "relaxation": "Relaxation",
                  "anytime": "DTA", "auto_admin": "AutoAdmin", "drop": "Drop"}
rl_agents_alias = {"swirl": "SWIRL", "drlinda": "DRLindex", "dqn": "DQN"}


# Bar: varying frequency

def merge_work_bench_single(bench, res_id, data_id):
    res_dir = f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_simple"

    data = dict()
    for file in sorted(os.listdir(res_dir)):
        if "swirl" in file or "drlinda" in file or "dqn" in file:
            continue

        with open(f"{res_dir}/{file}", "r") as rf:
            dat = json.load(rf)

        for algo in dat[0].keys():
            if algo not in data.keys():
                data[algo] = list()

            if data_id == "index_utility":
                data[algo].extend([1 - d[algo]["total_ind_cost"] / d[algo]["total_no_cost"] for d in dat])
            elif data_id == "time_duration":
                data[algo].extend([d[algo]["sel_info"]["time_duration"] for d in dat])

    data_load = f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_simple/" \
                f"{bench_id[bench]}_multi_{res_id}_index" + "_{}_simple.json"
    for algo in rl_agents.keys():
        if algo not in data.keys():
            data[algo] = list()

        # if not os.path.exists(data_load.format(algo)):
        #     data[algo].extend([0])
        # else:
        with open(data_load.format(algo), "r") as rf:
            dat = json.load(rf)

        for da in dat:
            if data_id == "index_utility":
                data[algo].extend([1 - d["total_ind_cost"] / d["total_no_cost"] for d in da.values()])
            elif data_id == "time_duration":
                data[algo].extend([d["sel_info"]["time_duration"] for d in da.values()])

    return data


def merge_work_bench(res_id, data_id, benchmarks):
    # res_id = "work_freq"
    # data_id = "index_utility"

    datas = list()
    for bench in benchmarks:
        datas.append(merge_work_bench_single(bench, res_id, data_id))

    datas_pre = list()
    for algo in heu_algos + list(rl_agents.keys()):
        datas_pre.append({"query_data": [100 * np.mean(data[algo]) for data in datas],
                          "yerr_data": [np.std(data[algo]) for data in datas]})

    return datas_pre


def ebar_bench_freq_plot(res_id, data_id, benchmarks, save_path=None,
                         save_conf={"format": "pdf", "bbox_inches": "tight"}):
    # 1. Get the data.
    data = merge_work_bench(res_id, data_id, benchmarks)
    # if save_path is not None:
    #     data_save = save_path.replace(".pdf", f"_{data_id}.json")
    #     with open(data_save, "w") as wf:
    #         json.dump(data, wf, indent=2)

    # data_load = save_path.replace(".pdf", ".json")

    # 2. Define the figure.
    nrows, ncols = 2, 1
    figsize = (20, 30 // 3)
    figsize = (27, 30 // 1.5)
    p_fig = ErrorBarPlot(nrows, ncols, figsize)
    # p_fig.fig.subplots_adjust(wspace=3.92, hspace=0.22)
    p_fig.fig.subplots_adjust(hspace=0.08)

    # 3. Draw the subplot.
    groups = benchmarks
    labels = heu_algos + list(rl_agents.keys())

    # 3.1 set Bar properties.
    colors = [p_fig.color["green"], p_fig.color["blue"], p_fig.color["grey"],
              p_fig.color["orange"], p_fig.color["pink"], p_fig.color["red"],
              p_fig.color["purple"], p_fig.color["brown"], p_fig.color["yellow"]]

    colors = ["#2A4458", "#336485", "#3E86B5", "#95A77E", "#E5BE79",
              # "#11325D", "#365083",
              "#736B9D", "#B783AF", "#F5A673", "#ABC9C8",  # "#FCDB72",
              # "#404D5B", "#5492C7",
              "#E7E5DF",
              # "#99B86B", "#688A4C",
              # "#545969",
              "#A4757D",
              # "#E7987C", "#8B91B6", "#7771A4"
              ]

    hatches = ["", "/", "-",
               "x", "||", "\\",
               "--", "|", "//",
               "", "/", "-", ]

    gap, width = 0.25, 0.2

    # "#F0F0F0", "#F1F1F1", "#2A4458"
    bar_conf = [{"color": colors[no], "hatch": hatches[no], "edgecolor": "#F0F0F0",
                 "error_kw": dict(lw=2, capsize=3, capthick=1.5)} for no in range(len(labels))]
    tick_conf = {"labelsize": 25, "pad": 20}
    leg_conf = {"loc": "upper center", "ncol": 9,
                "bbox_to_anchor": (.5, 1.15),
                "prop": {"size": 23, "weight": "normal"}}
    # leg_conf = None

    leg_conf = {"loc": "upper center", "ncol": 5,
                "bbox_to_anchor": (.5, 1.25),
                "prop": {"size": 30, "weight": 510}}

    # 3.2 set X properties.
    xlims = None
    xticks = None
    xticklabels_conf = {"labels": groups, "rotation": 15, "fontdict": {"fontsize": 13, "fontweight": "normal"}}

    xlabels = None
    xlabel_conf = None  # {"xlabel": xlabels, "fontdict": {"fontsize": "large", "fontweight": "bold"}}

    # 3.3 set Y properties.
    # if data_id == "index_utility":
    #     ylims = (0., 60.)  # 0.0, 22.0
    # else:
    #     ylims = None
    ylims = None
    yticks = None
    yticklabels_conf = None

    ylabel = "Relative Cost Reduction (%)"
    ylabel_conf = {"ylabel": ylabel, "fontdict": {"fontsize": 30, "fontweight": "bold"}}

    # query_datas = {"query_data": [np.mean(100 * np.array(data[algo])) for algo in data.keys()],
    #                "yerr_data": [np.std(100 * np.array(data[algo])) for algo in data.keys()]}
    # groups = list(data.keys())

    xticks = None
    xticklabels_conf = {"labels": groups, "fontdict": {"fontsize": 30, "fontweight": "bold"}}

    ax = p_fig.fig.add_subplot(p_fig.gs[0])
    ax.set_facecolor("#F3F3F3")

    xticklabels_conf = {"labels": ["" for _ in range(len(groups))], "fontdict": {"fontsize": 25, "fontweight": "bold"}}

    labels = list(heu_algo_alias.values()) + list(rl_agents_alias.values())
    p_fig.ebar_sub_plot(data, ax,
                        groups, labels, gap, width,
                        bar_conf=bar_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                        xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf, xlabel_conf=xlabel_conf,
                        ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf, ylabel_conf=ylabel_conf)

    # ax.set_yscale("log")

    # if save_path is not None:
    #     plt.savefig(save_path, **save_conf)

    plt.show()


if __name__ == "__main__":
    save_path = None

    # Bar: varying frequency
    # work_freq, work_num
    res_id = "work_num"
    data_id = "index_utility"

    benchmarks = ["tpch_skew", "job"]  # "tpch",
    # save_path = f"/data1/wz/index/index_eab/eab_olap/bench_result/multi_{res_id}_bench_ebar.pdf"
    ebar_bench_freq_plot(res_id, data_id, benchmarks, save_path=save_path)
