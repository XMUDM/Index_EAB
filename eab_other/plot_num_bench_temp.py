# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: plot_other_bench_temp
# @Author: Wei Zhou
# @Time: 2023/9/14 15:35

import os
import re
import json
import pickle

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("/data/wz/index/code_utils")

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


def merge_work_bench():
    res_dir = "/data/wz/index/index_eab/eab_olap/bench_result/tpch/work_num"
    # res_dir = "/data/wz/index/index_eab/eab_olap/bench_result/job/work_num"

    datas = {"utility": dict(), "time": dict()}
    for file in os.listdir(res_dir):
        if "num5" not in file:
            continue

        with open(f"{res_dir}/{file}", "r") as rf:
            data = json.load(rf)

        for algo in heu_algos + list(rl_agents.keys()):
            if algo not in datas["utility"].keys():
                datas["utility"][algo] = list()
                datas["time"][algo] = list()

            if algo not in data[0].keys():
                datas["utility"][algo].extend([0])
                datas["time"][algo].extend([0])
            else:
                datas["utility"][algo].extend(
                    [1 - item[algo]["total_ind_cost"] / item[algo]["total_no_cost"] for item in data])
                datas["time"][algo].extend([item[algo]["sel_info"]["time_duration"] for item in data])

    return datas


def pre_work_bench(datas):
    datas_pre = list()
    for no, typ in enumerate(datas.keys()):
        part = list()
        for algo in datas[typ].keys():
            if typ == "utility":
                part.append({"query_data": [100 * np.mean(datas[typ][algo])],
                             "yerr_data": [np.std(datas[typ][algo])]})
            else:
                # part.append({"query_data": [np.mean(item) for item in datas[typ][algo]],
                #              "yerr_data": [np.std(item) for item in datas[typ][algo]]})
                part.append({"query_data": [np.mean(datas[typ][algo])],
                             "yerr_data": [np.std(datas[typ][algo])]})
        datas_pre.append(part)

    return datas_pre


def ebar_bench_plot(save_path=None, save_conf={"format": "pdf", "bbox_inches": "tight"}):
    # 1. Get the data. True or
    if True or save_path is None or not os.path.exists(save_path.replace(".pdf", ".json")):
        data = merge_work_bench()
        data_save = save_path.replace(".pdf", ".json")
        with open(data_save, "w") as wf:
            json.dump(data, wf, indent=2)
    else:
        data_load = save_path.replace(".pdf", ".json")
        with open(data_load, "r") as rf:
            data = json.load(rf)

    data = pre_work_bench(data)

    # 2. Define the figure.
    nrows, ncols = 1, 2
    figsize = (20, 30 // 3)
    figsize = (27, 30 // 1.5)
    figsize = (18, 7)
    p_fig = ErrorBarPlot(nrows, ncols, figsize)
    # p_fig.fig.subplots_adjust(wspace=3.92, hspace=0.22)
    # p_fig.fig.subplots_adjust(hspace=0.08)
    p_fig.fig.subplots_adjust(wspace=0.35)

    # 3. Draw the subplot.
    groups = [algo for algo in heu_algos if algo != "relaxation"]
    labels = [""]

    groups = ["TPC-H", "TPC-H Skew"]
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
               "--", "|", "//"]

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
                "bbox_to_anchor": (1.25, 1.55),
                "prop": {"size": 30, "weight": 510}}

    # 3.2 set X properties.
    xlims = None
    xticks = None
    xticklabels_conf = {"labels": groups, "rotation": 15, "fontdict": {"fontsize": 13, "fontweight": "normal"}}

    xlabels = None
    xlabel_conf = None  # {"xlabel": xlabels, "fontdict": {"fontsize": "large", "fontweight": "bold"}}

    # 3.3 set Y properties.
    ylims = None
    yticks = None
    yticklabels_conf = None

    ylabel = "Relative Cost Reduction (%)"
    ylabel_conf = {"ylabel": ylabel, "fontdict": {"fontsize": 30, "fontweight": "bold"}}

    xticks = None
    xticklabels_conf = {"labels": groups, "fontdict": {"fontsize": 30, "fontweight": "bold"}}

    ylims = 0., 100.

    ax = p_fig.fig.add_subplot(p_fig.gs[0])
    ax.set_facecolor("#F3F3F3")

    xticklabels_conf = {"labels": groups,  # ["" for _ in range(len(groups))],
                        "fontdict": {"fontsize": 30, "fontweight": "bold"}}

    labels = list(heu_algo_alias.values()) + list(rl_agents_alias.values())
    p_fig.ebar_sub_plot(data[0], ax,
                        groups, labels, gap, width,
                        bar_conf=bar_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                        xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf, xlabel_conf=xlabel_conf,
                        ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf, ylabel_conf=ylabel_conf)

    ax = p_fig.fig.add_subplot(p_fig.gs[1])
    ax.set_facecolor("#F3F3F3")

    xticklabels_conf = {"labels": groups,
                        "fontdict": {"fontsize": 30, "fontweight": "bold"}}
    ylims = None
    ylabel = "Time Duration (s)"
    ylabel_conf = {"ylabel": ylabel, "fontdict": {"fontsize": 30, "fontweight": "bold"}}

    leg_conf = None

    labels = list(heu_algo_alias.values()) + list(rl_agents_alias.values())
    p_fig.ebar_sub_plot(data[1], ax,
                        groups, labels, gap, width,
                        bar_conf=bar_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                        xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf, xlabel_conf=xlabel_conf,
                        ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf, ylabel_conf=ylabel_conf)

    ax.set_yscale("log")

    if save_path is not None:
        plt.savefig(save_path, **save_conf)

    plt.show()


if __name__ == "__main__":
    save_path = None

    # save_path = f"/data/wz/index/index_eab/eab_olap/bench_result/multi_bench_num_bar.pdf"
    # ebar_bench_plot(save_path=save_path)
