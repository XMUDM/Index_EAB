# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: plot_factor_bench_temp
# @Author: Wei Zhou
# @Time: 2023/9/22 20:39

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
bench_name = ["(a) TPC-H", "(b) TPC-DS", "(c) JOB", "(d) TPC-H Skew", "(e) DSB"]

bench_id = {"tpch": "tpch_1gb_template_18",
            "tpcds": "tpcds_1gb_template_79",
            "job": "job_template_33",
            "tpch_skew": "tpch_skew_1gb_template_18",
            "dsb": "dsb_1gb_template_53"}

heu_algos = ["extend", "db2advis", "relaxation", "anytime", "auto_admin", "drop"]
rl_agents = {"swirl": list(), "drlinda": list(), "dqn": list()}  # , "mcts": list(), "bandits": list()

heu_algo_alias = {"extend": "Extend", "db2advis": "DB2Advis", "relaxation": "Relaxation",
                  "anytime": "DTA", "auto_admin": "AutoAdmin", "drop": "Drop"}
rl_agents_alias = {"swirl": "SWIRL", "drlinda": "DRLindex", "dqn": "DQN"}  # , "mcts": "MCTS", "bandits": "DBA Bandits"

factors = ["1_", "100_", "1k_"]
factors_alias = ["$X$ 1", "$X$ 100", "$X$ 1000"]


# Bar: varying frequency

def merge_work_bench_single(bench, res_id, data_id):
    data = dict()
    for algo in rl_agents.keys():
        if algo not in data.keys():
            data[algo] = list()

        data_load = f"/data/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_simple/" \
                    f"{bench_id[bench]}_multi_{res_id}_index_{algo}_simple.json"
        with open(data_load, "r") as rf:
            dat = json.load(rf)

        for fac in factors:
            temp = list()
            it = [key for key in dat[0].keys() if fac in key]

            for d in dat:
                if data_id == "index_utility":
                    temp.extend([1 - d[t]["total_ind_cost"] / d[t]["total_no_cost"] for t in it])
                elif data_id == "time_duration":
                    temp.extend([d[t]["sel_info"]["time_duration"] for t in it])

            data[algo].append(temp)

    for algo in rl_agents.keys():
        data_load = f"/data/wz/index/index_eab/eab_olap/bench_result/{bench}/" \
                    f"work_level_simple/{bench_id[bench]}_multi_work_index_{algo}_simple.json"
        with open(data_load, "r") as rf:
            dat = json.load(rf)

        it = [key for key in dat[0].keys() if "sto" in key and "unique" not in key]

        temp = list()
        for d in dat:
            if data_id == "index_utility":
                temp.extend([1 - d[t]["total_ind_cost"] / d[t]["total_no_cost"] for t in it])
            elif data_id == "time_duration":
                temp.extend([d[t]["sel_info"]["time_duration"] for t in it])

        if algo in ["swirl", "dqn"]:
            data[algo][0] = temp[:]
        else:
            data[algo][1] = temp[:]

    return data


def merge_work_bench(res_id, data_id, benchmarks):
    datas, datas_format = list(), list()
    for bench in benchmarks:
        data = merge_work_bench_single(bench, res_id, data_id)
        datas.append(data)

        temp = list()
        for algo in list(rl_agents.keys()):
            if data_id == "index_utility":
                temp.append({"query_data": [100 * np.mean(dat) for dat in data[algo]],
                             "yerr_data": [np.std(100 * np.array(dat)) for dat in data[algo]]})
            elif data_id == "time_duration":
                temp.append({"query_data": [np.mean(dat) for dat in data[algo]],
                             "yerr_data": [np.std(dat) for dat in data[algo]]})

        datas_format.append(temp)

    return datas_format


def ebar_bench_factor_plot(res_id, benchmarks, save_path=None,
                           save_conf={"format": "pdf", "bbox_inches": "tight"}):
    # 1. Get the data.
    data_id = "index_utility"
    data_u = merge_work_bench(res_id, data_id, benchmarks)
    data_save = save_path.replace(".pdf", f"_{data_id}.json")
    with open(data_save, "w") as wf:
        json.dump(data_u, wf, indent=2)

    data_id = "time_duration"
    data_t = merge_work_bench(res_id, data_id, benchmarks)
    data_save = save_path.replace(".pdf", f"_{data_id}.json")

    # data_load = save_path.replace(".pdf", ".json")
    with open(data_save, "w") as wf:
        json.dump(data_t, wf, indent=2)

    data = data_u + data_t

    # 2. Define the figure.
    nrows, ncols = 1, 2
    figsize = (27, 30 // 3)
    # figsize = (27, 30 // 1.5)
    p_fig = ErrorBarPlot(nrows, ncols, figsize)
    # p_fig.fig.subplots_adjust(wspace=3.92, hspace=0.22)
    p_fig.fig.subplots_adjust(hspace=0.08)

    # 3. Draw the subplot.
    groups = factors
    labels = list(rl_agents.keys())

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
              ][3:]

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
                "bbox_to_anchor": (.95, 1.15),
                "prop": {"size": 23, "weight": "normal"}}
    # leg_conf = None

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

    for no, dat in enumerate(data):
        ax = p_fig.fig.add_subplot(p_fig.gs[no])
        ax.set_facecolor("#F3F3F3")

        if no == 0:
            ylabel = "Relative Cost Reduction (%)"
            ylabel_conf = {"ylabel": ylabel, "fontdict": {"fontsize": 30, "fontweight": "bold"}}

            leg_conf = {"loc": "upper center", "ncol": 5,
                        "bbox_to_anchor": (1.05, 1.15),
                        "prop": {"size": 30, "weight": 510}}

        elif no == 1:
            ylabel = "Time Duration (s)"
            ylabel_conf = {"ylabel": ylabel, "fontdict": {"fontsize": 30, "fontweight": "bold"}}

            leg_conf = None

        xticklabels_conf = {"labels": factors_alias, "fontdict": {"fontsize": 25, "fontweight": "bold"}}

        labels = list(rl_agents_alias.values())
        p_fig.ebar_sub_plot(dat, ax,
                            groups, labels, gap, width,
                            bar_conf=bar_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                            xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf, xlabel_conf=xlabel_conf,
                            ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf, ylabel_conf=ylabel_conf)

        if no == 1:
            ax.set_yscale("log")

    if save_path is not None:
        plt.savefig(save_path, **save_conf)

    plt.show()


if __name__ == "__main__":
    save_path = None

    res_id = "work_factor"
    benchmarks = ["tpch"]

    save_path = f"/data/wz/index/index_eab/eab_olap/bench_result/multi_bench_{res_id}_bar.pdf"
    ebar_bench_factor_plot(res_id, benchmarks, save_path=save_path)
