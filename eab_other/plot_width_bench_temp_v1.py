# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: plot_width_bench_temp
# @Author: Wei Zhou
# @Time: 2023/9/29 11:11

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

storage_alias = ["100MB", "300MB", "500MB", "700MB", "900MB"]

widths = [i + 1 for i in range(4)]
widths_alias = [f"$W$ = {i + 1}" for i in range(4)]


# Line: constraint parameters value

def merge_constraint_data():
    sto_data = {"utility": dict(), "time": dict()}
    for sto_id in tqdm(["sto100", "sto300", "sto700", "sto900"]):
        # heuristic-based
        data_load = f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_sto_simple/" \
                    f"tpch_1gb_template_18_multi_work_{sto_id}_index_simple.json"
        with open(data_load, "r") as rf:
            data = json.load(rf)

        for algo in heu_algos:
            if algo not in sto_data["utility"].keys():
                sto_data["utility"][algo] = list()
                sto_data["time"][algo] = list()

            sto_data["utility"][algo].append(
                [1 - item[algo]["total_ind_cost"] / item[algo]["total_no_cost"] for item in data])
            sto_data["time"][algo].append([item[algo]["sel_info"]["time_duration"] for item in data])

        # rl-based
        for algo in rl_agents.keys():
            data_load = f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_sto_simple/" \
                        f"tpch_1gb_template_18_multi_work_{sto_id}_index_{algo}_simple.json"
            with open(data_load, "r") as rf:
                data = json.load(rf)

            if algo not in sto_data["utility"].keys():
                sto_data["utility"][algo] = list()
                sto_data["time"][algo] = list()

            u, t = list(), list()
            for item in data:
                u.extend([1 - item[ins]["total_ind_cost"] / item[ins]["total_no_cost"] for ins in item.keys()])
                t.extend([item[ins]["sel_info"]["time_duration"] for ins in item.keys()])

            sto_data["utility"][algo].append(u)
            sto_data["time"][algo].append(t)

    return sto_data


def pre_constraint_data(datas):
    data_loads = [
        "/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_level/tpch_1gb_multi_work_bench_index_utility_ebar.json",
        "/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_level/tpch_1gb_multi_work_bench_time_duration_ebar.json"]
    for no, typ in enumerate(["utility", "time"]):
        with open(data_loads[no], "r") as rf:
            data = json.load(rf)

        for algo in datas[typ].keys():
            datas[typ][algo] = datas[typ][algo][:2] + [data[algo]] + datas[typ][algo][2:]

    datas_pre = list()
    for no, typ in enumerate(datas.keys()):
        part = list()
        for algo in datas[typ].keys():
            if no == 0:
                part.append([100 * np.mean(item) for item in datas[typ][algo]])
            else:
                part.append([np.mean(item) for item in datas[typ][algo]])
        datas_pre.append(part)

    return datas_pre


# Line: maximum index width

def merge_work_bench_single(bench, res_id, data_id):
    data = dict()
    for algo in heu_algos:
        if algo not in data.keys():
            data[algo] = list()

        for width in widths:
            data_load = f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_v1_simple/" \
                        f"{bench_id[bench]}_multi_{res_id}{width}_index_simple.json"
            with open(data_load, "r") as rf:
                dat = json.load(rf)

            if data_id == "index_utility":
                temp = [1 - d[algo]["total_ind_cost"] / d[algo]["total_no_cost"] for d in dat]
            elif data_id == "time_duration":
                temp = [d[algo]["sel_info"]["time_duration"] for d in dat]

            data[algo].append(temp)

    for algo in rl_agents.keys():
        if algo not in data.keys():
            data[algo] = list()

        data_load = f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_v1_simple/" \
                    f"{bench_id[bench]}_multi_{res_id}_index_{algo}_simple.json"
        with open(data_load, "r") as rf:
            dat = json.load(rf)

        for width in widths:
            it = [key for key in dat[0].keys() if "sto" in key and f"width{width}" in key]

            temp = list()
            for d in dat:
                if data_id == "index_utility":
                    temp.extend([1 - d[t]["total_ind_cost"] / d[t]["total_no_cost"] for t in it])
                elif data_id == "time_duration":
                    temp.extend([d[t]["sel_info"]["time_duration"] for t in it])

            data[algo].append(temp)

    return data


def merge_work_bench(res_id, data_id, benchmarks):
    datas, datas_format = list(), list()
    for bench in benchmarks:
        data = merge_work_bench_single(bench, res_id, data_id)
        datas.append(data)

        temp = list()
        for algo in heu_algos + list(rl_agents.keys()):
            if data_id == "index_utility":
                # temp.append({"query_data": [100 * np.mean(dat) for dat in data[algo]],
                #              "yerr_data": [np.std(100 * np.array(dat)) for dat in data[algo]]})
                temp.append([100 * np.mean(dat) for dat in data[algo]])
            elif data_id == "time_duration":
                # temp.append({"query_data": [np.mean(dat) for dat in data[algo]],
                #              "yerr_data": [np.std(dat) for dat in data[algo]]})
                temp.append([np.mean(dat) for dat in data[algo]])

        datas_format.append(temp)

    return datas_format


def ebar_bench_width_plot(res_id, benchmarks, save_path=None,
                          save_conf={"format": "pdf", "bbox_inches": "tight"}):
    # 1. Get the data.
    datas_constraint = merge_constraint_data()
    datas_constraint = pre_constraint_data(datas_constraint)

    data_id = "index_utility"
    data_u = merge_work_bench(res_id, data_id, benchmarks)
    # data_save = save_path.replace(".pdf", f"_{data_id}.json")
    # with open(data_save, "w") as wf:
    #     json.dump(data_u, wf, indent=2)

    data_id = "time_duration"
    data_t = merge_work_bench(res_id, data_id, benchmarks)
    # data_save = save_path.replace(".pdf", f"_{data_id}.json")
    # # data_load = save_path.replace(".pdf", ".json")
    # with open(data_save, "w") as wf:
    #     json.dump(data_t, wf, indent=2)

    data = data_u + data_t
    data = [datas_constraint[0]] + data_u + [datas_constraint[1]] + data_t

    # 2. Define the figure.
    nrows, ncols = 2, 2
    # figsize = (27, 30 // 3)
    figsize = (27, 30 // 1.5)
    p_fig = FigurePlot(nrows, ncols, figsize)
    # p_fig.fig.subplots_adjust(wspace=3.92, hspace=0.22)
    p_fig.fig.subplots_adjust(hspace=0.08)

    # 3. Draw the subplot.
    groups = widths
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

    markers = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'][1:]
    styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']

    line_conf = [{"marker": markers[no], "markersize": 20,
                  "linestyle": styles[no], "linewidth": 5,
                  "color": colors[no], "zorder": 100} for no in range(len(labels))]

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

    xticklabels_conf = {"labels": groups, "fontdict": {"fontsize": 30, "fontweight": "bold"}}

    for no, dat in enumerate(data):
        ax = p_fig.fig.add_subplot(p_fig.gs[no])
        ax.set_facecolor("#F3F3F3")

        if no == 0 or no == 2:
            xticks = [i for i in range(len(storage_alias))]
            xticklabels_conf = {"labels": ["" for _ in xticks],
                                "fontdict": {"fontsize": 30, "fontweight": "bold"}}
        elif no == 1 or no == 3:
            xticks = [i for i in range(len(widths_alias))]
            xticklabels_conf = {"labels": ["" for _ in xticks],
                                "fontdict": {"fontsize": 30, "fontweight": "bold"}}

        if no == 0:
            ylabel = "Relative Cost Reduction (%)"
            ylabel_conf = {"ylabel": ylabel, "labelpad": 30,
                           "fontdict": {"fontsize": 30, "fontweight": "bold"}}

            leg_conf = {"loc": "upper center", "ncol": 5,
                        "bbox_to_anchor": (1.05, 1.3),
                        "prop": {"size": 33, "weight": 510}}

        elif no == 2:
            ylabel = "Time Duration (s)"
            ylabel_conf = {"ylabel": ylabel,
                           "fontdict": {"fontsize": 30, "fontweight": "bold"}}

            leg_conf = None

        else:
            ylabel_conf = None
            leg_conf = None

        if no == 2:
            xlabel_conf = {"xlabel": "(a) Storage Budget", "labelpad": 30,
                           "fontdict": {"fontsize": 35, "fontweight": "bold"}}
            xticklabels_conf = {"labels": storage_alias,
                                "fontdict": {"fontsize": 30, "fontweight": "bold"}}
        elif no == 3:
            xlabel_conf = {"xlabel": "(b) Maximum Index Width", "labelpad": 30,
                           "fontdict": {"fontsize": 35, "fontweight": "bold"}}
            xticklabels_conf = {"labels": widths_alias,
                                "fontdict": {"fontsize": 30, "fontweight": "bold"}}
        else:
            xlabel_conf = None

        labels = list(heu_algo_alias.values()) + list(rl_agents_alias.values())

        p_fig.line_sub_plot(dat, ax, labels, is_smooth=False,
                            line_conf=line_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                            xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf, xlabel_conf=xlabel_conf,
                            ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf, ylabel_conf=ylabel_conf)

        # p_fig.ebar_sub_plot(dat, ax,
        #                     groups, labels, gap, width,
        #                     bar_conf=bar_conf, tick_conf=tick_conf, leg_conf=leg_conf,
        #                     xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf, xlabel_conf=xlabel_conf,
        #                     ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf, ylabel_conf=ylabel_conf)

        if no == 2 or no == 3:
            ax.set_yscale("log")

    if save_path is not None:
        plt.savefig(save_path, **save_conf)

    plt.show()


if __name__ == "__main__":
    save_path = None

    res_id = "work_width"
    benchmarks = ["tpch"]

    # save_path = f"/data1/wz/index/index_eab/eab_olap/bench_result/multi_bench_{res_id}_sto_line.pdf"
    ebar_bench_width_plot(res_id, benchmarks, save_path=save_path)
