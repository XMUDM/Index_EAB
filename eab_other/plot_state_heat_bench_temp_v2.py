# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: plot_state_heat_bench_temp
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

mask = {"swirl": ["work_embed", "query_cost", "query_freq", "storage_budget", "storage_cons", "init_cost", "curr_cost"],
        "drlinda": ["work_matrix", "access_vector"],
        "dqn": ["work_embed", "query_cost", "query_freq", "storage_budget", "storage_cons", "init_cost", "curr_cost"]}
mask_alias = {"swirl": ["1. Workload Embedding", "2. Query Cost", "3. Query Frequency", "4. Storage Budget",
                        "5. Storage Consumption", "6. Initial Total Cost", "7. Current Total Cost"],
              "drlinda": ["1. Workload Matrix", "2. Column Access Actor"],
              "dqn": ["1. Workload Embedding", "2. Query Cost", "3. Query Frequency", "4. Storage Budget",
                      "5. Storage Consumption", "6. Initial Total Cost", "7. Current Total Cost"]}

mask_title = ["(a) SWIRL: 1. Workload Embedding; 2. Query Cost; 3. Query Frequency; 4. Storage Budget; \n "
              "5. Storage Consumption; 6. Initial Total Cost; 7. Current Total Cost",
              "(b) DRLindex", "(c) DQN"]
mask_title = ["SWIRL", "DRLindex", "DQN"]

text = ["plan", "sql"]
red_method = ["pca", "lsi", "doc"]

text_alias = ["Query Plan", "SQL Text"]
red_method_alias = ["PCA", "LSI", "Doc2Vec"]


def merge_work_bench_single_mask(bench, res_id, data_id):
    res_dir = f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_simple"

    data = dict()
    for algo in rl_agents.keys():  # ["dqn"]
        for ma in mask[algo]:
            if algo not in data.keys():
                data[algo] = list()

            data_load = f"{res_dir}/{bench_id[bench]}_multi_{res_id}_index_{ma}_v2_{algo}_simple.json"
            with open(data_load, "r") as rf:
                if algo == "dqn":
                    dat = json.load(rf)[:60]
                else:
                    dat = json.load(rf)[:10]

            temp = list()
            for da in dat:
                if data_id == "index_utility":
                    if algo == "dqn":
                        temp.extend([d["total_ind_cost"] / d["total_no_cost"] for k, d in da.items()
                                     if "666" not in k and "333" not in k])
                    else:
                        temp.extend([d["total_ind_cost"] / d["total_no_cost"] for k, d in da.items()])
                elif data_id == "time_duration":
                    data[algo].extend([d["sel_info"]["time_duration"] for d in da.values()])

            data[algo].append(temp[:30])

    return data


def merge_work_bench_mask(res_id, data_id, benchmarks):
    datas = list()
    for bench in benchmarks:
        datas.append(merge_work_bench_single_mask(bench, res_id, data_id))

    datas_pre = list()
    for algo in list(rl_agents.keys()):
        for data in datas:
            datas_pre.append(np.array(data[algo]))
            # datas_pre.append(100 * np.array(data[algo]).T)

    return datas_pre


def merge_work_bench_single_state(bench, res_id, data_id):
    res_dir = f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_simple"

    data = dict()
    algo = "swirl"

    for red in red_method:
        if red not in data.keys():
            data[red] = list()

        for t in text:
            data_load = f"{res_dir}/{bench_id[bench]}_multi_{res_id}_index_{t}_{algo}_simple.json"
            with open(data_load, "r") as rf:
                dat = json.load(rf)

            it = [k for k in dat[0].keys() if f"{t}_{red}" in k]

            temp = list()
            for da in dat:
                if data_id == "index_utility":
                    temp.extend([d["total_ind_cost"] / d["total_no_cost"] for k, d in da.items() if k in it])
                elif data_id == "time_duration":
                    data[algo].extend([d["sel_info"]["time_duration"] for k, d in da.items() if k in it])

            data[red].append(temp)

    return data


def merge_work_bench_state(res_id, data_id, benchmarks):
    datas, datas_format = list(), list()
    for bench in benchmarks:
        data = merge_work_bench_single_state(bench, res_id, data_id)
        datas.append(data)

        for red in red_method:
            temp = list()
            if data_id == "index_utility":
                temp.append({"query_data": [100 * np.mean(dat) for dat in data[red]],
                             "yerr_data": [np.std(100 * np.array(dat)) for dat in data[red]]})
                # temp.append([100 * np.mean(dat) for dat in data[red]])
            elif data_id == "time_duration":
                # temp.append({"query_data": [np.mean(dat) for dat in data[red]],
                #              "yerr_data": [np.std(dat) for dat in data[red]]})
                temp.append([np.mean(dat) for dat in data[red]])

            datas_format.extend(temp)

    return datas_format


def ebar_bench_freq_plot(res_id, data_id, benchmarks, save_path=None,
                         save_conf={"format": "pdf", "bbox_inches": "tight"}):
    # 1. Get the data.
    res_id = "work_mask"
    data_id = "index_utility"
    data_mask = merge_work_bench_mask(res_id, data_id, benchmarks)
    # data_save = save_path.replace(".pdf", f"_{data_id}.json")
    # data_load = save_path.replace(".pdf", ".json")
    # with open(data_save, "w") as wf:
    #     json.dump(data, wf, indent=2)

    res_id = "work_state"
    data_id = "index_utility"
    data_state = merge_work_bench_state(res_id, data_id, benchmarks)

    # 2. Define the figure.
    nrows, ncols = 3, 3
    figsize = (20, 30 // 3)
    figsize = (27, 30 // 1.5)
    figsize = (27, 30 // 2.7)

    # nrows, ncols = 1, 3
    # figsize = (30 // 1, 27)

    p_fig = FigurePlot(nrows, ncols, figsize)
    # p_fig.fig.subplots_adjust(wspace=3.92, hspace=0.22)
    # p_fig.fig.subplots_adjust(wspace=.005, hspace=0.03)

    # 3. Draw the subplot.
    groups = text_alias
    labels = red_method_alias

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
              ][2:]

    hatches = ["", "/", "-",
               "x", "||", "\\",
               "--", "|", "//",
               "", "/", "-", ]

    gap, width = 0.25, 0.2

    # "#F0F0F0", "#F1F1F1", "#2A4458"
    heat_conf = {"cmap": "Blues"}  # viridis, Blues

    bar_conf = [{"color": colors[no], "hatch": hatches[no], "edgecolor": "#F0F0F0",
                 "error_kw": dict(lw=2, capsize=3, capthick=1.5)} for no in range(len(labels))]

    tick_conf = {"labelsize": 25, "pad": 20}
    leg_conf = {"loc": "upper center", "ncol": 9,
                "bbox_to_anchor": (.5, 1.15),
                "prop": {"size": 23, "weight": "normal"}}
    leg_conf = None

    # leg_conf = {"loc": "upper center", "ncol": 5,
    #             "bbox_to_anchor": (.5, 1.25),
    #             "prop": {"size": 30, "weight": 510}}

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

    for no, dat in enumerate(data_mask):
        ax = p_fig.fig.add_subplot(p_fig.gs[no, :-1])
        ax.set_facecolor("#F3F3F3")

        if no == 0:
            ylabel = "Relative Cost Reduction (%)"
            ylabel_conf = {"ylabel": ylabel,
                           "fontdict": {"fontsize": 30, "fontweight": "bold"}}
            ylabel_conf = None

        else:
            ylabel_conf = None

        # xticks = range(len(mask[list(rl_agents_alias.keys())[no]]))
        # xticklabels_conf = {"labels": mask[list(rl_agents_alias.keys())[no]], "rotation": 15,
        #                     "fontdict": {"fontsize": 20, "fontweight": "normal"}}

        xticklabels_conf = {"labels": ["" for _ in groups], "rotation": 15,
                            "fontdict": {"fontsize": 20, "fontweight": "normal"}}

        xlabel_conf = {"xlabel": mask_title[no],
                       "fontdict": {"fontsize": 30, "fontweight": "bold"}}

        yticks = range(len(mask[list(rl_agents_alias.keys())[no]]))
        yticklabels_conf = {"labels": mask_alias[list(rl_agents_alias.keys())[no]],
                            # "rotation": 15,
                            "fontdict": {"fontsize": 20, "fontweight": "bold"}}

        # yticklabels_conf = {"labels": ["" for _ in groups],
        #                     "fontdict": {"fontsize": 13, "fontweight": "normal"}}

        p_fig.heatmap_sub_plot_1d(dat, ax,
                                  heat_conf=heat_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                                  xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf,
                                  xlabel_conf=xlabel_conf,
                                  ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf,
                                  ylabel_conf=ylabel_conf)

    for no, dat in enumerate([data_state]):
        ax = p_fig.fig.add_subplot(p_fig.gs[:, -1])
        ax.set_facecolor("#F3F3F3")

        if no == 0:
            ylabel = "Relative Cost Reduction (%)"
            ylabel_conf = {"ylabel": ylabel,
                           "fontdict": {"fontsize": 30, "fontweight": "bold"}}
            # ylabel_conf = None

        else:
            ylabel_conf = None

        # xticks = range(len(mask[list(rl_agents_alias.keys())[no]]))
        # xticklabels_conf = {"labels": mask[list(rl_agents_alias.keys())[no]], "rotation": 15,
        #                     "fontdict": {"fontsize": 20, "fontweight": "normal"}}

        xticklabels_conf = {"labels": [g for g in groups],
                            "fontdict": {"fontsize": 30, "fontweight": "bold"}}

        xlabel_conf = None
        yticks = None
        yticklabels_conf = None
        # yticklabels_conf = {"labels": ["" for _ in groups],
        #                     "fontdict": {"fontsize": 13, "fontweight": "normal"}}

        ylims = 0, 100
        # ax.yaxis.tick_right()

        leg_conf = {"loc": "upper center", "ncol": 2,
                    "bbox_to_anchor": (.5, 1.),
                    "prop": {"size": 25, "weight": 510}}

        p_fig.ebar_sub_plot(dat, ax, groups, labels, gap, width,
                            bar_conf=bar_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                            xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf,
                            xlabel_conf=xlabel_conf,
                            ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf,
                            ylabel_conf=ylabel_conf)

    # if save_path is not None:
    #     plt.savefig(save_path, **save_conf)

    plt.show()


if __name__ == "__main__":
    save_path = None

    # Bar: varying frequency
    # work_freq, work_num
    res_id = "work_mask"
    data_id = "index_utility"

    benchmarks = ["tpch"]
    # save_path = f"/data1/wz/index/index_eab/eab_olap/bench_result/multi_{res_id}_bench_ebar.pdf"
    ebar_bench_freq_plot(res_id, data_id, benchmarks, save_path=save_path)
