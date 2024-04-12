# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: plot_mask_bench_temp
# @Author: Wei Zhou
# @Time: 2023/10/16 11:39

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

benchmarks = ["TPC-H", "TPC-DS", "JOB", "TPC-H Skew", "DSB"]
bench_id = {"tpch": "tpch_1gb_template_18",
            "tpcds": "tpcds_1gb_template_79",
            "job": "job_template_33",
            "tpch_skew": "tpch_skew_1gb_template_18",
            "dsb": "dsb_1gb_template_53"}

bench_name = ["(a) TPC-H", "(b) TPC-DS", "(c) JOB", "(d) TPC-H Skew", "(e) DSB"]

benchmarks_alias = {"tpch": "TPC-H", "tpcds": "TPC-DS", "job": "JOB", "tpch_skew": "TPC-H Skew", "dsb": "DSB"}

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
              "drlinda": ["1. Workload Matrix", "2. Column Access Vector"],
              "dqn": ["1. Workload Embedding", "2. Query Cost", "3. Query Frequency", "4. Storage Budget",
                      "5. Storage Consumption", "6. Initial Total Cost", "7. Current Total Cost"]}

mask_title = ["(a) SWIRL: 1. Workload Embedding; 2. Query Cost; 3. Query Frequency; 4. Storage Budget; \n "
              "5. Storage Consumption; 6. Initial Total Cost; 7. Current Total Cost",
              "(b) DRLindex", "(c) DQN"]
mask_title = ["SWIRL", "DRLindex", "DQN"]

rl_agents_alias = {"swirl": "SWIRL (②)", "drlinda": "DRLindex (②)", "dqn": "DQN (②)"}
mask_title = ["SWIRL (②)", "DRLindex (②)", "DQN (②)"]


def merge_work_bench_single_mask(bench, res_id, data_id):
    res_dir = f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_simple"

    data = dict()
    for algo in rl_agents.keys():  # ["dqn"]
        for ma in mask[algo]:
            if algo not in data.keys():
                data[algo] = list()

            if bench == "tpch":
                data_load = f"{res_dir}/{bench_id[bench]}_multi_{res_id}_index_{ma}_v2_{algo}_simple.json"
            elif bench == "tpcds":
                data_load = f"{res_dir}/{bench_id[bench]}_multi_{res_id}_index_{ma}_v3_{algo}_simple.json"

            with open(data_load, "r") as rf:
                if algo == "dqn":
                    dat = json.load(rf)[:60]
                else:
                    dat = json.load(rf)[:30]

            temp = list()
            for da in dat:
                if data_id == "index_utility":
                    if algo == "swirl":
                        temp.extend([d["total_ind_cost"] / d["total_no_cost"]
                                     for k, d in da.items() if "666" in k])
                    elif algo == "drlinda":
                        temp.extend([d["total_ind_cost"] / d["total_no_cost"]
                                     for k, d in da.items() if "999" in k])
                    if algo == "dqn":
                        temp.extend([d["total_ind_cost"] / d["total_no_cost"]
                                     for k, d in da.items() if "333" in k])

                    # if algo == "dqn":
                    #     temp.extend([d["total_ind_cost"] / d["total_no_cost"] for k, d in da.items()
                    #                  if "666" not in k and "333" not in k])
                    # else:
                    #     temp.extend([d["total_ind_cost"] / d["total_no_cost"] for k, d in da.items()])

                elif data_id == "time_duration":
                    data[algo].extend([d["sel_info"]["time_duration"] for d in da.values()])

            # data[algo].append(temp[:30])
            data[algo].append(temp[:22])

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


# Bar: workload permutation

def merge_work_bench_single(bench, res_id, data_id):
    res_dir = f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_simple"

    data = dict()
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
    for algo in list(rl_agents.keys()):
        datas_pre.append({"query_data": [100 * np.mean(data[algo]) for data in datas],
                          "yerr_data": [np.std(100 * np.array(data[algo])) for data in datas]})
        # datas_pre.append([100 * np.array(data[algo]) for data in datas])

    return datas_pre


def heat_bench_mask_plot(res_id, data_id, benchmarks, save_path=None,
                         save_conf={"format": "pdf", "bbox_inches": "tight"}):
    # 1. Get the data.
    benchmarks = ["tpch"]
    data_mask1 = merge_work_bench_mask(res_id, data_id, benchmarks)
    # data_save = save_path.replace(".pdf", f"_{data_id}.json")
    # data_load = save_path.replace(".pdf", ".json")
    # with open(data_save, "w") as wf:
    #     json.dump(data, wf, indent=2)

    # benchmarks = ["tpcds"]
    # data_mask2 = merge_work_bench_mask(res_id, data_id, benchmarks)

    # data_masks = [data_mask1] + [data_mask2]
    data_masks = [data_mask1]

    # 2. Define the figure.
    nrows, ncols = 3, 2
    figsize = (20, 30 // 3)
    figsize = (27, 30 // 1.5)
    figsize = (27, 30 // 2.7)

    # nrows, ncols = 1, 3
    # figsize = (30 // 1, 27)

    p_fig = FigurePlot(nrows, ncols, figsize)
    # p_fig.fig.subplots_adjust(wspace=3.92, hspace=0.22)
    p_fig.fig.subplots_adjust(wspace=.06, hspace=0.)  # 0.005

    # 3. Draw the subplot.
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

    tick_conf = {"labelsize": 25, "pad": 10}  # 20
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

    xlabels = None
    xlabel_conf = None  # {"xlabel": xlabels, "fontdict": {"fontsize": "large", "fontweight": "bold"}}
    xticklabels_conf = None

    # 3.3 set Y properties.
    # if data_id == "index_utility":
    #     ylims = (0., 60.)  # 0.0, 22.0
    # else:
    #     ylims = None
    ylims = None
    yticks = None
    yticklabels_conf = None

    ylabel = "Relative Cost Reduction (%)"
    ylabel_conf = None  # {"ylabel": ylabel, "fontdict": {"fontsize": 30, "fontweight": "bold"}}

    # Heatmap: Feature Importance

    heat_color = ["Greens", "Greys", "Blues"]
    for i, data_mask in enumerate(data_masks):
        for no, dat in enumerate(data_mask):
            heat_conf = {"cmap": heat_color[no]}

            ax = p_fig.fig.add_subplot(p_fig.gs[no, i])
            ax.set_facecolor("#F3F3F3")

            ax.tick_params(axis='x', which='both', bottom=False, top=False)

            xticklabels_conf = {"labels": list(),
                                "fontdict": {"fontsize": 30, "fontweight": "bold"}}

            xlabel_conf = {"xlabel": mask_title[no],
                           "fontdict": {"fontsize": 30, "fontweight": "bold"}}

            if i == 0:
                yticks = range(len(mask[list(rl_agents_alias.keys())[no]]))
                yticklabels_conf = {"labels": mask_alias[list(rl_agents_alias.keys())[no]],
                                    "rotation": 0,
                                    "fontdict": {"fontsize": 27, "fontweight": "bold"}}  # 22

            else:
                yticks = range(len(mask[list(rl_agents_alias.keys())[no]]))
                yticklabels_conf = {"labels": ["" for _ in yticks],
                                    "fontdict": {"fontsize": 20, "fontweight": "bold"}}

                ax.tick_params(axis='y', which='both', left=False, right=False)

            p_fig.heatmap_sub_plot_1d(dat, ax,
                                      heat_conf=heat_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                                      xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf,
                                      xlabel_conf=xlabel_conf,
                                      ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf,
                                      ylabel_conf=ylabel_conf)

    # Bar: workload permutation

    res_id = "work_permutation"
    data_id = "index_utility"
    # "tpch", "tpcds", "job", "tpch_skew", "dsb"
    benchmarks = ["tpch", "tpch_skew", "dsb"]
    data_perm = merge_work_bench(res_id, data_id, benchmarks)

    groups = benchmarks
    labels = list(rl_agents_alias.values())

    bar_conf = [{"color": colors[no], "hatch": hatches[no], "edgecolor": "#F0F0F0",
                 "error_kw": dict(lw=2, capsize=3, capthick=1.5)} for no in range(len(labels))]
    tick_conf = {"labelsize": 25, "pad": 20}

    xticks = None
    xlabel_conf = None
    xticklabels_conf = {"labels": [benchmarks_alias[g] for g in groups],
                        "fontdict": {"fontsize": 30, "fontweight": "bold"}}

    ylims = 0, None
    yticks = None
    yticklabels_conf = None

    ylabel = "Relative Cost Reduction (%)"
    ylabel_conf = {"ylabel": ylabel, "fontdict": {"fontsize": 30, "fontweight": "bold"}}

    leg_conf = {"loc": "upper center", "ncol": 5,
                "bbox_to_anchor": (.45, 1.2),
                "prop": {"size": 30, "weight": 510}}

    ax = p_fig.fig.add_subplot(p_fig.gs[:, 1])
    ax.set_aspect(0.027)
    ax.set_facecolor("#F3F3F3")

    p_fig.ebar_sub_plot(data_perm, ax,
                        groups, labels, gap, width,
                        bar_conf=bar_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                        xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf, xlabel_conf=xlabel_conf,
                        ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf, ylabel_conf=ylabel_conf)

    # ax.text(-1.55, -19., "(a) Feature Importance",
    #         fontdict={"fontsize": 33, "fontweight": "bold"})
    # ax.text(1.3, -19., "(b) Permutation Variance",
    #         fontdict={"fontsize": 33, "fontweight": "bold"})

    # ax.text(-1.63, -19., "(a) Feature Importance",
    #         fontdict={"fontsize": 36, "fontweight": "bold"})
    # ax.text(1.31, -19., "(b) Permutation Variance",
    #         fontdict={"fontsize": 36, "fontweight": "bold"})

    ax.text(-1.99, -19., "(a) Importance Distribution",
            fontdict={"fontsize": 36, "fontweight": "bold"})
    ax.text(1.11, -19., "(b) Performance Fluctuation",
            fontdict={"fontsize": 36, "fontweight": "bold"})

    if save_path is not None:
        plt.savefig(save_path, **save_conf)

    plt.show()


if __name__ == "__main__":
    save_path = None

    # Bar: varying frequency
    # work_freq, work_num
    res_id = "work_mask"
    data_id = "index_utility"

    benchmarks = ["tpch"]  # "tpch", "tpcds"
    # save_path = f"/data1/wz/index/index_eab/eab_olap/bench_result/multi_{res_id}_bench_heat.pdf"

    save_path = f"/data1/wz/index/index_eab/eab_olap/bench_result/multi_{res_id}_perm_bench_heat_bar.pdf"
    heat_bench_mask_plot(res_id, data_id, benchmarks, save_path=save_path)
