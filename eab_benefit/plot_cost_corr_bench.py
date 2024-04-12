# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: plot_cost_corr_bench
# @Author: Wei Zhou
# @Time: 2023/10/2 19:46

import os
import re
import json
import pickle

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import sys

sys.path.append("/data/wz/index/code_utils")

from result_plot.fig_plot_all import FigurePlot

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
mask_title = ["Log(Actual Query Latency)", "Log(Actual Query Latency)", "Log(Actual Query Latency)"]

text_alias = ["Query Plan", "SQL Text"]
red_method_alias = ["PCA", "LSI", "Doc2Vec"]

labels = [">", "≈", "<"]
labels = ["Act > Est", "Act < Est"]


def merge_work_bench_single(bench, est_id, act_id):
    data_load = f"/data/wz/index/index_eab/eab_other/cost_data/{bench}_corr_cost_data_tgt.json"
    with open(data_load, "r") as rf:
        data = json.load(rf)

    est_data = [dat[est_id] for dat in data]
    act_data = [dat[act_id] for dat in data]

    return {"est_data": est_data, "act_data": act_data}


def merge_work_bench(est_id, act_id, benchmarks):
    datas, datas_format, model_fit = list(), list(), list()
    for bench in benchmarks:
        data = merge_work_bench_single(bench, est_id, act_id)
        datas.append(data)

        x, y = np.log(data["est_data"]), np.log(data["act_data"])
        x_pre = np.array([e for e, a in zip(x, y) if e != -np.inf and a != -np.inf])
        y_pre = np.array([a for e, a in zip(x, y) if e != -np.inf and a != -np.inf])

        x_pre = x_pre.reshape(-1, 1)
        model = LinearRegression()
        model.fit(x_pre, y_pre)
        y_pred = model.predict(x_pre)

        model_fit.append([x_pre, y_pred])

        temp = list()
        x1 = [d1 for d1, d2, d3 in zip(x_pre, y_pre, y_pred) if d2 > d3]
        y1 = [d2 for d1, d2, d3 in zip(x_pre, y_pre, y_pred) if d2 > d3]
        y2 = [d3 for d1, d2, d3 in zip(x_pre, y_pre, y_pred) if d2 > d3]
        temp.append([x1, y1, y2])

        # x1 = [d1 for d1, d2, d3 in zip(x_pre, y_pre, y_pred) if d2 // d3 == 1]
        # y1 = [d2 for d1, d2, d3 in zip(x_pre, y_pre, y_pred) if d2 // d3 == 1]
        # y2 = [d3 for d1, d2, d3 in zip(x_pre, y_pre, y_pred) if d2 // d3 == 1]
        # temp.append([x1, y1, y2])

        x1 = [d1 for d1, d2, d3 in zip(x_pre, y_pre, y_pred) if d2 < d3]
        y1 = [d2 for d1, d2, d3 in zip(x_pre, y_pre, y_pred) if d2 < d3]
        y2 = [d3 for d1, d2, d3 in zip(x_pre, y_pre, y_pred) if d2 < d3]
        temp.append([x1, y1, y2])

        datas_format.append(temp)

    return datas_format, model_fit


def scatter_bench_corr_plot(res_id, data_id, benchmarks, save_path=None,
                            save_conf={"format": "pdf", "bbox_inches": "tight"}):
    # 1. Get the data.
    if not os.path.exists(save_path.replace(".pdf", f"_data_act_tgt.npy")):
        est_id, act_id = "tgt_est_hypo_cost", "tgt_act_not_hypo_cost"
        data_act, fit_act = merge_work_bench(est_id, act_id, benchmarks)
        data_save = save_path.replace(".pdf", f"_data_act_tgt.npy")
        np.save(data_save, data_act)
        data_save = save_path.replace(".pdf", f"_fit_act_tgt.npy")
        np.save(data_save, fit_act)

        est_id, act_id = "tgt_est_hypo_cost", "tgt_est_not_hypo_cost"
        data_hypo, fit_hypo = merge_work_bench(est_id, act_id, benchmarks)
        data_save = save_path.replace(".pdf", f"_data_hypo_tgt.npy")
        np.save(data_save, data_hypo)
        data_save = save_path.replace(".pdf", f"_fit_hypo_tgt.npy")
        np.save(data_save, fit_hypo)

        data = data_act + data_hypo
        fit = fit_act + fit_hypo
    else:
        data_load = save_path.replace(".pdf", f"_data_act_tgt.npy")
        data_act = np.load(data_load, allow_pickle=True)
        data_load = save_path.replace(".pdf", f"_fit_act_tgt.npy")
        fit_act = np.load(data_load, allow_pickle=True)

        data_load = save_path.replace(".pdf", f"_data_hypo_tgt.npy")
        data_hypo = np.load(data_load, allow_pickle=True)
        data_load = save_path.replace(".pdf", f"_fit_hypo_tgt.npy")
        fit_hypo = np.load(data_load, allow_pickle=True)

        data = np.append(data_act, data_hypo, axis=0)
        fit = np.append(fit_act, fit_hypo, axis=0)

    # 2. Define the figure.
    nrows, ncols = 2, 3
    figsize = (20, 30 // 3)
    figsize = (27, 30 // 2)
    # figsize = (27, 30 // 2.7)

    # nrows, ncols = 1, 3
    # figsize = (30 // 1, 27)

    p_fig = FigurePlot(nrows, ncols, figsize)
    # p_fig.fig.subplots_adjust(wspace=3.92, hspace=0.22)
    # p_fig.fig.subplots_adjust(wspace=.005, hspace=0.03)

    # 3. Draw the subplot.
    groups = text_alias
    # labels = benchmarks

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
    markers = [".", "+", "*", "d", "x", "^"][2:]

    gap, width = 0.25, 0.2

    # "#F0F0F0", "#F1F1F1", "#2A4458"
    scatter_conf = [{"c": colors[no], "s": 100, "alpha": 1.0, "marker": markers[no]} for no, _ in enumerate(labels)]

    tick_conf = {"labelsize": 25, "pad": 20}
    leg_conf = {"loc": "upper center", "ncol": 9,
                "bbox_to_anchor": (.5, 1.15),
                "prop": {"size": 23, "weight": "normal"}}
    leg_conf = None

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
    ylabel_conf = {"ylabel": ylabel,
                   "fontdict": {"fontsize": 30, "fontweight": "bold"}}

    for no, dat in enumerate(data):
        ax = p_fig.fig.add_subplot(p_fig.gs[no])
        ax.set_facecolor("#F3F3F3")

        if no == 0:
            ax.text(-1, 18.7, "Pearson Correlation: 0.7390",
                    fontdict={"fontsize": 30, "fontweight": "bold"})
        elif no == 3:
            ax.text(-1, 26.7, "Pearson Correlation: 0.9994",
                    fontdict={"fontsize": 30, "fontweight": "bold"})
        elif no == 1:
            ax.text(7, 15, "Pearson Correlation: 0.5567",
                    fontdict={"fontsize": 30, "fontweight": "bold"})
        elif no == 4:
            ax.text(7, 21.5, "Pearson Correlation: 0.9990",
                    fontdict={"fontsize": 30, "fontweight": "bold"})
        elif no == 2:
            ax.text(0, 21, "Pearson Correlation: 0.7401",
                    fontdict={"fontsize": 30, "fontweight": "bold"})
        elif no == 5:
            ax.text(0, 29.5, "Pearson Correlation: 0.9994",
                    fontdict={"fontsize": 30, "fontweight": "bold"})

        if no == 4:
            xlabel_conf = {"xlabel": "Estimated Query Cost", "labelpad": 10,
                           "fontdict": {"fontsize": 30, "fontweight": "bold"}}
        else:
            xlabel_conf = None

        if no == 0:
            ylabel = "Actual Query Latency"
            ylabel_conf = {"ylabel": ylabel,
                           "fontdict": {"fontsize": 27, "fontweight": "bold"}}
        elif no == 3:
            ylabel = "Estimated Query Cost w/ Index"
            ylabel_conf = {"ylabel": ylabel,
                           "fontdict": {"fontsize": 27, "fontweight": "bold"}}
        else:
            ylabel_conf = None

        if no == 0:
            # leg_conf = {"loc": "upper center", "ncol": 5,
            #             "bbox_to_anchor": (.5, 1.15),
            #             "prop": {"size": 30, "weight": 510}}
            leg_conf = {"loc": "upper left", "ncol": 1,
                        # "bbox_to_anchor": (.5, 1.15),
                        "prop": {"size": 30, "weight": 510}}
        else:
            leg_conf = None

        xticklabels_conf = {"labels": ["" for _ in groups], "rotation": 15,
                            "fontdict": {"fontsize": 20, "fontweight": "normal"}}

        # yticks = range(len(mask[list(rl_agents_alias.keys())[no]]))
        yticklabels_conf = {"labels": ["" for _ in labels],
                            "fontdict": {"fontsize": 20, "fontweight": "normal"}}

        p_fig.scatter_sub_plot_1d(dat, ax, labels,
                                  scatter_conf=scatter_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                                  xlims=xlims, xticks=xticks,
                                  xticklabels_conf=xticklabels_conf, xlabel_conf=xlabel_conf,
                                  ylims=ylims, yticks=yticks,
                                  yticklabels_conf=yticklabels_conf, ylabel_conf=ylabel_conf)

        ax.plot(fit[no][0], fit[no][1], color=colors[-1], linewidth=6)

    if save_path is not None:
        plt.savefig(save_path, **save_conf)

    plt.show()


if __name__ == "__main__":
    save_path = None

    # Bar: varying frequency
    # work_freq, work_num
    res_id = "cost_corr"
    data_id = "index_utility"

    benchmarks = ["tpch", "tpcds", "job"]
    save_path = f"/data/wz/index/index_eab/eab_other/bench_result/multi_{res_id}_bench_ebar.pdf"
    scatter_bench_corr_plot(res_id, data_id, benchmarks, save_path=save_path)
