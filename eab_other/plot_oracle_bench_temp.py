# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: plot_oracle_bench_temp
# @Author: Wei Zhou
# @Time: 2023/9/22 20:39

import os
import re
import json
import copy
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

# heu_algos = ["extend", "db2advis", "relaxation", "anytime"]
# heu_algo_alias = {"extend": "Extend", "db2advis": "DB2Advis",
#                   "relaxation": "Relaxation", "anytime": "DTA"}

algos_selected = ["extend", "db2advis"] + ["anytime", "drop"] + ["swirl", "drlinda"] + ["mcts"]
algos_selected_alias = ["Extend", "DB2Advis"] + ["DTA", "Drop"] + ["SWIRL", "DRLindex"] + ["MCTS"]
# ["Extend", "DB2Advis", "Relaxation"] + ["DTA", "AutoAdmin", "Drop"] + ["MCTS"] + ["SWIRL", "DRLindex", "DQN"]

oracle = ["cost_pure", "cost_per_sto"]
# oracle = ["benefit_pure", "benefit_per_sto"]

oracle_alias = ["cost", r"$\frac{cost}{storage}$"]
oracle_alias = ["Cost", "Cost / Storage"]

algos_selected = ["extend", "db2advis"] + ["anytime", "drop"] + ["swirl", "drlinda"] + ["mcts"]
algos_selected_alias = ["Extend (①)", "DB2Advis (①)"] + ["DTA (①)", "Drop (①)"] \
                       + ["SWIRL (②)", "DRLindex (②)"] + ["MCTS (②)"]


# Bar: varying frequency

def merge_work_bench_single_bak(bench, constraint, res_id, data_id):
    data = dict()
    for algo in heu_algos:
        for o in oracle:
            temp = list()

            data_load = f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_v2_simple/" \
                        f"{bench_id[bench]}_multi_{res_id}_{constraint}_{o}_index_simple.json"
            with open(data_load, "r") as rf:
                dat = json.load(rf)

            if algo not in dat[0].keys():
                continue

            if algo not in data.keys():
                data[algo] = list()

            if data_id == "index_utility":
                temp.extend([1 - d[algo]["total_ind_cost"] / d[algo]["total_no_cost"] for d in dat])
            elif data_id == "time_duration":
                temp.extend([d[algo]["sel_info"]["time_duration"] for d in dat])

            data[algo].append(temp)

    for algo in rl_agents.keys():
        if algo not in data.keys():
            data[algo] = list()

        data_load = f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_v2_simple/" \
                    f"{bench_id[bench]}_multi_{res_id}_index_{algo}_simple.json"
        with open(data_load, "r") as rf:
            dat = json.load(rf)[:20]

        for o in ["cost_pure", "cost_per_sto"]:  # oracle, ["cost_pure", "cost_per_sto"]
            if constraint == "storage":
                c = "sto10w"
            elif constraint == "number":
                c = "num10w"
            it = [key for key in dat[0].keys() if o in key and c in key]

            temp = list()
            for da in dat:
                if data_id == "index_utility":
                    temp.extend([1 - da[k]["total_ind_cost"] / da[k]["total_no_cost"]
                                 for k in da.keys() if k in it])
                elif data_id == "time_duration":
                    temp.extend([da[k]["sel_info"]["time_duration"] for k in da.keys() if k in it])
            data[algo].append(temp)

    # add more results.

    data_load = f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/" \
                f"work_level_simple/{bench_id[bench]}_multi_work_index_swirl_simple.json"
    with open(data_load, "r") as rf:
        dat = json.load(rf)[:20]

    it = [key for key in dat[0].keys() if c in key and "unique" not in key]

    temp = list()
    for da in dat:
        if data_id == "index_utility":
            temp.extend([1 - da[k]["total_ind_cost"] / da[k]["total_no_cost"]
                         for k in da.keys() if k in it])
        elif data_id == "time_duration":
            temp.extend([da[k]["sel_info"]["time_duration"] for k in da.keys() if k in it])
    data["swirl"][-1] = temp[:]

    for algo in ["drlinda", "dqn"]:
        data_load = f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/" \
                    f"work_level_simple/{bench_id[bench]}_multi_work_index_{algo}_simple.json"
        with open(data_load, "r") as rf:
            dat = json.load(rf)[:20]

        it = [key for key in dat[0].keys() if "sto10w" in key and "unique" not in key]

        temp = list()
        for da in dat:
            if data_id == "index_utility":
                temp.extend([1 - da[k]["total_ind_cost"] / da[k]["total_no_cost"]
                             for k in da.keys() if k in it])
            elif data_id == "time_duration":
                temp.extend(
                    [da[k]["sel_info"]["time_duration"] for k in da.keys() if k in it])
        data[algo][0] = temp[:]

    return data


def merge_work_bench_single(bench, constraint, res_id, data_id):
    data = dict()
    data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_oracle_new_simple/"
                  f"tpch_1gb_template_18_multi_work_oracle_{constraint}_benefit_pure_group1_index_simple.json",
                  f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_oracle_new_simple/"
                  f"tpch_1gb_template_18_multi_work_oracle_{constraint}_None_group1_index_simple.json",
                  f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_oracle_new_simple/"
                  f"tpch_1gb_template_18_multi_work_oracle_{constraint}_None_group2_index_simple.json",
                  f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_oracle_new_simple/"
                  f"tpch_1gb_template_18_multi_work_oracle_{constraint}_cost_per_sto_group2_index_simple.json",

                  f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_oracle_new_simple/"
                  f"tpch_1gb_template_18_multi_work_oracle_{constraint}_cost_per_sto_drop_index_simple.json",

                  f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_oracle_new_simple/"
                  f"tpch_1gb_template_18_multi_work_oracle_{constraint}_None_mcts_index_simple.json",
                  f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_oracle_new_simple/"
                  f"tpch_1gb_template_18_multi_work_oracle_{constraint}_benefit_per_sto_mcts_index_simple.json"]

    for data_load in data_loads:
        with open(data_load, "r") as rf:
            dat = json.load(rf)

        if "mcts" in data_load:
            continue
        else:
            for algo in dat[0].keys():
                if "cost_per_sto_group2" in data_load and algo == "drop":
                    continue

                temp = list()
                if algo not in data.keys():
                    data[algo] = list()

                if data_id == "index_utility":
                    temp.extend([1 - d[algo]["total_ind_cost"] / d[algo]["total_no_cost"] for d in dat])
                elif data_id == "time_duration":
                    temp.extend([d[algo]["sel_info"]["time_duration"] for d in dat])

                data[algo].append(temp)

    for algo in rl_agents.keys():
        if algo not in data.keys():
            data[algo] = list()

        data_load = f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_v2_simple/" \
                    f"{bench_id[bench]}_multi_{res_id}_index_{algo}_simple.json"
        with open(data_load, "r") as rf:
            dat = json.load(rf)[:20]

        for o in ["cost_pure", "cost_per_sto"]:  # oracle, ["cost_pure", "cost_per_sto"]
            if constraint == "storage":
                c = "sto10w"
            elif constraint == "number":
                c = "num10w"
            it = [key for key in dat[0].keys() if o in key and c in key]

            temp = list()
            for da in dat:
                if data_id == "index_utility":
                    temp.extend([1 - da[k]["total_ind_cost"] / da[k]["total_no_cost"]
                                 for k in da.keys() if k in it])
                elif data_id == "time_duration":
                    temp.extend([da[k]["sel_info"]["time_duration"] for k in da.keys() if k in it])
            data[algo].append(temp)

    # add more results.

    data_load = f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/" \
                f"work_level_simple/{bench_id[bench]}_multi_work_index_swirl_simple.json"
    with open(data_load, "r") as rf:
        dat = json.load(rf)[:20]

    it = [key for key in dat[0].keys() if c in key and "unique" not in key]

    temp = list()
    for da in dat:
        if data_id == "index_utility":
            temp.extend([1 - da[k]["total_ind_cost"] / da[k]["total_no_cost"]
                         for k in da.keys() if k in it])
        elif data_id == "time_duration":
            temp.extend([da[k]["sel_info"]["time_duration"] for k in da.keys() if k in it])
    data["swirl"][-1] = temp[:]

    for algo in ["drlinda", "dqn"]:
        data_load = f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/" \
                    f"work_level_simple/{bench_id[bench]}_multi_work_index_{algo}_simple.json"
        with open(data_load, "r") as rf:
            dat = json.load(rf)[:20]

        it = [key for key in dat[0].keys() if "sto10w" in key and "unique" not in key]

        temp = list()
        for da in dat:
            if data_id == "index_utility":
                temp.extend([1 - da[k]["total_ind_cost"] / da[k]["total_no_cost"]
                             for k in da.keys() if k in it])
            elif data_id == "time_duration":
                temp.extend(
                    [da[k]["sel_info"]["time_duration"] for k in da.keys() if k in it])
        data[algo][0] = temp[:]

    for data_load in data_loads:
        temp = list()
        with open(data_load, "r") as rf:
            dat = json.load(rf)

        if "mcts" in data_load:
            algo = "mcts"
            if algo not in data.keys():
                data[algo] = list()

            if data_id == "index_utility":
                temp.extend([1 - d["total_ind_cost"] / d["total_no_cost"] for d in dat])
            elif data_id == "time_duration":
                temp.extend([d["sel_info"]["time_duration"] for d in dat])

            data[algo].append(temp)

    # for algo in list(data.keys()):
    #     if algo not in algos_selected:
    #         data.pop(algo)

    data_new = dict()
    for algo in algos_selected:
        data_new[algo] = copy.deepcopy(data[algo])

    return data_new


def merge_work_bench(res_id, data_id, benchmarks):
    datas, datas_format = list(), list()
    for bench in benchmarks:
        for constraint in ["storage", "number"]:
            data = merge_work_bench_single(bench, constraint, res_id, data_id)
            datas.append(data)

            temp = list()
            for algo in list(data.keys()):
                if data_id == "index_utility":
                    temp.append({"query_data": [100 * np.mean(dat) for dat in data[algo]],
                                 "yerr_data": [np.std(100 * np.array(dat)) for dat in data[algo]]})
                elif data_id == "time_duration":
                    temp.append({"query_data": [np.mean(dat) for dat in data[algo]],
                                 "yerr_data": [np.log(dat) for dat in data[algo]]})

            datas_format.append(temp)

    return datas_format


def ebar_bench_oracle_plot(res_id, benchmarks, save_path=None,
                           save_conf={"format": "pdf", "bbox_inches": "tight"}):
    # 1. Get the data.
    data_id = "index_utility"
    data_u = merge_work_bench(res_id, data_id, benchmarks)
    # data_save = save_path.replace(".pdf", f"_{data_id}.json")
    # with open(data_save, "w") as wf:
    #     json.dump(data_u, wf, indent=2)

    # data_id = "time_duration"
    # data_t = merge_work_bench(res_id, data_id, benchmarks)
    # data_save = save_path.replace(".pdf", f"_{data_id}.json")
    #
    # # data_load = save_path.replace(".pdf", ".json")
    # with open(data_save, "w") as wf:
    #     json.dump(data_t, wf, indent=2)

    # data = data_u + data_t
    data = data_u
    # [print("\t".join(list(map(str, [np.round(d, 2) for d in dat['query_data']])))) for dat in data[0]]

    # 2. Define the figure.
    nrows, ncols = 1, 2
    figsize = (27, 30 // 3)
    figsize = (27, 30 // 3.3)
    # figsize = (27, 30 // 1.5)
    p_fig = ErrorBarPlot(nrows, ncols, figsize)
    # p_fig.fig.subplots_adjust(wspace=3.92, hspace=0.22)
    p_fig.fig.subplots_adjust(hspace=0.08)

    # 3. Draw the subplot.
    groups = oracle
    # labels = ["extend", "db2advis", "anytime", "relaxation"] + list(rl_agents.keys())
    labels = heu_algos + list(rl_agents.keys())
    labels = algos_selected_alias

    # 3.1 set Bar properties.
    colors = [p_fig.color["green"], p_fig.color["blue"], p_fig.color["grey"],
              p_fig.color["orange"], p_fig.color["pink"], p_fig.color["red"],
              p_fig.color["purple"], p_fig.color["brown"], p_fig.color["yellow"]]

    colors = ["#2A4458", "#336485", "#3E86B5", "#95A77E", "#E5BE79",
              # "#11325D", "#365083",
              "#736B9D", "#B783AF", "#F5A673", "#ABC9C8",  # "#FCDB72",
              # "#404D5B", "#5492C7",
              "#98989C",  # "#B0B0B5", "#757576", "#E7E5DF",
              # "#99B86B", "#688A4C",
              # "#545969",
              "#A4757D",
              # "#E7987C", "#8B91B6", "#7771A4"
              ]  # [2:]

    hatches = ["", "/", "-",
               "x", "||", "\\",
               "--", "|", "//",
               "-", "x", "||"]

    gap, width = 0.25, 0.2

    # "#F0F0F0", "#F1F1F1", "#2A4458"
    bar_conf = [{"color": colors[no], "hatch": hatches[no], "edgecolor": "#F0F0F0",
                 "error_kw": dict(lw=2, capsize=3, capthick=1.5)} for no in range(len(labels))]
    tick_conf = {"labelsize": 27, "pad": 20}
    leg_conf = {"loc": "upper center", "ncol": 9,
                "bbox_to_anchor": (.95, 1.15),
                "prop": {"size": 23, "weight": "normal"}}
    # leg_conf = None

    # 3.2 set X properties.
    xlims = None
    xticks = None
    xticklabels_conf = {"labels": groups, "rotation": 15,
                        "fontdict": {"fontsize": 13, "fontweight": "normal"}}

    xlabels = None
    xlabel_conf = None  # {"xlabel": xlabels, "fontdict": {"fontsize": "large", "fontweight": "bold"}}

    # 3.3 set Y properties.
    # if data_id == "index_utility":
    #     ylims = (0., 60.)  # 0.0, 22.0
    # else:
    #     ylims = None
    ylims = 0., None
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

        if no in [0, 1]:
            xticklabels_conf = {"labels": ["" for _ in oracle_alias],
                                "fontdict": {"fontsize": 30, "fontweight": "bold"}}
            if no == 0:
                ylabel = "Relative Cost Reduction (%)"
                ylabel_conf = {"ylabel": ylabel,
                               "fontdict": {"fontsize": 30, "fontweight": "bold"}}
            else:
                ylabel_conf = None

        elif no in [2, 3]:
            xticklabels_conf = {"labels": oracle_alias,
                                "fontdict": {"fontsize": 33, "fontweight": "bold"}}

            if no == 2:
                xlabel_conf = {"xlabel": "(a) Constraint: Storage", "labelpad": 30,
                               "fontdict": {"fontsize": 35, "fontweight": "bold"}}

                ylabel = "Time Duration (s)"
                ylabel_conf = {"ylabel": ylabel, "fontdict": {"fontsize": 30, "fontweight": "bold"}}

            else:
                xlabel_conf = {"xlabel": "(b) Constraint: #Index", "labelpad": 30,
                               "fontdict": {"fontsize": 35, "fontweight": "bold"}}

                ylabel_conf = None

            ax.set_yscale("log")

        xticklabels_conf = {"labels": oracle_alias,
                            "fontdict": {"fontsize": 33, "fontweight": "bold"}}
        if no == 0:
            xlabel_conf = {"xlabel": "(a) Constraint: Storage", "labelpad": 30,
                           "fontdict": {"fontsize": 35, "fontweight": "bold"}}

            # ylabel = "Time Duration (s)"
            # ylabel_conf = {"ylabel": ylabel, "fontdict": {"fontsize": 30, "fontweight": "bold"}}

        else:
            xlabel_conf = {"xlabel": "(b) Constraint: #Index", "labelpad": 30,
                           "fontdict": {"fontsize": 35, "fontweight": "bold"}}

            ylabel_conf = None

        if no == 0:
            leg_conf = {"loc": "upper center", "ncol": 5,
                        "bbox_to_anchor": (1.05, 1.25),
                        "prop": {"size": 30, "weight": 510}}

            leg_conf = {"loc": "upper center", "ncol": 7,
                        "bbox_to_anchor": (1.0, 1.2),
                        "prop": {"size": 30, "weight": 510}}
        else:
            leg_conf = None

        # labels = list(heu_algo_alias.values()) + list(rl_agents_alias.values())
        p_fig.ebar_sub_plot(dat, ax,
                            groups, labels, gap, width,
                            bar_conf=bar_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                            xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf, xlabel_conf=xlabel_conf,
                            ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf, ylabel_conf=ylabel_conf)

        for i in range(len(groups)):
            pos = gap + (1 + i) * (gap + len(labels) * width) + 3.5 * width
            ax.axvline(pos, color="#d82821", linestyle="-.", linewidth=6, zorder=200)

    if save_path is not None:
        plt.savefig(save_path, **save_conf)

    plt.show()


if __name__ == "__main__":
    save_path = None

    res_id = "work_oracle"
    benchmarks = ["tpch"]

    save_path = f"/data1/wz/index/index_eab/eab_olap/bench_result/multi_bench_{res_id}_bar_v2.pdf"
    ebar_bench_oracle_plot(res_id, benchmarks, save_path=save_path)
