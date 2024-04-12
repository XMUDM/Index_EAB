# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: plot_cand_bench_temp.py
# @Author: Wei Zhou
# @Time: 2023/9/22 20:19

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

benchmarks = ["tpch", "tpcds", "job"]
bench_id = {"tpch": "tpch_1gb_template_18",
            "tpcds": "tpcds_1gb_template_79",
            "job": "job_template_33",
            "tpch_skew": "tpch_skew_1gb_template_18",
            "dsb": "dsb_1gb_template_53"}

heu_algos = ["extend", "db2advis", "relaxation", "anytime", "auto_admin", "drop"]
rl_agents = {"swirl": list(), "drlinda": list(), "dqn": list()}  # , "mcts": list(), "bandits": list()

heu_algos = ["db2advis", "relaxation", "anytime"]
rl_agents = {"swirl": list(), "dqn": list(), "mcts": list()}

heu_algo_alias = {"db2advis": "DB2Advis", "relaxation": "Relaxation", "anytime": "DTA"}
rl_agents_alias = {"swirl": "SWIRL (w/o invalid action masking)", "dqn": "DQN", "mcts": "MCTS"}

cand_method = ["permutation", "dqn_rule", "openGauss"]
cand_method_alias = ["Permutation", "Rule", "openGauss"]
cand_method_alias = ["Permutation", "SyntacticRule", "openGauss"]

titles = ["(a) TPC-H", "(b) TPC-DS", "(c) JOB"]

benchmarks = ["tpch", "tpcds", "dsb"]

heu_algo_alias = {"db2advis": "DB2Advis (①)", "relaxation": "Relaxation (①)", "anytime": "DTA (①)"}
rl_agents_alias = {"swirl": "SWIRL (②)", "dqn": "DQN (②)", "mcts": "MCTS (②)"}

rl_agents = {"swirl": list(), "dqn": list()}
rl_agents_alias = {"swirl": "SWIRL (②)", "dqn": "DQN (②)"}

titles = ["(a) TPC-H", "(b) TPC-DS", "(c) DSB"]

# Bar: varying candidate generation method

def merge_work_bench_single(bench, res_id, data_id):
    data = dict()

    # 1. heu_algos:
    for algo in heu_algos:
        if bench == "tpch":
            data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/" \
                          f"work_level_simple/{bench_id[bench]}_multi_work_index0-20_simple.json"]

        elif bench == "tpcds":
            if algo == "relaxation":
                data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/" \
                              f"work_level_simple/{bench_id[bench]}_multi_work_index_{algo}{no_id}_simple.json"
                              for no_id in ["0", "1", "2-5", "6-10"]]
            else:
                data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/" \
                              f"work_level_simple/{bench_id[bench]}_multi_work_index0-10_simple.json"]

        elif bench == "job":
            data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/" \
                          f"work_level_simple/{bench_id[bench]}_multi_work_index0-20_simple.json"]

        elif bench == "dsb":
            if algo == "relaxation":
                data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/" \
                              f"work_level_simple/{bench_id[bench]}_multi_work_index_relaxation{no_id}_simple.json"
                              for no_id in ["0-20", "20-40", "40-60", "60-80", "80-100"]]
            else:
                data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/" \
                              f"work_level_simple/{bench_id[bench]}_multi_work_index{no_id}_simple.json"
                              for no_id in ["0-20", "20-40", "40-60", "60-80", "80-100"]]

        for data_load in data_loads:
            with open(data_load, "r") as rf:
                dat = json.load(rf)[:10]

            if f"{algo}_permutation" not in data.keys():
                data[f"{algo}_permutation"] = list()

            if data_id == "index_utility":
                data[f"{algo}_permutation"].extend(
                    [1 - d[algo]["total_ind_cost"] / d[algo]["total_no_cost"] for d in dat])
            elif data_id == "time_duration":
                data[f"{algo}_permutation"].extend([d[algo]["sel_info"]["time_duration"] for d in dat])

    for cand in cand_method:
        if bench != "dsb" and cand == "permutation":
            continue

        if bench == "tpch":
            data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_simple/" \
                          f"{bench_id[bench]}_multi_{res_id}_{cand}_True_index_simple.json"]

        elif bench == "tpcds":
            data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_simple/" \
                          f"{bench_id[bench]}_multi_{res_id}_{cand}_True_index0-10_simple.json"]

        elif bench == "dsb":
            data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_simple/" \
                          f"{bench_id[bench]}_multi_work_True_{cand}_group1_index0-10_simple.json",
                          f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_simple/" \
                          f"{bench_id[bench]}_multi_work_True_{cand}_group2_index0-10_simple.json"]

        for data_load in data_loads:
            with open(data_load, "r") as rf:
                dat = json.load(rf)[:10]

            for algo in dat[0].keys():
                if f"{algo}_{cand}" not in data.keys():
                    data[f"{algo}_{cand}"] = list()

                if data_id == "index_utility":
                    data[f"{algo}_{cand}"].extend(
                        [1 - d[algo]["total_ind_cost"] / d[algo]["total_no_cost"] for d in dat])
                elif data_id == "time_duration":
                    data[f"{algo}_{cand}"].extend([d[algo]["sel_info"]["time_duration"] for d in dat])

    # 2. rl_agents
    if bench == "job":
        data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_simple/" \
                      f"{bench_id[bench]}_multi_{res_id}_index" + "_{}_simple.json",
                      f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_simple/" \
                      f"{bench_id[bench]}_multi_{res_id}_index" + "_{}_2_simple.json"]
    else:
        data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/{res_id}_simple/" \
                      f"{bench_id[bench]}_multi_{res_id}_index" + "_{}_simple.json"]

    for data_load in data_loads:
        for algo in rl_agents.keys():
            if algo == "mcts":
                continue

            # if not os.path.exists(data_load.format(algo)):
            #     data[algo].extend([0])
            # else:
            with open(data_load.format(algo), "r") as rf:
                dat = json.load(rf)[:10]

            for cand in cand_method:
                if f"{algo}_{cand}" not in data.keys():
                    data[f"{algo}_{cand}"] = list()

                cs = [c for c in dat[0].keys() if cand in c]
                for da in dat:
                    if data_id == "index_utility":
                        data[f"{algo}_{cand}"].extend(
                            [1 - d["total_ind_cost"] / d["total_no_cost"] for k, d in da.items() if k in cs])
                    elif data_id == "time_duration":
                        data[f"{algo}_{cand}"].extend(
                            [d["sel_info"]["time_duration"] for k, d in da.items() if k in cs])

    # 3. mcts
    if bench == "tpch":
        data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/work_cand_simple/"
                      f"{bench_id[bench]}_multi_{res_id}_True_" + "{}_index0-10_mcts_simple.json"]
    elif bench == "tpcds":
        data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/work_cand_simple/" \
                      f"{bench_id[bench]}_multi_work_True_" + "{}_index0-10_mcts_simple.json"]
    elif bench == "dsb":
        data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/work_cand_simple/"
                      f"{bench_id[bench]}_multi_work_True_" + "{}_index0-10_mcts_simple.json"]
    else:
        data_loads = [f"/data1/wz/index/index_eab/eab_olap/bench_result/{bench}/work_cand_simple/"
                      f"{bench_id[bench]}_multi_{res_id}_True_" + "{}_index0-10_mcts_simple.json"]

    algo = "mcts"
    for data_load in data_loads:
        for cand in cand_method:
            if f"{algo}_{cand}" not in data.keys():
                data[f"{algo}_{cand}"] = list()

            if not os.path.exists(data_load.format(cand)):
                data[f"{algo}_{cand}"].extend([0])
                continue

            with open(data_load.format(cand), "r") as rf:
                dat = json.load(rf)[:10]

            if data_id == "index_utility":
                data[f"{algo}_{cand}"].extend(
                    [1 - da["total_ind_cost"] / da["total_no_cost"] for da in dat])
            elif data_id == "time_duration":
                data[f"{algo}_{cand}"].extend([da["sel_info"]["time_duration"] for da in dat])

    if data_id == "index_utility":
        data_format = list()
        for algo in heu_algos + list(rl_agents.keys()):
            data_format.append({"query_data": [100 * np.mean(data[f"{algo}_{cand}"]) for cand in cand_method],
                                "yerr_data": [np.std(100 * np.array(data[f"{algo}_{cand}"])) for cand in cand_method]})

    elif data_id == "time_duration":
        # data_format = dict()
        data_format = list()
        for algo in heu_algos + list(rl_agents.keys()):
            # data_format[algo] = {"query_data": [np.mean(data[f"{algo}_{cand}"]) for cand in cand_method],
            #                      "yerr_data": [np.std(data[f"{algo}_{cand}"]) for cand in cand_method]}
            data_format.append({"query_data": [np.mean(data[f"{algo}_{cand}"]) for cand in cand_method],
                                "yerr_data": [np.std(data[f"{algo}_{cand}"]) for cand in cand_method]})

    return data_format


def merge_work_bench(res_id, data_id, benchmarks):
    datas = list()
    for bench in benchmarks:
        datas.append(merge_work_bench_single(bench, res_id, data_id))

    return datas


def ebar_bench_cand_plot(res_id, benchmarks, save_path=None,
                         save_conf={"format": "pdf", "bbox_inches": "tight"}):
    # 1. Get the data.
    data_id = "index_utility"
    data_u = merge_work_bench(res_id, data_id, benchmarks)
    # data_save = save_path.replace(".pdf", f"_{data_id}.json")
    # with open(data_save, "w") as wf:
    #     json.dump(data_u, wf, indent=2)
    # data_u += data_u

    data_id = "time_duration"
    data_t = merge_work_bench(res_id, data_id, benchmarks)
    # data_save = save_path.replace(".pdf", f"_{data_id}.json")
    # with open(data_save, "w") as wf:
    #     json.dump(data_t, wf, indent=2)
    # data_t += data_t

    data = data_u + data_t
    # [print("\t".join(list(map(str, [np.round(d, 2) for d in dat['query_data']])))) for dat in data_u[0]]
    # [print("\t".join(list(map(str, [np.round(d, 2) for d in dat['query_data']])))) for dat in data_t[0]]

    # data_load = save_path.replace(".pdf", ".json")

    # 2. Define the figure.
    nrows, ncols = 2, 2
    figsize = (27, 30 // 1.5)
    figsize = (20, 30 // 3)

    nrows, ncols = 2, 3
    figsize = (27, 30 // 2.5)

    figsize = (27, 30 // 2.5)

    fontsize = 25  # 25, 30

    p_fig = ErrorBarPlot(nrows, ncols, figsize)
    # p_fig.fig.subplots_adjust(wspace=3.92, hspace=0.22)
    p_fig.fig.subplots_adjust(hspace=0.12)
    p_fig.fig.subplots_adjust(wspace=0.15, hspace=0.12)

    # 3. Draw the subplot.
    groups = cand_method
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
               "-", "x", "||"]

    gap, width = 0.25, 0.2

    # "#F0F0F0", "#F1F1F1", "#2A4458"
    bar_conf = [{"color": colors[no], "hatch": hatches[no], "edgecolor": "#F0F0F0",
                 "error_kw": dict(lw=2, capsize=3, capthick=1.5)} for no in range(len(labels))]
    # tick_conf = {"labelsize": 25, "pad": 20}
    tick_conf = {"labelsize": 25}
    # leg_conf = {"loc": "upper center", "ncol": 9,
    #             "bbox_to_anchor": (.5, 1.15),
    #             "prop": {"size": 23, "weight": "normal"}}
    leg_conf = {"loc": "upper center", "ncol": 9,
                "bbox_to_anchor": (.1, 1.15),
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

    # query_datas = {"query_data": [np.mean(100 * np.array(data[algo])) for algo in data.keys()],
    #                "yerr_data": [np.std(100 * np.array(data[algo])) for algo in data.keys()]}
    # groups = list(data.keys())

    xticks = None

    labels = list(heu_algo_alias.values()) + list(rl_agents_alias.values())

    for no, dat in enumerate(data):
        if no // ncols == 0:
            # colors = ["#2A4458", "#336485", "#3E86B5", "#95A77E", "#E5BE79"]
            bar_conf = [{"color": colors[no], "hatch": hatches[no], "edgecolor": "#F0F0F0",
                         "error_kw": dict(lw=2, capsize=3, capthick=1.5)} for no in range(len(labels))]

            xticklabels_conf = {"labels": ["" for _ in cand_method_alias],
                                "fontdict": {"fontsize": fontsize, "fontweight": "bold"}}
        elif no // ncols == 1:
            # colors = ["#736B9D", "#B783AF", "#F5A673", "#ABC9C8", "#A4757D"]
            bar_conf = [{"color": colors[no], "hatch": hatches[no], "edgecolor": "#F0F0F0",
                         "error_kw": dict(lw=2, capsize=3, capthick=1.5)} for no in range(len(labels))]

            xticklabels_conf = {"labels": cand_method_alias,
                                "fontdict": {"fontsize": fontsize, "fontweight": "bold"}}

            xticklabels_conf = {"labels": cand_method_alias,
                                "rotation": 10,
                                "fontdict": {"fontsize": fontsize, "fontweight": "bold"}}

        if no // ncols == 1:
            xlabel_conf = {"xlabel": titles[no % ncols],
                           # "labelpad": 20,
                           "fontdict": {"fontsize": 30, "fontweight": "bold"}}
        else:
            xlabel_conf = None

        if no == 0:
            ylabel = "Relative Cost Reduction (%)"
            ylabel_conf = {"ylabel": ylabel, "labelpad": 20,
                           "fontdict": {"fontsize": fontsize, "fontweight": "bold"}}
        elif no == ncols:
            ylabel = "Time Duration (s)"
            ylabel_conf = {"ylabel": ylabel,
                           "fontdict": {"fontsize": fontsize, "fontweight": "bold"}}
        else:
            ylabel_conf = None

        if no == 0:
            if ncols == 2:
                leg_conf = {"loc": "upper center", "ncol": 5,
                            "bbox_to_anchor": (1.06, 1.2),
                            "prop": {"size": 30, "weight": 510}}
            elif ncols == 3:
                leg_conf = {"loc": "upper center", "ncol": 5,
                            "bbox_to_anchor": (1.7, 1.3),
                            "prop": {"size": 30, "weight": 510}}

                leg_conf = {"loc": "upper center", "ncol": 6,
                            "bbox_to_anchor": (1.7, 1.35),
                            "prop": {"size": 30, "weight": 510}}

                leg_conf = {"loc": "upper center", "ncol": 6,
                            "bbox_to_anchor": (1.6, 1.33),
                            "prop": {"size": 30, "weight": 510}}

        else:
            leg_conf = None

        if no == 2:
            ylims = 0, None
        else:
            ylims = None

        ax = p_fig.fig.add_subplot(p_fig.gs[no // ncols, no % ncols])
        ax.set_facecolor("#F3F3F3")

        p_fig.ebar_sub_plot(dat, ax,
                            groups, labels, gap, width,
                            bar_conf=bar_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                            xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf, xlabel_conf=xlabel_conf,
                            ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf, ylabel_conf=ylabel_conf)

        for i in range(len(groups)):
            pos = gap + (1 + i) * (gap + len(labels) * width) + 2.5 * width
            ax.axvline(pos, color="#d82821", linestyle="-.", linewidth=6, zorder=200)

        if no // ncols == 1:
            ax.set_yscale("log")

    if save_path is not None:
        plt.savefig(save_path, **save_conf)

    plt.show()


if __name__ == "__main__":
    save_path = None

    # Bar: varying frequency
    # work_freq, work_num, work_cand
    res_id = "work_cand"

    # "tpch", "tpch_skew", "job"
    save_path = f"/data1/wz/index/index_eab/eab_olap/bench_result/multi_bench_{res_id}_ebar_v3.pdf"
    ebar_bench_cand_plot(res_id, benchmarks, save_path=save_path)
