# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: tpch_multi_bench_result
# @Author: Wei Zhou
# @Time: 2023/8/24 22:44

import re
import json

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/data1/wz/index/code_utils")

from result_plot.fig_plot import BoxPlot, ErrorBarPlot

heu_algos = ["extend", "db2advis", "relaxation", "anytime", "auto_admin", "drop"]
rl_agents = {"swirl": list(), "drlinda": list(), "dqn": list()}

temp_num = 22
row_qnum = 11


def get_res_data(data, algo, data_id):
    if data_id == "index_utility":
        return [1 - item[algo]["total_ind_cost"] / item[algo]["total_no_cost"] for item in data]

    elif data_id == "time_duration":
        return [item[algo]["sel_info"]["time_duration"] for item in data]

    elif data_id == "enumeration_step":
        return [len(item[algo]["sel_info"]["step"]["selected"]) for item in data]

    elif data_id == "enumeration_num":
        return [len(re.findall("combination", str(item[algo]["sel_info"]["step"]))) for item in data]

    elif data_id == "cost_requests":
        return [item[algo]["sel_info"]["cost_requests"] for item in data]

    elif data_id == "cache_hits":
        return [item[algo]["sel_info"]["cache_hits"] for item in data]

    elif data_id == "estimation_duration":
        return [item[algo]["sel_info"]["estimation_duration"] for item in data]

    elif data_id == "estimation_num":
        return [item[algo]["sel_info"]["estimation_num"] for item in data]

    elif data_id == "simulation_duration":
        return [item[algo]["sel_info"]["simulation_duration"] for item in data]

    elif data_id == "simulation_num":
        return [item[algo]["sel_info"]["simulation_num"] for item in data]


def merge_query_result(data_id):
    result = dict()

    sel_load = "/data1/wz/index/index_eab/eab_olap/bench_result/tpch_skew/query_level/tpch_skew_1gb_template_22_multi_query_index_simple.json"
    with open(sel_load, "r") as rf:
        sel_data = json.load(rf)

    for qno in sel_data.keys():
        result[qno] = dict()
        for algo in heu_algos:
            result[qno][algo] = get_res_data(sel_data[qno], algo, data_id)

    for algo in list(rl_agents.keys()):
        sel_load = f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch_skew/query_level/tpch_skew_1gb_template_22_multi_query_index_{algo}_simple.json"
        with open(sel_load, "r") as rf:
            sel_data = json.load(rf)

        for qno in sel_data.keys():
            for instance in sel_data[qno][0].keys():
                name = algo

                if "num" in instance:
                    continue

                # if "sto" in instance:
                #     name = f"{algo}_sto_s{re.findall(r'_s([0-9]+)_', instance)[0]}"
                # elif "num" in instance:
                #     name = f"{algo}_num_s{re.findall(r'_s([0-9]+)_', instance)[0]}"
                #
                # if "unique" in instance:
                #     name += "unique"
                # elif "duplicate" in instance:
                #     name += "duplicate"

                if name not in result[qno].keys():
                    result[qno][name] = list()

                result[qno][name].extend(get_res_data(sel_data[qno], instance, data_id))

    return result


def merge_work_result(data_id):
    result = dict()

    sel_loads = ["/data1/wz/index/index_eab/eab_olap/bench_result/tpch_skew/work_level/tpch_skew_1gb_template_18_multi_work_index0-20_simple.json",
                 "/data1/wz/index/index_eab/eab_olap/bench_result/tpch_skew/work_level/tpch_skew_1gb_template_18_multi_work_index20-40_simple.json",
                 "/data1/wz/index/index_eab/eab_olap/bench_result/tpch_skew/work_level/tpch_skew_1gb_template_18_multi_work_index40-60_simple.json",
                 "/data1/wz/index/index_eab/eab_olap/bench_result/tpch_skew/work_level/tpch_skew_1gb_template_18_multi_work_index60-80_simple.json",
                 "/data1/wz/index/index_eab/eab_olap/bench_result/tpch_skew/work_level/tpch_skew_1gb_template_18_multi_work_index80-100_simple.json"]
    for algo in heu_algos:
        result[algo] = list()
        for sel_load in sel_loads:
            with open(sel_load, "r") as rf:
                sel_data = json.load(rf)

            result[algo].extend(get_res_data(sel_data, algo, data_id))

    for algo in list(rl_agents.keys()):
        sel_load = f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch_skew/work_level/tpch_skew_1gb_template_18_multi_work_index_{algo}_simple.json"
        with open(sel_load, "r") as rf:
            sel_data = json.load(rf)

        for instance in sel_data[0].keys():
            name = algo

            if "num" in instance:
                continue

            # if "sto" in instance:
            #     name = f"{algo}_sto_s{re.findall(r'_s([0-9]+)_', instance)[0]}"
            # elif "num" in instance:
            #     name = f"{algo}_num_s{re.findall(r'_s([0-9]+)_', instance)[0]}"
            #
            # if "unique" in instance:
            #     name += "unique"
            # elif "duplicate" in instance:
            #     name += "duplicate"

            if name not in result.keys():
                result[name] = list()

            result[name].extend(get_res_data(sel_data, instance, data_id))

    return result


def ebar_query_plot(data_id, save_path=None, save_conf={"format": "pdf", "bbox_inches": "tight"}):
    # 1. Get the data.
    data = merge_query_result(data_id)
    if save_path is not None:
        with open(save_path.replace(".pdf", ".json"), "w") as wf:
            json.dump(data, wf, indent=2)

    # data_load = save_path.replace(".pdf", ".json")
    # with open(data_load, "r") as rf:
    #     data = json.load(rf)

    # 2. Define the figure.
    nrows, ncols = 2, 1
    figsize = (30, 30 // 3)
    p_fig = ErrorBarPlot(nrows, ncols, figsize)
    # p_fig.fig.subplots_adjust(wspace=3.92, hspace=0.22)

    # 3. Draw the subplot.
    groups = heu_algos
    labels = [""]

    # 3.1 set Bar properties.
    colors = [p_fig.color["grey"], p_fig.color["blue"], p_fig.color["green"],
              p_fig.color["orange"], p_fig.color["pink"], p_fig.color["red"],
              p_fig.color["purple"], p_fig.color["brown"], p_fig.color["yellow"]]
    hatches = ["/", "-", "",
               "xx", "||", "\\",
               "--", "|", "//"]

    gap, width = 0.25, 0.25

    box_conf = dict(patch_artist=True, showmeans=False, showfliers=False)
    tick_conf = {"labelsize": 13}
    leg_conf = {"loc": "upper center", "ncol": 9,
                "bbox_to_anchor": (.5, 1.3),
                "prop": {"size": 18, "weight": "normal"}}

    # 3.2 set X properties.
    xlims = None
    xticks = None
    xticklabels_conf = None  # {"labels": groups, "rotation": 15, "fontdict": {"fontsize": 13, "fontweight": "normal"}}

    xlabels = None
    xlabel_conf = None  # {"xlabel": xlabels, "fontdict": {"fontsize": "large", "fontweight": "bold"}}

    # 3.3 set Y properties.
    if data_id == "index_utility":
        ylims = (0., 1.)  # 0.0, 22.0
    else:
        ylims = None
    yticks = None
    yticklabels_conf = None

    ylabels = None
    ylabel_conf = None  # {"ylabel": ylabels, "fontdict": {"fontsize": "large", "fontweight": "bold"}}

    for rid in range(int(np.ceil(temp_num / row_qnum))):
        if rid == 0:
            leg_conf = {"loc": "upper center", "ncol": 9,
                        "bbox_to_anchor": (.5, 1.3),
                        "prop": {"size": 18, "weight": "normal"}}
        else:
            leg_conf = None

        groups = [f"Q{qno}" for qno in range(1, temp_num + 1) if qno != 15][rid * row_qnum:(rid + 1) * row_qnum]
        labels = list(data["1"].keys())

        bar_conf = [{"color": colors[no], "hatch": hatches[no], "edgecolor": "black",
                     "error_kw": dict(lw=2, capsize=3, capthick=1.5)} for no in range(len(labels))]

        query_datas = list()
        for algo in data["1"].keys():
            query_data = [np.mean(data[str(qno)][algo]) for qno in range(1, temp_num + 1) if qno != 15][
                         rid * row_qnum:(rid + 1) * row_qnum]
            yerr_data = [np.std(data[str(qno)][algo]) for qno in range(1, temp_num + 1) if qno != 15][
                        rid * row_qnum:(rid + 1) * row_qnum]

            query_datas.append({"query_data": query_data, "yerr_data": yerr_data})

        xticks = None
        xticklabels_conf = {"labels": groups, "fontdict": {"fontsize": 22, "fontweight": "bold"}}

        ax = p_fig.fig.add_subplot(p_fig.gs[rid])

        p_fig.ebar_sub_plot(query_datas, ax,
                            groups, labels, gap, width,
                            bar_conf=bar_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                            xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf, xlabel_conf=xlabel_conf,
                            ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf, ylabel_conf=ylabel_conf)

    if save_path is not None:
        plt.savefig(save_path, **save_conf)

    plt.show()


def box_query_plot(data_id, save_path=None, save_conf={"format": "pdf", "bbox_inches": "tight"}):
    # 1. Get the data.
    data = merge_query_result(data_id)

    # 2. Define the figure.
    nrows, ncols = 2, 1
    figsize = (30, 30 // 3)
    p_fig = BoxPlot(nrows, ncols, figsize)
    # p_fig.fig.subplots_adjust(wspace=3.92, hspace=0.22)

    # 3. Draw the subplot.
    groups = heu_algos
    labels = [""]

    # 3.1 set Bar properties.
    colors = [p_fig.color["grey"], p_fig.color["blue"], p_fig.color["green"],
              p_fig.color["orange"], p_fig.color["pink"], p_fig.color["red"],
              p_fig.color["purple"], p_fig.color["brown"], p_fig.color["yellow"]]
    hatches = ["/", "-", "",
               "xx", "||", "\\",
               "--", "|", "//"]

    gap, width = 0.25, 0.25

    box_conf = dict(patch_artist=True, showmeans=False, showfliers=False)
    tick_conf = {"labelsize": 13}
    leg_conf = {"loc": "upper center", "ncol": 9,
                "bbox_to_anchor": (.5, 1.3),
                "prop": {"size": 18, "weight": "normal"}}

    # 3.2 set X properties.
    xlims = None
    xticks = None
    xticklabels_conf = None  # {"labels": groups, "rotation": 15, "fontdict": {"fontsize": 13, "fontweight": "normal"}}

    xlabels = None
    xlabel_conf = None  # {"xlabel": xlabels, "fontdict": {"fontsize": "large", "fontweight": "bold"}}

    # 3.3 set Y properties.
    if data_id == "index_utility":
        ylims = (0., 1.)  # 0.0, 22.0
    else:
        ylims = None
    yticks = None
    yticklabels_conf = None

    ylabels = None
    ylabel_conf = None  # {"ylabel": ylabels, "fontdict": {"fontsize": "large", "fontweight": "bold"}}

    for rid in range(int(np.ceil(temp_num / row_qnum))):
        if rid == 0:
            leg_conf = {"loc": "upper center", "ncol": 9,
                        "bbox_to_anchor": (.5, 1.3),
                        "prop": {"size": 18, "weight": "normal"}}
        else:
            leg_conf = None

        groups = [f"Q{qno}" for qno in range(1, temp_num + 1) if qno != 15][rid * row_qnum:(rid + 1) * row_qnum]
        labels = list(data["1"].keys())

        bar_conf = [{"color": colors[no], "hatch": hatches[no], "edgecolor": "black",
                     "error_kw": dict(lw=2, capsize=3, capthick=1.5)} for no in range(len(labels))]

        query_datas = list()
        for algo in data["1"].keys():
            query_data = [np.mean(data[str(qno)][algo]) for qno in range(1, temp_num + 1) if qno != 15][
                         rid * row_qnum:(rid + 1) * row_qnum]
            yerr_data = [np.std(data[str(qno)][algo]) for qno in range(1, temp_num + 1) if qno != 15][
                        rid * row_qnum:(rid + 1) * row_qnum]

            query_datas.append({"query_data": query_data, "yerr_data": yerr_data})

        xticks = None
        xticklabels_conf = {"labels": groups, "fontdict": {"fontsize": 22, "fontweight": "bold"}}

        ax = p_fig.fig.add_subplot(p_fig.gs[rid])

        medianprops = [{"color": "red", "linewidth": 1.5} for _ in range(len(labels))]
        boxprops = [{"facecolor": colors[no], "edgecolor": "black",
                     "linewidth": 1., "hatch": hatches[no]} for no in range(len(labels))]
        whiskerprops = [{"color": "C0", "linewidth": 1.5} for _ in range(len(labels))]
        capprops = [{"color": "C0", "linewidth": 1.5} for _ in range(len(labels))]

        # p_fig.box_sub_plot(query_datas, ax,
        #                    groups, labels, gap, width,
        #                    medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops,
        #                    box_conf=box_conf, tick_conf=tick_conf, leg_conf=leg_conf,
        #                    xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf, xlabel_conf=xlabel_conf,
        #                    ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf, ylabel_conf=ylabel_conf)

        p_fig.box_sub_plot(query_datas, ax,
                           groups, labels, gap, width,
                           bar_conf=bar_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                           xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf, xlabel_conf=xlabel_conf,
                           ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf, ylabel_conf=ylabel_conf)

    if save_path is not None:
        with open(save_path.replace(".pdf", ".json"), "w") as wf:
            json.dump(query_datas, wf, indent=2)
        plt.savefig(save_path, **save_conf)

    plt.show()


def ebar_work_plot(data_id, save_path=None, save_conf={"format": "pdf", "bbox_inches": "tight"}):
    # 1. Get the data.
    data = merge_work_result(data_id)
    if save_path is not None:
        with open(save_path.replace(".pdf", ".json"), "w") as wf:
            json.dump(data, wf, indent=2)

    # data_load = save_path.replace(".pdf", ".json")
    # with open(data_load, "r") as rf:
    #     data = json.load(rf)

    # 2. Define the figure.
    nrows, ncols = 1, 1
    figsize = (20, 30 // 3)
    p_fig = ErrorBarPlot(nrows, ncols, figsize)
    # p_fig.fig.subplots_adjust(wspace=3.92, hspace=0.22)

    # 3. Draw the subplot.
    groups = heu_algos
    labels = [""]

    # 3.1 set Bar properties.
    colors = [p_fig.color["green"], p_fig.color["blue"], p_fig.color["grey"],
              p_fig.color["orange"], p_fig.color["pink"], p_fig.color["red"],
              p_fig.color["purple"], p_fig.color["brown"], p_fig.color["yellow"]]
    hatches = ["/", "-", "",
               "xx", "||", "\\",
               "--", "|", "//"]

    gap, width = 0.25, 0.2

    bar_conf = [{"color": colors[no], "hatch": hatches[no], "edgecolor": "black",
                 "error_kw": dict(lw=2, capsize=3, capthick=1.5)} for no in range(len(labels))]
    tick_conf = {"labelsize": 13}
    leg_conf = {"loc": "upper center", "ncol": 9,
                "bbox_to_anchor": (.5, 1.3),
                "prop": {"size": 18, "weight": "normal"}}
    leg_conf = None

    # 3.2 set X properties.
    xlims = None
    xticks = None
    xticklabels_conf = {"labels": groups, "rotation": 15, "fontdict": {"fontsize": 13, "fontweight": "normal"}}

    xlabels = None
    xlabel_conf = None  # {"xlabel": xlabels, "fontdict": {"fontsize": "large", "fontweight": "bold"}}

    # 3.3 set Y properties.
    if data_id == "index_utility":
        ylims = (0., 1.)  # 0.0, 22.0
    else:
        ylims = None
    yticks = None
    yticklabels_conf = None

    ylabels = None
    ylabel_conf = None  # {"ylabel": ylabels, "fontdict": {"fontsize": "large", "fontweight": "bold"}}

    query_datas = {"query_data": [np.mean(data[algo]) for algo in data.keys()],
                   "yerr_data": [np.std(data[algo]) for algo in data.keys()]}
    groups = list(data.keys())

    xticks = None
    xticklabels_conf = {"labels": groups, "fontdict": {"fontsize": 22, "fontweight": "bold"}}

    ax = p_fig.fig.add_subplot(p_fig.gs[0])

    p_fig.ebar_sub_plot(query_datas, ax,
                        groups, labels, gap, width,
                        bar_conf=bar_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                        xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf, xlabel_conf=xlabel_conf,
                        ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf, ylabel_conf=ylabel_conf)

    if save_path is not None:
        plt.savefig(save_path, **save_conf)

    plt.show()


if __name__ == "__main__":
    # ana_data_result()

    data_id = "index_utility"
    # merge_query_result(data_id)
    # merge_work_result(data_id)

    save_path = None
    for data_id in ["index_utility", "time_duration"]:
        save_path = f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch_skew/query_level/" \
                    f"tpch_skew_1gb_multi_query_bench_{data_id.replace(' ', '_')}_ebar.pdf"
        ebar_query_plot(data_id, save_path=save_path)

    save_path = None
    for data_id in ["index_utility", "time_duration"]:
        save_path = f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch_skew/work_level/" \
                    f"tpch_skew_1gb_multi_work_bench_{data_id.replace(' ', '_')}_ebar.pdf"
        ebar_work_plot(data_id, save_path=save_path)
