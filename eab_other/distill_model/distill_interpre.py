# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: distill_interpre
# @Author: Wei Zhou
# @Time: 2023/11/13 16:57

import json

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/data1/wz/index/")

from result_plot.fig_plot_all import FigurePlot

from index_eab.eab_other.distill_model.distill_model import XGBoost, LightGBM

algos = ["IG", "SL", "DL", "FP", "FA"]
benchmarks = ["TPC-H", "TPC-DS", "JOB"]

titles = ["(a) XGBoost", "(b) LightGBM"]

def get_feat_imp(model_type):
    if model_type == "XGBoost":
        model_loads = [
            "/data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/"
            "exp_xgb_tpch_round5k/model/reg_xgb_cost.xgb.model",
            "/data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/"
            "exp_xgb_tpcds_round5k/model/reg_xgb_cost.xgb.model",
            "/data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/"
            "exp_xgb_job_round5k/model/reg_xgb_cost.xgb.model"
        ]

        imp_four_total = list()
        for model_load in model_loads:
            model = XGBoost(path=model_load)

            importance_type = "weight"
            # importance_type = "gain"
            # importance_type = "cover"
            importance = model.model.get_score(importance_type=importance_type)

            # ["utility", "query_shape", "index_shape", "physical_operator"]
            imp_pre = [importance[f"f{i}"] if f"f{i}" in importance.keys() else 0 for i in range(55)]

            imp_four = list()
            # 1. utility (1)
            imp_four.append(float(imp_pre[0] / 1))
            imp_four.append(float(imp_pre[0] / 1))

            # 2. query_shape (6 * 7)
            imp_four.append(float(np.sum(imp_pre[1:1 + 6 * 7]) / 42))
            imp_four.append(float(np.sum(imp_pre[1:1 + 6 * 7]) / 42))

            # 3. index_shape (1 * 2)
            imp_four.append(float(np.sum(imp_pre[1 + 6 * 7:1 + 6 * 7 + 1 * 2]) / 2))
            imp_four.append(float(np.sum(imp_pre[1 + 6 * 7:1 + 6 * 7 + 1 * 2]) / 2))

            # 4. physical_operator (10)
            imp_four.append(float(np.sum(imp_pre[1 + 6 * 7 + 1 * 2:]) / 10))
            imp_four.append(float(np.sum(imp_pre[1 + 6 * 7 + 1 * 2:]) / 10))

            imp_four_total.append(imp_four)

    elif model_type == "LightGBM":
        model_loads = [
            "/data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/"
            "exp_lgb_tpch_round5k/model/reg_lgb_cost.lgb.model",
            "/data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/"
            "exp_lgb_tpcds_round5k/model/reg_lgb_cost.lgb.model",
            "/data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/"
            "exp_lgb_job_round5k/model/reg_lgb_cost.lgb.model"
        ]

        imp_four_total = list()
        for model_load in model_loads:
            model = LightGBM(model_load)

            importance = model.model.feature_importance()

            # ["utility", "query_shape", "index_shape", "physical_operator"]
            imp_pre = importance

            imp_four = list()
            # 1. utility (1)
            imp_four.append(float(imp_pre[0] / 1))
            imp_four.append(float(imp_pre[0] / 1))

            # 2. query_shape (6 * 7)
            imp_four.append(float(np.sum(imp_pre[1:1 + 6 * 7]) / 42))
            imp_four.append(float(np.sum(imp_pre[1:1 + 6 * 7]) / 42))

            # 3. index_shape (1 * 2)
            imp_four.append(float(np.sum(imp_pre[1 + 6 * 7:1 + 6 * 7 + 1 * 2]) / 2))
            imp_four.append(float(np.sum(imp_pre[1 + 6 * 7:1 + 6 * 7 + 1 * 2]) / 2))

            # 4. physical_operator (10)
            imp_four.append(float(np.sum(imp_pre[1 + 6 * 7 + 1 * 2:]) / 10))
            imp_four.append(float(np.sum(imp_pre[1 + 6 * 7 + 1 * 2:]) / 10))

            imp_four_total.append(imp_four)

    return imp_four_total


def multi_heat_plot(data, save_path=None,
                    save_conf={"format": "pdf", "bbox_inches": "tight"}):
    # 2. Define the figure.
    nrows, ncols = 1, 2
    figsize = (20, 30 // 3)
    # figsize = (27, 30 // 2.7)
    figsize = (27, 30 // 4)

    p_fig = FigurePlot(nrows, ncols, figsize)
    # p_fig.fig.subplots_adjust(wspace=3.92, hspace=0.22)
    p_fig.fig.subplots_adjust(wspace=.005, hspace=0.1)

    # 3. Draw the subplot.
    # 3.1 set Bar properties.
    # "#F0F0F0", "#F1F1F1", "#2A4458"
    heat_conf = {"cmap": "Blues"}  # viridis, Blues

    tick_conf = None
    # tick_conf = {"labelsize": 25, "pad": 20}
    tick_conf = {"pad": 20}

    leg_conf = None
    # leg_conf = {"loc": "upper center", "ncol": 5,
    #             "bbox_to_anchor": (.5, 1.25),
    #             "prop": {"size": 30, "weight": 510}}

    # 3.2 set X properties.
    xlims = None
    # xticks = [16 * i + 17 / 2 for i in range(4)]
    xticklabels_conf = None
    xticks = [i for i in range(4)]

    xlabels = None
    xlabel_conf = None  # {"xlabel": xlabels, "fontdict": {"fontsize": "large", "fontweight": "bold"}}

    # 3.3 set Y properties.
    # if data_id == "index_utility":
    #     ylims = (0., 60.)  # 0.0, 22.0
    # else:
    #     ylims = None
    ylims = None
    ylabel_conf = None

    yticks = None
    yticklabels_conf = None

    for no, dat in enumerate(data):
        if no == 0:
            heat_conf = {"cmap": "Greens", "aspect": .7}
        elif no == 1:
            heat_conf = {"cmap": "Blues", "aspect": .7}

        ax = p_fig.fig.add_subplot(p_fig.gs[no % nrows, no // nrows])
        ax.set_facecolor("#F3F3F3")

        yticks = [i for i in range(3)]

        if no % nrows == nrows - 1:
            xticks = [i for i in range(4)]

            features = ["1. Improvement", "2. Query Shape", "3. Index Shape", "4. Operator"]
            xticklabels_conf = {"labels": features, "rotation": 11,
                                "fontdict": {"fontsize": 30, "fontweight": "bold"}}
        else:
            xticks = list()
            # xticklabels_conf = {"labels": ["" for _ in xticks],
            #                     "fontdict": {"fontsize": 27, "fontweight": "bold"}}

        # if no % nrows == 2:
        #     xlabel_conf = {"xlabel": titles[no // nrows - 1],
        #                    "fontdict": {"fontsize": 30, "fontweight": "bold"}}
        # else:
        #     xlabel_conf = None

        xlabel_conf = {"xlabel": titles[no],
                       "fontdict": {"fontsize": 33, "fontweight": "bold"}}

        # if no // nrows == 0:
        #     ylabel_conf = {"ylabel": benchmarks[no % nrows],
        #                    "fontdict": {"fontsize": 27, "fontweight": "bold"}}
        # else:
        #     ylabel_conf = None

        if no == 0:
            yticklabels_conf = {"labels": benchmarks,
                                "fontdict": {"fontsize": 30, "fontweight": "bold"}}
        else:
            yticks = list()
            yticklabels_conf = {"labels": "",
                                "fontdict": {"fontsize": 30, "fontweight": "bold"}}

        p_fig.heatmap_sub_plot_1d(dat, ax,
                                  heat_conf=heat_conf, tick_conf=tick_conf, leg_conf=leg_conf,
                                  xlims=xlims, xticks=xticks, xticklabels_conf=xticklabels_conf,
                                  xlabel_conf=xlabel_conf,
                                  ylims=ylims, yticks=yticks, yticklabels_conf=yticklabels_conf,
                                  ylabel_conf=ylabel_conf)

    if save_path is not None:
        plt.savefig(save_path, **save_conf)

    plt.show()


def heat_plot(data, title):
    # 'viridis', Blues is a colormap, you can change it to any other colormap
    plt.imshow(data, cmap='Blues')
    # plt.imshow(data[0, :, :].detach().numpy(), cmap='Blues')
    plt.colorbar()  # Add a colorbar to the plot
    plt.title(title)  # Add a title to the plot
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Show the heatmap
    plt.show()


if __name__ == "__main__":
    save_path = None
    save_path = "/data1/wz/index/index_eab/eab_other/tree_filter_inter_heat.pdf"

    # XGBoost, LightGBM
    model_type = "XGBoost"
    # importance_xgb = get_feat_imp(model_type)
    # heat_plot(importance_xgb, model_type)

    model_type = "LightGBM"
    # importance_lgb = get_feat_imp(model_type)
    # heat_plot(importance_lgb, model_type)

    # data = [importance_xgb] + [importance_lgb]

    # with open(save_path.replace(".pdf", ".json"), "w") as wf:
    #     json.dump(data, wf, indent=2)

    with open(save_path.replace(".pdf", ".json"), "r") as rf:
        data = json.load(rf)

    data = [data[:3]] + [data[3:]]
    multi_heat_plot(data, save_path)
