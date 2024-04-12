# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: ana_bench_temp
# @Author: Wei Zhou
# @Time: 2023/8/15 15:06

import os
import json
import numpy as np

from index_eab.eab_algo.heu_selection.heu_algos.auto_admin_algorithm import AutoAdminAlgorithm
from index_eab.eab_algo.heu_selection.heu_algos.db2advis_algorithm import DB2AdvisAlgorithm
from index_eab.eab_algo.heu_selection.heu_algos.drop_heuristic_algorithm import DropHeuristicAlgorithm
from index_eab.eab_algo.heu_selection.heu_algos.extend_algorithm import ExtendAlgorithm
from index_eab.eab_algo.heu_selection.heu_algos.relaxation_algorithm import RelaxationAlgorithm
from index_eab.eab_algo.heu_selection.heu_algos.anytime_algorithm import AnytimeAlgorithm

excluded_qno = {"tpch": [20 - 1, 17 - 1, 18 - 1],
                "tpcds": [2, 29, 36, 56, 87, 89, 95,
                          3, 34, 55, 73,
                          21, 25, 16,
                          6, 39],
                "job": []}

ALGORITHMS = {
    "auto_admin": AutoAdminAlgorithm,
    "db2advis": DB2AdvisAlgorithm,
    "drop": DropHeuristicAlgorithm,
    "extend": ExtendAlgorithm,
    "relaxation": RelaxationAlgorithm,
    "anytime": AnytimeAlgorithm
}


def ana_multi_instance():
    data_loads = os.listdir("/data/wz/index/index_eab/eab_olap/bench_result/tpch/query_level")

    variance, std = dict(), dict()
    for data_load in data_loads:
        with open(data_load, "r") as rf:
            bench_res = json.load(rf)

        for qno in bench_res.keys():
            variance[qno], std[qno] = list(), list()
            for algo in ALGORITHMS.keys():
                data = [item[algo]["total_ind_cost"] / item[algo]["total_no_cost"] for item in bench_res[qno]]
                variance[qno].append(np.var(data))
                std[qno].append(np.std(data))
    pass


def ana_bench():
    # data_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_1gb_template_22_multi500_difference.json"
    # with open(data_load, "r") as rf:
    #     bench_data = json.load(rf)

    data_load = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_template_113.sql"
    with open(data_load, "r") as rf:
        bench_data = rf.readlines()

    data_load = "/data/wz/index/index_eab/eab_olap/bench_result/tpch/work_level/tpch_1gb_template_18_multi_work_index_swirl.json"
    # data_load = "/data/wz/index/index_eab/eab_olap/bench_result/tpch/work_level/tpch_1gb_template_18_multi_work_index_drlinda.json"
    # data_load = "/data/wz/index/index_eab/eab_olap/bench_result/tpch/work_level/tpch_1gb_template_18_multi_work_index_dqn.json"

    data_load = "/data/wz/index/index_eab/eab_olap/bench_result/tpch/work_level/tpch_1gb_template_18_multi_work_index0-20.json"

    with open(data_load, "r") as rf:
        bench_data = json.load(rf)

    models = bench_data[0].keys()

    res = dict()
    for m in models:
        res[m] = [1 - work[m]["total_ind_cost"] / work[m]["total_no_cost"] for work in bench_data]
    pass


if __name__ == "__main__":
    ana_bench()
    # ana_multi_instance()
