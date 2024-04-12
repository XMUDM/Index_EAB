# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: ana_cand_bench_temp
# @Author: Wei Zhou
# @Time: 2023/11/4 16:30

import os
import json
import configparser

import numpy as np
from tqdm import tqdm

import sys

sys.path.append("/data/wz/index")
sys.path.append("/data/wz/index/code_utils")
sys.path.append("/data/wz/index/index_eab/eab_algo")
sys.path.append("/data/wz/index/index_eab/eab_algo/swirl_selection")
sys.path.append("/data/wz/index/index_eab/eab_algo/mcts_selection")
sys.path.append("/data/wz/index/index_eab/eab_algo/mab_selection")

from index_eab.eab_algo.heu_selection.heu_utils import selec_com
from index_eab.eab_algo.swirl_selection.swirl_utils import swirl_com

from index_eab.eab_algo.heu_selection.heu_run import get_heu_result, IndexEncoder
from index_eab.eab_algo.swirl_selection.swirl_run import get_swirl_res, pre_infer_obj

from index_eab.eab_algo.mcts_selection.mcts_utils import mcts_com
from index_eab.eab_algo.mcts_selection.mcts_run import get_mcts_res, MCTSEncoder

from index_eab.eab_algo.mab_selection.shared import mab_com
from index_eab.eab_algo.mab_selection.mab_run import get_mab_res

heu_algos = ["extend", "db2advis", "relaxation", "anytime", "auto_admin", "drop"]
rl_agents = {"swirl": list(), "drlinda": list(), "dqn": list()}


def get_heu_cand_num():
    parser = selec_com.get_parser()
    args = parser.parse_args()

    args.algos = ["db2advis", "relaxation", "anytime"]
    args.algos = ["anytime"]

    gap, start, end = 20, 0, 1
    args.work_file = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_n100_test.json"
    # args.work_file = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_work_temp_multi_w33_n100_test.json"

    res_data = dict()
    for algo in args.algos:
        res_data[algo] = dict()
        for method in ["permutation", "dqn_rule", "openGauss"]:  # "permutation", "dqn_rule", "openGauss"
            res_data[algo][method] = list()
            for utilized in [True]:  # True, False
                args.res_save = f"/data/wz/index/index_eab/eab_olap/bench_result/tpch/work_cand/" \
                                f"tpch_1gb_template_18_multi_work_cand_{str(utilized)}_cand_{algo}_detail.json"
                # args.res_save = f"/data/wz/index/index_eab/eab_olap/bench_result/job/work_cand/" \
                #                 f"job_template_33_multi_work_cand_{str(utilized)}_cand_{algo}_detail.json"

                args.cand_gen = method
                args.is_utilized = utilized

                args.process, args.overhead = True, True
                args.sel_params = "parameters"
                args.exp_conf_file = "/data/wz/index/index_eab/eab_data/heu_run_conf/{}_config_tpch.json"

                args.constraint = "storage"
                args.budget_MB = 500

                # args.constraint = "number"
                args.max_indexes = 5

                args.db_conf_file = "/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf"
                args.schema_file = "/data/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json"

                # args.db_conf_file = "/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_job.conf"
                # args.schema_file = "/data/wz/index/index_eab/eab_data/db_info_conf/schema_job.json"

                with open(args.work_file, "r") as rf:
                    query_list = json.load(rf)
                    # query_list = json.load(rf)[start * gap:end * gap]

                for query in tqdm(query_list):
                    workload = query
                    res_data[algo][method].append(get_heu_result(args, [algo], workload))

                    if args.res_save is not None:
                        if not os.path.exists(os.path.dirname(args.res_save)):
                            os.makedirs(os.path.dirname(args.res_save))
                        with open(args.res_save, "w") as wf:
                            json.dump(res_data, wf, indent=2, cls=IndexEncoder)


def get_swirl_cand_num():
    parser = swirl_com.get_parser()
    args = parser.parse_args()

    args.seed = 666
    args.max_budgets = 500

    # exp_dir = "/data/wz/index/index_eab/eab_algo/swirl_selection/exp_res_cand"
    # rl_agents["swirl"] = [exp for exp in os.listdir(exp_dir) if "swirl" in exp and "w18" in exp]
    # rl_agents["dqn"] = [exp for exp in os.listdir(exp_dir) if "dqn" in exp and "w18" in exp]

    exp_dir = "/data/wz/index/index_eab/eab_algo/swirl_selection/exp_res_cand_job"
    rl_agents["swirl"] = [exp for exp in os.listdir(exp_dir) if "swirl" in exp and "w33" in exp]
    rl_agents["dqn"] = [exp for exp in os.listdir(exp_dir) if "dqn" in exp and "w33" in exp]

    res_data = dict()
    # for agent in rl_agents.keys():
    for agent in ["swirl", "dqn"]:
        res_data[agent] = dict()
        # for gen_method in ["permutation", "dqn_rule", "openGauss"]:
        for gen_method in ["openGauss"]:
            for instance in rl_agents[agent]:
                if gen_method in instance:
                    args.rl_exp_load = f"{exp_dir}/{instance}/experiment_object.pickle"
                    args.rl_model_load = f"{exp_dir}/{instance}/best_mean_reward_model.zip"
                    args.rl_env_load = f"{exp_dir}/{instance}/vec_normalize.pkl"

                    db_conf = None

                    swirl_exp, swirl_model = pre_infer_obj(args.rl_exp_load, args.rl_model_load,
                                                           args.rl_env_load, db_conf=db_conf)

                    res_data[agent][gen_method] = swirl_exp.globally_index_candidates_flat

                    break

            # args.res_save = f"/data/wz/index/index_eab/eab_olap/bench_result/tpch/work_cand/" \
            #                 f"tpch_1gb_template_18_multi_work_cand_rl_cand_detail.json"

            args.res_save = f"/data/wz/index/index_eab/eab_olap/bench_result/job/work_cand/" \
                            f"job_template_33_multi_work_cand_rl_cand_detail_v2.json"

            if args.res_save is not None:
                if not os.path.exists(os.path.dirname(args.res_save)):
                    os.makedirs(os.path.dirname(args.res_save))
                with open(args.res_save, "w") as wf:
                    json.dump(res_data, wf, indent=2, cls=IndexEncoder)


def get_cand_data():
    for algo in ["db2advis", "relaxation", "anytime"]:
        cand_load = f"/data/wz/index/index_eab/eab_olap/bench_result/tpcds/work_cand/" \
                    f"tpcds_1gb_template_79_multi_work_cand_True_cand_{algo}_detail.json"
        with open(cand_load, "r") as rf:
            cand_data = json.load(rf)[algo]
        print("\t".join(map(str, [int(np.mean([len(it[1]) for it in item])) for item in cand_data.values()])))

    for algo in ["db2advis", "relaxation", "anytime"]:
        cand_load1 = f"/data/wz/index/index_eab/eab_olap/bench_result/job/work_cand/" \
                     f"job_template_33_multi_work_cand_True_cand_{algo}_detail.json"
        with open(cand_load1, "r") as rf:
            cand_data1 = json.load(rf)[algo]

        cand_load2 = f"/data/wz/index/index_eab/eab_olap/bench_result/job/work_cand/" \
                     f"job_template_33_multi_work_cand_True_cand_{algo}_detail_v2.json"
        with open(cand_load2, "r") as rf:
            cand_data2 = json.load(rf)[algo]

        print("\t".join(map(str, [int(np.mean([len(it[1]) for it in item])) for item in cand_data1.values()] +
                            [int(np.mean([len(it[1]) for it in item])) for item in cand_data2.values()])))

    for algo in ["db2advis", "relaxation", "anytime"]:
        cand_load = f"/data/wz/index/index_eab/eab_olap/bench_result/tpch/work_cand/" \
                    f"tpch_1gb_template_18_multi_work_cand_True_cand_{algo}_detail.json"
        with open(cand_load, "r") as rf:
            cand_data = json.load(rf)[algo]
        print("\t".join(map(str, [int(np.mean([len(it[1]) for it in item])) for item in cand_data.values()])))

    cand_load = "/data/wz/index/index_eab/eab_olap/bench_result/tpch/work_cand/tpch_1gb_template_18_multi_work_cand_rl_cand_detail.json"
    with open(cand_load, "r") as rf:
        cand_data = json.load(rf)

    for agent in cand_data:
        print("\t".join(map(str, [len(item) for item in cand_data[agent].values()])))


if __name__ == "__main__":
    # get_heu_cand_num()
    # get_swirl_cand_num()

    get_cand_data()
