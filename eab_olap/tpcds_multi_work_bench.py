# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: tpcds_multi_work_bench
# @Author: Wei Zhou
# @Time: 2023/8/21 17:05

import os
import json
import configparser

from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("/data2/wz/index")
sys.path.append("/data2/wz/index/code_utils")
sys.path.append("/data2/wz/index/index_eab/eab_algo")
sys.path.append("/data2/wz/index/index_eab/eab_algo/swirl_selection")
sys.path.append("/data2/wz/index/index_eab/eab_algo/mcts_selection")
sys.path.append("/data2/wz/index/index_eab/eab_algo/mab_selection")

from index_eab.eab_algo.heu_selection.heu_utils import selec_com
from index_eab.eab_algo.swirl_selection.swirl_utils import swirl_com
from index_eab.eab_algo.swirl_selection.swirl_run import get_swirl_res, pre_infer_obj
from index_eab.eab_algo.heu_selection.heu_run import get_heu_result, IndexEncoder

from index_eab.eab_algo.mcts_selection.mcts_run import get_mcts_res, MCTSEncoder
from mcts_selection.mcts_utils import mcts_com
from index_eab.eab_algo.mab_selection.shared import mab_com
from index_eab.eab_algo.mab_selection.mab_run import get_mab_res

heu_algos = ["extend", "db2advis", "relaxation", "anytime", "auto_admin", "drop"]
rl_agents = {"swirl": list(), "drlinda": list(), "dqn": list()}


def get_heu_multi_work_res():
    parser = selec_com.get_parser()
    args = parser.parse_args()

    args.algo = "drop"
    args.algo = [algo for algo in heu_algos if algo != "relaxation"]
    args.algo = ["db2advis"]

    # args.algo = ["extend", "db2advis", "relaxation", "anytime", "auto_admin", "drop"]

    gap, start, end = 10, 5, 6
    args.work_file = "/data2/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_work_temp_multi_w79_n100_test.json"
    args.res_save = f"/data2/wz/index/index_eab/eab_olap/bench_result/tpcds/work_level/" \
                    f"tpcds_1gb_template_79_multi_work_index_relaxation{start * gap + 8}-{end * gap}.json"

    # args.res_save = f"/data2/wz/index/index_eab/eab_olap/bench_result/tpcds/work_level/" \
    #                 f"tpcds_1gb_template_79_multi_work_index_extend.json"

    args.work_file = "/data2/wz/index/index_eab/eab_olap/bench_drift/tpcds/tpcds_work_drift_multi_w79_n10_eval.json"
    args.res_save = "/data2/wz/index/index_eab/eab_olap/bench_drift/tpcds/tpcds_work_drift_multi_w79_n10_eval_db2advis_v2.json"

    # args.work_file = "/data2/wz/index/index_eab/eab_olap/bench_drift/tpcds/tpcds_work_drift_multi_w79_n100_test.json"
    # args.res_save = "/data2/wz/index/index_eab/eab_olap/bench_result/tpcds/" \
    #                 "work_drift/tpcds_work_drift_multi_w79_n100_test_index.json"
    #
    # args.work_file = "/data2/wz/index/index_eab/eab_olap/bench_random/tpcds/tpcds_work_multi_w79_n100_test.json"
    # args.res_save = "/data2/wz/index/index_eab/eab_olap/bench_result/tpcds/" \
    #                 "work_random/tpcds_work_multi_w79_n100_test_index.json"

    args.is_utilized = True

    args.process, args.overhead = False, True
    args.sel_params = "parameters"
    args.exp_file = "/data2/wz/index/index_eab/eab_data/heu_run_conf/{}_config_tpch.json"

    args.constraint = "storage"
    args.budget_MB = 500

    # args.constraint = "number"
    args.max_indexes = 5

    args.db_conf_file = "/data2/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf"
    args.schema_file = "/data2/wz/index/index_eab/eab_data/db_info_conf/schema_tpcds_1gb.json"

    args.host = "10.26.42.166"
    args.port = "5432"
    args.db_name = "tpcds_1gb103"
    args.user = "wz"
    args.password = "ai4db2021"

    with open(args.work_file, "r") as rf:
        query_list = json.load(rf)
        # query_list = json.load(rf)[start * gap + 8:end * gap]

    # data_load = "/data2/wz/index/index_eab/eab_olap/bench_temp/tpcds_template_79.sql"
    # with open(data_load, "r") as rf:
    #     query_list = rf.readlines()
    # query_list = [query_list]

    res_data = list()
    for query in tqdm(query_list):
        # workload = [info[1] for info in query]
        workload = query
        res_data.append(get_heu_result(args, args.algo, workload))

        if args.res_save is not None:
            if not os.path.exists(os.path.dirname(args.res_save)):
                os.makedirs(os.path.dirname(args.res_save))
            with open(args.res_save, "w") as wf:
                json.dump(res_data, wf, indent=2, cls=IndexEncoder)


def get_heu_multi_work_cand_res():
    parser = selec_com.get_parser()
    args = parser.parse_args()

    args.algo = ["anytime", "relaxation", "db2advis"]

    gap, start, end = 10, 0, 1
    args.work_file = "/data2/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_work_temp_multi_w79_n100_test.json"
    args.res_save = f"/data2/wz/index/index_eab/eab_olap/bench_result/tpcds/work_level/" \
                    f"tpcds_1gb_template_79_multi_work_cand_index_relaxation{start * gap}-{end * gap}.json"

    # args.res_save = f"/data2/wz/index/index_eab/eab_olap/bench_result/tpcds/work_level/" \
    #                 f"tpcds_1gb_template_79_multi_work_index_extend.json"

    for method in ["openGauss"]:  # "permutation", "dqn_rule", "openGauss"
        for utilized in [True]:  # True, False
            args.res_save = f"/data2/wz/index/index_eab/eab_olap/bench_result/tpcds/work_cand/" \
                            f"tpcds_1gb_template_79_multi_work_cand_{method}_{str(utilized)}index{start * gap}-{end * gap}.json"

            args.cand_gen = method
            args.is_utilized = utilized

            args.process, args.overhead = False, True
            args.sel_params = "parameters"
            args.exp_conf_file = "/data2/wz/index/index_eab/eab_data/heu_run_conf/{}_config_tpch.json"

            args.constraint = "storage"
            args.budget_MB = 500

            # args.constraint = "number"
            args.max_indexes = 5

            args.db_conf_file = "/data2/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf"
            args.schema_file = "/data2/wz/index/index_eab/eab_data/db_info_conf/schema_tpcds_1gb.json"

            with open(args.work_file, "r") as rf:
                query_list = json.load(rf)[start * gap:end * gap]

            # data_load = "/data2/wz/index/index_eab/eab_olap/bench_temp/tpcds_template_79.sql"
            # with open(data_load, "r") as rf:
            #     query_list = rf.readlines()
            # query_list = [query_list]

            res_data = list()
            for query in tqdm(query_list):
                # workload = [info[1] for info in query]
                workload = query
                res_data.append(get_heu_result(args, args.algo, workload))

                if args.res_save is not None:
                    if not os.path.exists(os.path.dirname(args.res_save)):
                        os.makedirs(os.path.dirname(args.res_save))
                    with open(args.res_save, "w") as wf:
                        json.dump(res_data, wf, indent=2, cls=IndexEncoder)


def get_swirl_multi_work_res(exp_id, res_id):
    parser = swirl_com.get_parser()
    args = parser.parse_args()

    args.seed = 666
    args.max_budgets = 500
    # args.max_budgets = 5000

    args.db_conf_file = "/data2/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf"

    args.work_file = "/data2/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_work_temp_multi_w79_n100_test.json"

    # args.work_file = "/data2/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_work_temp_multi_w79_permutation_n100_test.json"

    # args.work_file = "/data2/wz/index/index_eab/eab_olap/bench_random/tpcds/tpcds_work_multi_w79_n100_test.json"
    # args.work_file = "/data2/wz/index/index_eab/eab_olap/bench_drift/tpcds/tpcds_work_drift_multi_w79_n100_test.json"

    with open(args.work_file, "r") as rf:
        work_list = json.load(rf)

    exp_dir = f"/data2/wz/index/index_eab/eab_algo/swirl_selection/{exp_id}"

    rl_agents["swirl"] = [exp for exp in os.listdir(exp_dir) if "swirl" in exp and "w79" in exp]
    rl_agents["drlinda"] = [exp for exp in os.listdir(exp_dir) if "drlinda" in exp and "w79" in exp]
    rl_agents["dqn"] = [exp for exp in os.listdir(exp_dir) if "dqn" in exp and "w79" in exp]

    # for agent in rl_agents.keys():
    for agent in ["dqn"]:
        res = dict()
        for instance in rl_agents[agent]:
            res[instance] = list()

            args.rl_exp_load = f"{exp_dir}/{instance}/experiment_object.pickle"
            args.rl_model_load = f"{exp_dir}/{instance}/best_mean_reward_model.zip"
            args.rl_env_load = f"{exp_dir}/{instance}/vec_normalize.pkl"

            # db_conf = None

            db_conf = configparser.ConfigParser()
            db_conf.read(args.db_conf_file)

            db_conf["postgresql"]["host"] = "10.26.42.166"
            db_conf["postgresql"]["database"] = "tpcds_1gb103"
            db_conf["postgresql"]["port"] = "5432"
            db_conf["postgresql"]["user"] = "wz"
            db_conf["postgresql"]["password"] = "ai4db2021"

            swirl_exp, swirl_model = pre_infer_obj(args.rl_exp_load, args.rl_model_load,
                                                   args.rl_env_load, db_conf=db_conf)
            for work in tqdm(work_list):
                # workload = [info[1] for info in work]
                res[instance].append(get_swirl_res(args, work, swirl_exp, swirl_model))

        res_data = list()
        for i in range(len(res[instance])):
            r = dict()
            for instance in rl_agents[agent]:
                r[instance] = res[instance][i]
            res_data.append(r)

        args.res_save = f"/data2/wz/index/index_eab/eab_olap/bench_result/tpcds/{res_id}/" \
                        f"tpcds_10-1gb_template_79_multi_{res_id}_index_{agent}.json"
        if args.res_save is not None:
            if not os.path.exists(os.path.dirname(args.res_save)):
                os.makedirs(os.path.dirname(args.res_save))
            with open(args.res_save, "w") as wf:
                json.dump(res_data, wf, indent=2, cls=IndexEncoder)


def get_swirl_work_res_state_mask(exp_id, res_id):
    """
    work_embed: obs[:, 54: 54 + 50 * 18] = np.zeros((1, 50 * 18))
    query_cost: obs[:, 54 + 50 * 18: 54 + 50 * 18 + 18] = np.zeros((1, 18))
    query_freq: obs[:, 54 + 50 * 18 + 18: 54 + 50 * 18 + 18 + 18] = np.zeros((1, 18))
    storage_budget: obs[:, 54 + 50 * 18 + 18 + 18: 54 + 50 * 18 + 18 + 18 + 1] = np.zeros((1, 1))
    storage_cons: obs[:, 54 + 50 * 18 + 18 + 18 + 1: 54 + 50 * 18 + 18 + 18 + 1 + 1] = np.zeros((1, 1))
    init_cost: obs[:, 54 + 50 * 18 + 18 + 18 + 1 + 1: 54 + 50 * 18 + 18 + 18 + 1 + 1 + 1] = np.zeros((1, 1))
    curr_cost: obs[:, 54 + 50 * 18 + 18 + 18 + 1 + 1 + 1: 54 + 50 * 18 + 18 + 18 + 1 + 1 + 1 + 1] = np.zeros((1, 1))

    :param exp_id:
    :param res_id:
    :return:
    """
    parser = swirl_com.get_parser()
    args = parser.parse_args()

    args.seed = 666
    args.max_budgets = 500

    args.db_conf_file = "/data2/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf"

    args.work_file = "/data2/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_work_temp_multi_w79_n100_test.json"
    with open(args.work_file, "r") as rf:
        work_list = json.load(rf)

    exp_dir = f"/data2/wz/index/index_eab/eab_algo/swirl_selection/{exp_id}"
    rl_agents["swirl"] = [exp for exp in os.listdir(exp_dir) if "swirl" in exp and "w79" in exp]
    rl_agents["drlinda"] = [exp for exp in os.listdir(exp_dir) if "drlinda" in exp and "w79" in exp]
    rl_agents["dqn"] = [exp for exp in os.listdir(exp_dir) if "dqn" in exp and "w79" in exp]

    # for agent in rl_agents.keys():
    for agent in ["dqn"]:
        res = dict()
        for instance in rl_agents[agent]:
            res[instance] = list()

            args.rl_exp_load = f"{exp_dir}/{instance}/experiment_object.pickle"
            args.rl_model_load = f"{exp_dir}/{instance}/best_mean_reward_model.zip"
            args.rl_env_load = f"{exp_dir}/{instance}/vec_normalize.pkl"

            # db_conf = None

            db_conf = configparser.ConfigParser()
            db_conf.read(args.db_conf_file)

            db_conf["postgresql"]["host"] = "10.26.42.166"
            db_conf["postgresql"]["database"] = "tpcds_1gb103"
            db_conf["postgresql"]["port"] = "5432"
            db_conf["postgresql"]["user"] = "wz"
            db_conf["postgresql"]["password"] = "ai4db2021"

            swirl_exp, swirl_model = pre_infer_obj(args.rl_exp_load, args.rl_model_load,
                                                   args.rl_env_load, db_conf=db_conf)
            for work in tqdm(work_list):
                # workload = [info[1] for info in work]
                res[instance].append(get_swirl_res(args, work, swirl_exp, swirl_model))

        res_data = list()
        for i in range(len(res[instance])):
            r = dict()
            for instance in rl_agents[agent]:
                r[instance] = res[instance][i]
            res_data.append(r)

        args.res_save = f"/data2/wz/index/index_eab/eab_olap/bench_result/tpcds/{res_id}/" \
                        f"tpcds_1gb_template_79_multi_{res_id}_index_curr_cost_v3_{agent}.json"
        if args.res_save is not None:
            if not os.path.exists(os.path.dirname(args.res_save)):
                os.makedirs(os.path.dirname(args.res_save))
            with open(args.res_save, "w") as wf:
                json.dump(res_data, wf, indent=2, cls=IndexEncoder)


def get_mcts_multi_work_res():
    parser = mcts_com.get_parser()
    args = parser.parse_args()

    gap, start, end = 20, 0, 1
    args.work_file = "/data2/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_work_temp_multi_w79_n100_test.json"
    args.res_save = f"/data2/wz/index/index_eab/eab_olap/bench_result/tpcds/work_level/" \
                    f"tpcds_1gb_template_79_multi_work_index_mcts.json"

    # args.work_file = "/data2/wz/index/index_eab/eab_olap/bench_random/tpcds/tpcds_work_multi_w18_n100_test.json"
    # args.res_save = "/data2/wz/index/index_eab/eab_olap/bench_result/tpcds/work_random/" \
    #                 "tpch_work_random_multi_w18_n100_test_index_mcts.json"

    # args.is_utilized = True

    args.is_trace = True

    args.process, args.overhead = True, True
    args.sel_params = "parameters"
    args.exp_conf_file = "/data2/wz/index/index_eab/eab_data/heu_run_conf/{}_config_tpch.json"

    args.budget = 1000

    args.constraint = "storage"
    args.storage = 500

    # args.constraint = "number"
    args.cardinality = 5

    args.select_policy = "UCT"
    args.roll_num = 1
    args.best_policy = "BCE"

    args.db_file = "/data2/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf"
    args.schema_file = "/data2/wz/index/index_eab/eab_data/db_info_conf/schema_tpcds_1gb.json"

    with open(args.work_file, "r") as rf:
        query_list = json.load(rf)
        # query_list = json.load(rf)[start * gap:end * gap]

    res_data = list()
    for query in tqdm(query_list):
        # workload = [info[1].replace(" OR ", " AND ") for info in query]
        workload = query
        res_data.append(get_mcts_res(args, workload))

        if args.res_save is not None:
            if not os.path.exists(os.path.dirname(args.res_save)):
                os.makedirs(os.path.dirname(args.res_save))
            with open(args.res_save, "w") as wf:
                json.dump(res_data, wf, indent=2, cls=MCTSEncoder)


def get_mab_multi_work_res():
    parser = mab_com.get_parser()
    args = parser.parse_args()

    # logging.disable(logging.CRITICAL)

    gap, start, end = 20, 0, 1
    args.work_file = "/data2/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_work_temp_multi_w79_n100_test.json"
    args.res_save = f"/data2/wz/index/index_eab/eab_olap/bench_result/tpcds/work_level/" \
                    f"tpcds_1gb_template_79_multi_work_index_mab.json"

    # args.varying_frequencies = True
    # args.work_file = "/data2/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_freq_n100_test.json"
    # args.res_save = f"/data2/wz/index/index_eab/eab_olap/bench_result/tpch/work_freq/" \
    #                 f"tpch_1gb_template_18_multi_work_freq_index{start * gap}-{end * gap}.json"

    # args.work_file = "/data2/wz/index/index_eab/eab_olap/bench_random/tpch/tpch_work_multi_w18_n100_test.json"
    # args.res_save = "/data2/wz/index/index_eab/eab_olap/bench_result/tpch/work_random/" \
    #                 "tpch_work_random_multi_w18_n100_test_index_mab.json"

    args.is_utilized = True

    args.process = True
    args.sel_params = "parameters"
    args.exp_file = "/data2/wz/index/index_eab/eab_algo/mab_selection/config/exp.conf"

    args.bench = "tpcds"
    args.rounds = 1000

    args.constraint = "storage"
    args.max_memory = 500

    # args.constraint = "number"
    args.max_count = 5

    args.db_file = "/data2/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf"
    args.schema_file = "/data2/wz/index/index_eab/eab_data/db_info_conf/schema_tpcds_1gb.json"

    # args.db_name = "tpch_10gb103"
    with open(args.work_file, "r") as rf:
        query_list = json.load(rf)
        # query_list = json.load(rf)[start * gap:end * gap]

    res_data = list()
    for query in tqdm(query_list):
        # workload = [info[1].replace(" OR ", " AND ") for info in query]
        workload = query
        res_data.append(get_mab_res(args, workload))

        if args.res_save is not None:
            if not os.path.exists(os.path.dirname(args.res_save)):
                os.makedirs(os.path.dirname(args.res_save))
            with open(args.res_save, "w") as wf:
                json.dump(res_data, wf, indent=2, cls=IndexEncoder)


def remove_step_info(res_id):
    res_dir = f"/data2/wz/index/index_eab/eab_olap/bench_result/tpcds/{res_id}"
    for file in tqdm(os.listdir(res_dir)):
        # if "tpcds_1gb_template_79" not in file:
        #     continue

        if os.path.exists(f"{res_dir}_simple/{file.replace('.json', '_simple.json')}"):
            continue

        try:
            with open(f"{res_dir}/{file}", "r") as rf:
                res_data = json.load(rf)

            for item in res_data:
                for algo in item.keys():
                    if "step" in item[algo]["sel_info"].keys():
                        item[algo]["sel_info"].pop("step")

            save_path = f"{res_dir}_simple/{file.replace('.json', '_simple.json')}"
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            with open(save_path, "w") as wf:
                json.dump(res_data, wf, indent=2)
        except:
            print(file)


if __name__ == "__main__":
    # get_heu_multi_work_res()
    # get_heu_multi_work_cand_res()

    # exp_res_main, exp_res_num, exp_res_cand
    # exp_res_vol, exp_res_action
    # exp_res_random, exp_res_drift
    exp_id = "exp_res_vol"

    # work_cand, work_vol, work_action, work_permutation
    # work_shift, work_random, work_drift
    res_id = "work_shift"
    # get_swirl_multi_work_res(exp_id, res_id)

    exp_id = "exp_res_main"
    res_id = "work_mask"
    # get_swirl_work_res_state_mask(exp_id, res_id)

    # get_mcts_multi_work_res()
    get_mab_multi_work_res()

    # work_cand, work_shift, work_vol, work_permutation, work_mask
    res_id = "work_mask"
    # remove_step_info(res_id)
