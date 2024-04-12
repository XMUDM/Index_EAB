# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: dsb_multi_work_bench
# @Author: Wei Zhou
# @Time: 2023/8/21 17:05

import os
import json
import configparser

from tqdm import tqdm
import matplotlib.pyplot as plt

import sys

sys.path.append("/data1/wz/index")
sys.path.append("/data1/wz/index/code_utils")
sys.path.append("/data1/wz/index/index_eab/eab_algo")
sys.path.append("/data1/wz/index/index_eab/eab_algo/swirl_selection")
sys.path.append("/data1/wz/index/index_eab/eab_algo/mcts_selection")
sys.path.append("/data1/wz/index/index_eab/eab_algo/mab_selection")

from index_eab.eab_algo.heu_selection.heu_utils import selec_com
from index_eab.eab_algo.heu_selection.heu_run import get_heu_result, IndexEncoder

from index_eab.eab_algo.swirl_selection.swirl_utils import swirl_com
from index_eab.eab_algo.swirl_selection.swirl_run import get_swirl_res, pre_infer_obj

from index_eab.eab_algo.mcts_selection.mcts_utils import mcts_com
from index_eab.eab_algo.mcts_selection.mcts_run import get_mcts_res, MCTSEncoder

from index_eab.eab_algo.mab_selection.shared import mab_com
from index_eab.eab_algo.mab_selection.mab_run import get_mab_res

heu_algos = ["extend", "db2advis", "relaxation", "anytime", "auto_admin", "drop"]
rl_agents = {"swirl": list(), "drlinda": list(), "dqn": list()}


def get_heu_multi_work_res():
    parser = selec_com.get_parser()
    args = parser.parse_args()

    gap, start, end = 20, 0, 1
    args.work_file = "/data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_work_temp_multi_w53_n100_test.json"
    args.res_save = f"/data1/wz/index/index_eab/eab_olap/bench_result/dsb/work_level/" \
                    f"dsb_1gb_template_53_multi_work_index{start * gap}-{end * gap}.json"

    args.process, args.overhead = True, True
    args.sel_params = "parameters"
    args.exp_conf_file = "/data1/wz/index/index_eab/eab_data/heu_run_conf/{}_config_tpch.json"

    args.constraint = "storage"
    args.budget_MB = 500

    # args.constraint = "number"
    args.max_indexes = 5

    args.db_conf_file = "/data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf"
    args.schema_file = "/data1/wz/index/index_eab/eab_data/db_info_conf/schema_tpcds_1gb.json"

    args.host = "10.26.42.166"
    args.db_name = "dsb_1gb103"
    args.port = "5432"
    args.user = "wz"
    args.password = "ai4db2021"

    with open(args.work_file, "r") as rf:
        query_list = json.load(rf)[start * gap:end * gap]

    res_data = list()
    for query in tqdm(query_list):
        workload = [info[1] for info in query]
        res_data.append(get_heu_result(args, heu_algos, workload))

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

    args.db_conf_file = "/data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf"

    args.work_file = "/data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_work_temp_multi_w53_n100_test.json"
    args.work_file = "/data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_work_temp_multi_w53_permutation_n100_test.json"
    with open(args.work_file, "r") as rf:
        work_list = json.load(rf)

    exp_dir = f"/data1/wz/index/index_eab/eab_algo/swirl_selection/{exp_id}"

    rl_agents["swirl"] = [exp for exp in os.listdir(exp_dir) if "swirl" in exp and "w53" in exp]
    rl_agents["drlinda"] = [exp for exp in os.listdir(exp_dir) if "drlinda" in exp and "w53" in exp]
    rl_agents["dqn"] = [exp for exp in os.listdir(exp_dir) if "dqn" in exp and "w53" in exp]

    # for agent in rl_agents.keys():
    for agent in ["dqn"]:
        res = dict()
        for instance in rl_agents[agent]:
            res[instance] = list()

            args.rl_exp_load = f"{exp_dir}/{instance}/experiment_object.pickle"
            args.rl_model_load = f"{exp_dir}/{instance}/best_mean_reward_model.zip"
            args.rl_env_load = f"{exp_dir}/{instance}/vec_normalize.pkl"

            db_conf = None

            # db_conf = configparser.ConfigParser()
            # db_conf.read(args.db_conf_file)
            # db_conf["postgresql"]["host"] = "10.26.42.166"
            # db_conf["postgresql"]["database"] = "dsb_1gb103"
            # db_conf["postgresql"]["port"] = "5432"
            # db_conf["postgresql"]["user"] = "wz"
            # db_conf["postgresql"]["password"] = "ai4db2021"

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

        args.res_save = f"/data1/wz/index/index_eab/eab_olap/bench_result/dsb/{res_id}/" \
                        f"dsb_1gb_template_53_multi_{res_id}_index_{agent}.json"
        if args.res_save is not None:
            if not os.path.exists(os.path.dirname(args.res_save)):
                os.makedirs(os.path.dirname(args.res_save))
            with open(args.res_save, "w") as wf:
                json.dump(res_data, wf, indent=2, cls=IndexEncoder)


def get_swirl_multi_work_res_data_shift(exp_id, res_id):
    parser = swirl_com.get_parser()
    args = parser.parse_args()

    args.seed = 666
    args.max_budgets = 500
    # args.max_budgets = 5000

    args.db_conf_file = "/data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf"

    args.work_file = "/data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_work_temp_multi_w53_n100_test.json"
    with open(args.work_file, "r") as rf:
        work_list = json.load(rf)

    exp_dir = f"/data1/wz/index/index_eab/eab_algo/swirl_selection/{exp_id}"

    rl_agents["swirl"] = [exp for exp in os.listdir(exp_dir) if "swirl" in exp and "w53" in exp]
    rl_agents["drlinda"] = [exp for exp in os.listdir(exp_dir) if "drlinda" in exp and "w53" in exp]
    rl_agents["dqn"] = [exp for exp in os.listdir(exp_dir) if "dqn" in exp and "w53" in exp]

    # for agent in rl_agents.keys():
    for agent in ["swirl"]:
        res = dict()
        for instance in rl_agents[agent]:
            res[instance] = list()

            args.rl_exp_load = f"{exp_dir}/{instance}/experiment_object.pickle"
            args.rl_model_load = f"{exp_dir}/{instance}/best_mean_reward_model.zip"
            args.rl_env_load = f"{exp_dir}/{instance}/vec_normalize.pkl"

            db_conf = None

            db_conf = configparser.ConfigParser()
            db_conf.read(args.db_conf_file)
            db_conf["postgresql"]["host"] = "10.26.42.166"
            db_conf["postgresql"]["database"] = "dsb_1gb103"
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

        args.res_save = f"/data1/wz/index/index_eab/eab_olap/bench_result/dsb/{res_id}/" \
                        f"dsb_10-1gb_template_53_multi_{res_id}_index_{agent}.json"
        if args.res_save is not None:
            if not os.path.exists(os.path.dirname(args.res_save)):
                os.makedirs(os.path.dirname(args.res_save))
            with open(args.res_save, "w") as wf:
                json.dump(res_data, wf, indent=2, cls=IndexEncoder)


def get_mcts_multi_work_res():
    parser = mcts_com.get_parser()
    args = parser.parse_args()

    gap, start, end = 20, 0, 1
    args.work_file = "/data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_work_temp_multi_w53_n100_test.json"
    # args.res_save = f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_level/" \
    #                 f"tpch_1gb_template_18_multi_work_index{start * gap}-{end * gap}.json"
    args.res_save = f"/data1/wz/index/index_eab/eab_olap/bench_result/dsb/work_level/" \
                    f"dsb_1gb_template_53_multi_work_index_mcts.json"

    # args.work_file = "/data1/wz/index/index_eab/eab_olap/bench_random/tpch/tpch_work_multi_w18_n100_test.json"
    # args.res_save = "/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_random/" \
    #                 "tpch_work_random_multi_w18_n100_test_index_mcts.json"

    # args.is_utilized = True

    args.is_trace = True

    args.process, args.overhead = True, True
    args.sel_params = "parameters"
    
    args.budget = 1000

    args.constraint = "storage"
    args.storage = 500

    # args.constraint = "number"
    args.cardinality = 5

    args.select_policy = "UCT"
    args.roll_num = 1
    args.best_policy = "BCE"

    args.db_file = "/data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_dsb_1gb.conf"
    args.schema_file = "/data1/wz/index/index_eab/eab_data/db_info_conf/schema_tpcds_1gb.json"

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
    args.work_file = "/data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_work_temp_multi_w53_n100_test.json"
    # args.res_save = f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_level/" \
    #                 f"tpch_1gb_template_18_multi_work_index{start * gap}-{end * gap}.json"
    args.res_save = f"/data1/wz/index/index_eab/eab_olap/bench_result/dsb/work_level/" \
                    f"dsb_1gb_template_53_multi_work_index_mab.json"

    # args.varying_frequencies = True
    # args.work_file = "/data1/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_freq_n100_test.json"
    # args.res_save = f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_freq/" \
    #                 f"tpch_1gb_template_18_multi_work_freq_index{start * gap}-{end * gap}.json"

    # args.work_file = "/data1/wz/index/index_eab/eab_olap/bench_random/tpch/tpch_work_multi_w18_n100_test.json"
    # args.res_save = "/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_random/" \
    #                 "tpch_work_random_multi_w18_n100_test_index_mab.json"

    args.is_utilized = True

    args.process = True
    args.sel_params = "parameters"
    args.exp_file = "/data1/wz/index/index_eab/eab_algo/mab_selection/config/exp.conf"

    args.bench = "dsb"
    args.rounds = 1000

    args.constraint = "storage"
    args.max_memory = 500

    # args.constraint = "number"
    args.max_count = 5

    args.db_file = "/data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_dsb_1gb.conf"
    args.schema_file = "/data1/wz/index/index_eab/eab_data/db_info_conf/schema_tpcds_1gb.json"

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
    res_dir = f"/data1/wz/index/index_eab/eab_olap/bench_result/dsb/{res_id}"
    for file in tqdm(os.listdir(res_dir)):
        # if "dsb_1gb_template_53_multi_work_index" not in file:
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

            if not os.path.exists(f"{res_dir}_simple"):
                os.makedirs(f"{res_dir}_simple")
            with open(f"{res_dir}_simple/{file.replace('.json', '_simple.json')}", "w") as wf:
                json.dump(res_data, wf, indent=2)
        except:
            import traceback
            traceback.print_exc()

            print(file)


if __name__ == "__main__":
    # get_heu_multi_work_res()

    exp_id = "exp_res_main"  # exp_res_main, exp_res_vol
    res_id = "work_permutation"  # work_level, work_vol, work_shift, work_permutation
    # get_swirl_multi_work_res(exp_id, res_id)

    # get_swirl_multi_work_res_data_shift(exp_id, res_id)

    # get_mcts_multi_work_res()
    get_mab_multi_work_res()

    # work_shift, work_vol
    res_id = "work_permutation"
    # remove_step_info(res_id)
