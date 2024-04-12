# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: job_multi_work_bench
# @Author: Wei Zhou
# @Time: 2023/8/21 17:05

import os
import json
import configparser

from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("/data/wz/index")
sys.path.append("/data/wz/index/code_utils")
sys.path.append("/data/wz/index/index_eab/eab_algo")
sys.path.append("/data/wz/index/index_eab/eab_algo/swirl_selection")
sys.path.append("/data/wz/index/index_eab/eab_algo/mcts_selection")
sys.path.append("/data/wz/index/index_eab/eab_algo/mab_selection")

from index_eab.eab_algo.heu_selection.heu_utils import selec_com
from index_eab.eab_algo.heu_selection.heu_run import get_heu_result, IndexEncoder

from index_eab.eab_algo.swirl_selection.swirl_utils import swirl_com
from index_eab.eab_algo.swirl_selection.swirl_run import get_swirl_res, pre_infer_obj

from index_eab.eab_algo.mcts_selection.mcts_run import get_mcts_res, MCTSEncoder
from mcts_selection.mcts_utils import mcts_com
from index_eab.eab_algo.mab_selection.shared import mab_com
from index_eab.eab_algo.mab_selection.mab_run import get_mab_res

heu_algos = ["extend", "db2advis", "relaxation", "anytime", "auto_admin", "drop"]
# heu_algos = ["extend", "db2advis"]

rl_agents = {"swirl": list(), "drlinda": list(), "dqn": list()}


def get_heu_multi_work_res():
    parser = selec_com.get_parser()
    args = parser.parse_args()

    args.algos = ["extend", "db2advis", "relaxation", "anytime", "auto_admin", "drop"]
    args.algos = ["extend"]

    gap, start, end = 20, 4, 5
    args.work_file = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_work_temp_multi_w33_n100_test.json"
    args.res_save = f"/data/wz/index/index_eab/eab_olap/bench_result/job/work_level/" \
                    f"job_template_33_multi_work_index_drop{start * gap}-{end * gap}.json"

    # args.res_save = f"/data/wz/index/index_eab/eab_olap/bench_result/job/work_num/" \
    #                 f"job_template_33_multi_work_num5_index{start * gap}-{end * gap}.json"

    # args.res_save = f"/data/wz/index/index_eab/eab_olap/bench_result/job/work_level/" \
    #                 f"job_template_33_multi_work_sto900_index.json"

    # args.varying_frequencies = True
    # args.work_file = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_work_temp_multi_w33_freq_n100_test.json"
    # args.res_save = f"/data/wz/index/index_eab/eab_olap/bench_result/job/work_freq/" \
    #                 f"job_template_33_multi_work_freq_index{start * gap}-{end * gap}.json"

    args.work_file = "/data/wz/index/index_eab/eab_olap/bench_random/job/job_work_multi_w33_n10_eval.json"
    args.res_save = "/data/wz/index/index_eab/eab_olap/bench_random/job/job_work_multi_w33_n10_eval_db2advis.json"

    args.work_file = "/data/wz/index/index_eab/eab_olap/bench_random/job/job_work_multi_w33_n100_test.json"
    args.res_save = "/data/wz/index/index_eab/eab_olap/bench_result/job/" \
                    "work_model/job_work_random_model_former_multi_w33_n100_test_sto500_index_db2advis.json"

    args.work_file = "/data/wz/index/index_eab/eab_olap/bench_random/job/job_work_multi_w33_n100_test.json"
    args.res_save = "/data/wz/index/index_eab/eab_olap/bench_result/job/" \
                    "work_random/job_work_random_multi_w18_n100_test_index_extend.json"

    args.process, args.overhead = True, True
    args.sel_params = "parameters"
    args.exp_conf_file = "/data/wz/index/index_eab/eab_data/heu_run_conf/{}_config_tpch.json"

    args.constraint = "storage"
    args.budget_MB = 500
    # args.budget_MB = 3000

    # args.constraint = "number"
    args.max_indexes = 5

    args.db_conf_file = "/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_job.conf"
    args.schema_file = "/data/wz/index/index_eab/eab_data/db_info_conf/schema_job.json"

    with open(args.work_file, "r") as rf:
        query_list = json.load(rf)
        # query_list = json.load(rf)[start * gap:end * gap]

    res_data = list()
    for query in tqdm(query_list):
        # workload = [info[1] for info in query]
        workload = query
        res_data.append(get_heu_result(args, args.algos, workload))

        if args.res_save is not None:
            if not os.path.exists(os.path.dirname(args.res_save)):
                os.makedirs(os.path.dirname(args.res_save))
            with open(args.res_save, "w") as wf:
                json.dump(res_data, wf, indent=2, cls=IndexEncoder)


def get_heu_multi_work_cand_res():
    parser = selec_com.get_parser()
    args = parser.parse_args()

    args.algos = ["anytime", "relaxation", "db2advis"]

    gap, start, end = 10, 0, 1
    args.work_file = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_work_temp_multi_w33_n100_test.json"
    args.res_save = f"/data/wz/index/index_eab/eab_olap/bench_result/job/work_level/" \
                    f"job_template_33_multi_work_index_drop{start * gap}-{end * gap}.json"

    for method in ["openGauss"]:  # "permutation", "dqn_rule", "openGauss"
        for utilized in [False]:  # True, False
            args.res_save = f"/data/wz/index/index_eab/eab_olap/bench_result/job/work_cand/" \
                            f"job_template_33_multi_work_cand_{method}_{str(utilized)}_index{start * gap}-{end * gap}.json"

            args.process, args.overhead = True, True
            args.sel_params = "parameters"
            args.exp_conf_file = "/data/wz/index/index_eab/eab_data/heu_run_conf/{}_config_tpch.json"

            args.constraint = "storage"
            args.budget_MB = 500

            # args.constraint = "number"
            args.max_indexes = 5

            args.db_conf_file = "/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_job.conf"
            args.schema_file = "/data/wz/index/index_eab/eab_data/db_info_conf/schema_job.json"

            with open(args.work_file, "r") as rf:
                # query_list = json.load(rf)
                query_list = json.load(rf)[start * gap:end * gap]

            res_data = list()
            for query in tqdm(query_list):
                # workload = [info[1] for info in query]
                workload = query
                res_data.append(get_heu_result(args, args.algos, workload))

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

    args.db_conf_file = "/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_job.conf"

    args.work_file = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_work_temp_multi_w33_n100_test.json"

    # args.work_file = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_work_temp_multi_w33_permutation_n100_test.json"

    # args.work_file = "/data/wz/index/index_eab/eab_olap/bench_random/job/job_work_multi_w33_n100_test.json"

    with open(args.work_file, "r") as rf:
        work_list = json.load(rf)

    # args.varying_frequencies = True

    exp_dir = f"/data/wz/index/index_eab/eab_algo/swirl_selection/{exp_id}"

    rl_agents["swirl"] = [exp for exp in os.listdir(exp_dir) if "swirl" in exp and "w33" in exp and "sql" in exp]
    rl_agents["drlinda"] = [exp for exp in os.listdir(exp_dir) if "drlinda" in exp and "w33" in exp]
    rl_agents["dqn"] = [exp for exp in os.listdir(exp_dir) if "dqn" in exp and "w33" in exp]

    # for agent in rl_agents.keys():
    for agent in ["swirl"]:
        res = dict()
        for instance in rl_agents[agent]:
            # if "openGauss" in instance:
            #     continue

            res[instance] = list()

            args.rl_exp_load = f"{exp_dir}/{instance}/experiment_object.pickle"
            args.rl_model_load = f"{exp_dir}/{instance}/best_mean_reward_model.zip"
            args.rl_env_load = f"{exp_dir}/{instance}/vec_normalize.pkl"

            # db_conf = configparser.ConfigParser()
            # db_conf.read(args.db_conf_file)

            db_conf = None

            swirl_exp, swirl_model = pre_infer_obj(args.rl_exp_load, args.rl_model_load,
                                                   args.rl_env_load, db_conf=db_conf)
            for work in tqdm(work_list):
                # workload = [info[1] for info in work]
                res[instance].append(get_swirl_res(args, work, swirl_exp, swirl_model))

        res_data = list()
        for i in range(len(res[list(res.keys())[0]])):
            # for i in range(len(res[instance])):
            r = dict()
            for instance in rl_agents[agent]:
                # if "openGauss" in instance:
                #     continue
                r[instance] = res[instance][i]
            res_data.append(r)

        args.res_save = f"/data/wz/index/index_eab/eab_olap/bench_result/job/{res_id}/" \
                        f"job_template_33_multi_{res_id}_index_sql_{agent}.json"
        if args.res_save is not None:
            if not os.path.exists(os.path.dirname(args.res_save)):
                os.makedirs(os.path.dirname(args.res_save))
            with open(args.res_save, "w") as wf:
                json.dump(res_data, wf, indent=2, cls=IndexEncoder)


def get_mcts_multi_work_res():
    parser = mcts_com.get_parser()
    args = parser.parse_args()

    gap, start, end = 20, 0, 1
    args.work_file = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_work_temp_multi_w33_n100_test.json"
    args.res_save = f"/data/wz/index/index_eab/eab_olap/bench_result/job/work_level/" \
                    f"job_template_33_multi_work_index_mcts.json"

    # args.work_file = "/data/wz/index/index_eab/eab_olap/bench_random/tpch/tpch_work_multi_w18_n100_test.json"
    # args.res_save = "/data/wz/index/index_eab/eab_olap/bench_result/tpch/work_random/" \
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

    args.db_file = "/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_job.conf"
    args.schema_file = "/data/wz/index/index_eab/eab_data/db_info_conf/schema_job.json"

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
    args.work_file = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_work_temp_multi_w33_n100_test.json"
    args.res_save = f"/data/wz/index/index_eab/eab_olap/bench_result/job/work_level/" \
                    f"job_template_33_multi_work_index_mab.json"

    # args.varying_frequencies = True
    # args.work_file = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_freq_n100_test.json"
    # args.res_save = f"/data/wz/index/index_eab/eab_olap/bench_result/tpch/work_freq/" \
    #                 f"tpch_1gb_template_18_multi_work_freq_index{start * gap}-{end * gap}.json"

    # args.work_file = "/data/wz/index/index_eab/eab_olap/bench_random/tpch/tpch_work_multi_w18_n100_test.json"
    # args.res_save = "/data/wz/index/index_eab/eab_olap/bench_result/tpch/work_random/" \
    #                 "tpch_work_random_multi_w18_n100_test_index_mab.json"

    args.is_utilized = True

    args.process = True
    args.sel_params = "parameters"
    args.exp_file = "/data/wz/index/index_eab/eab_algo/mab_selection/config/exp.conf"

    args.bench = "job"
    args.rounds = 1000

    args.constraint = "storage"
    args.max_memory = 500

    # args.constraint = "number"
    args.max_count = 5

    args.db_file = "/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_job.conf"
    args.schema_file = "/data/wz/index/index_eab/eab_data/db_info_conf/schema_job.json"

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
    res_dir = f"/data/wz/index/index_eab/eab_olap/bench_result/job/{res_id}"
    for file in tqdm(os.listdir(res_dir)):
        if "job_template_33" not in file:
            continue

        if os.path.exists(f"{res_dir}_simple/{file.replace('.json', '_simple.json')}"):
            continue

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


if __name__ == "__main__":
    # get_heu_multi_work_res()
    # get_heu_multi_work_cand_res()

    # exp_res_main, exp_res_num, exp_res_sto, exp_res_freq, exp_res_cand
    # exp_res_oracle, exp_res_cand, exp_res_state, exp_res_action
    # exp_res_random
    exp_id = "exp_res_state"

    # work_level, work_num, work_sto, work_freq, work_cand
    # work_oracle, work_cand, work_state, work_action, work_permutation
    # work_random
    res_id = "work_state"
    # get_swirl_multi_work_res(exp_id, res_id)

    # get_mcts_multi_work_res()
    get_mab_multi_work_res()

    # work_level, work_num, work_freq, work_permutation
    # work_random, work_state
    res_id = "work_state"
    # remove_step_info(res_id)
