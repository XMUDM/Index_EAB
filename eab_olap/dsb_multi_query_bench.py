# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: dsb_multi_query_bench
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

from mcts_selection.mcts_utils import mcts_com
from mcts_selection.mcts_run import get_mcts_res, MCTSEncoder

from shared import mab_com
from mab_run import get_mab_res

heu_algos = ["extend", "db2advis", "relaxation", "anytime", "auto_admin", "drop"]
rl_agents = {"swirl": list(), "drlinda": list(), "dqn": list()}


def flat_multi_query():
    work_load = "/data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_template_multi_query.json"
    with open(work_load, "r") as rf:
        query_list = json.load(rf)

    new_query_list = dict()
    for typ in query_list.keys():
        if typ == "agg_queries":
            for temp in sorted(query_list[typ].keys(), key=lambda x: int(x)):
                new_query_list[temp] = query_list[typ][temp]
        elif typ == "multi_block_queries":
            for temp in sorted(query_list[typ].keys(), key=lambda x: int(x)):
                new_query_list[str(int(temp) + 15)] = query_list[typ][temp]
        elif typ == "spj_queries":
            for temp in sorted(query_list[typ].keys(), key=lambda x: int(x)):
                new_query_list[str(int(temp) + 15 + 23)] = query_list[typ][temp]

    work_save = "/data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_template_multi_query_flat.json"
    with open(work_save, "w") as wf:
        json.dump(new_query_list, wf, indent=2)


def get_heu_multi_query_res():
    parser = selec_com.get_parser()
    args = parser.parse_args()

    # gap, start, end = 9, 0, 1
    # qnos = list(range(1, 53 + 1))[start * gap:end * gap]
    #
    # args.work_file = "/data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_template_multi_query_flat.json"
    # args.res_save = f"/data1/wz/index/index_eab/eab_olap/bench_result/tpcds/query_level/" \
    #                 f"dsb_1gb_template_53_multi500_query_index{start * gap}-{end * gap}.json"

    args.work_file = "/data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_query_temp_multi_c100_n5174_test.json"
    args.res_save = f"/data1/wz/index/index_eab/eab_olap/bench_result/dsb/query_level/" \
                    f"dsb_1gb_template_53_multi_query_index.json"

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
        query_list = json.load(rf)

    # res_data = dict()
    # for qno in qnos:
    #     res_data[str(qno)] = list()
    #     for query in query_list[str(qno)]:
    #         res_data[str(qno)].append(get_heu_result(args, [query]))

    res_data = dict()
    for query in tqdm(query_list):
        if query[0][0] not in res_data.keys():
            res_data[query[0][0]] = list()
        # res_data[query[0][0]].append(get_heu_result(args, heu_algos, [query[0][1]]))
        res_data[query[0][0]].append(get_heu_result(args, heu_algos, query))

    if args.res_save is not None:
        if not os.path.exists(os.path.dirname(args.res_save)):
            os.makedirs(os.path.dirname(args.res_save))
        with open(args.res_save, "w") as wf:
            json.dump(res_data, wf, indent=2, cls=IndexEncoder)


def get_swirl_multi_query_res():
    parser = swirl_com.get_parser()
    args = parser.parse_args()

    args.seed = 666
    args.max_budgets = 500

    args.db_conf_file = "/data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf"

    args.work_file = "/data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_query_temp_multi_c100_n5174_test.json"
    if args.work_file.endswith(".sql"):
        with open(args.work_file, "r") as rf:
            query_list = rf.readlines()
    elif args.work_file.endswith(".json"):
        with open(args.work_file, "r") as rf:
            query_list = json.load(rf)

    exp_dir = "/data1/wz/index/index_eab/eab_algo/swirl_selection/exp_res"
    rl_agents["swirl"] = [exp for exp in os.listdir(exp_dir) if "swirl" in exp and "dsb_query" in exp]
    rl_agents["drlinda"] = [exp for exp in os.listdir(exp_dir) if "drlinda" in exp and "dsb_query" in exp]
    rl_agents["dqn"] = [exp for exp in os.listdir(exp_dir) if "dqn" in exp and "dsb_query" in exp]

    # for agent in rl_agents.keys():
    for agent in ["swirl"]:
        res = dict()
        for instance in rl_agents[agent]:
            res[instance] = dict()

            args.rl_exp_load = f"{exp_dir}/{instance}/experiment_object.pickle"
            args.rl_model_load = f"{exp_dir}/{instance}/best_mean_reward_model.zip"
            args.rl_env_load = f"{exp_dir}/{instance}/vec_normalize.pkl"

            db_conf = None

            db_conf = configparser.ConfigParser()
            db_conf.read(args.db_conf_file)
            db_conf["postgresql"]["host"] = "10.26.42.166"
            db_conf["postgresql"]["db_name"] = "dsb_1gb103"
            db_conf["postgresql"]["port"] = "5432"
            db_conf["postgresql"]["user"] = "wz"
            db_conf["postgresql"]["password"] = "ai4db2021"

            swirl_exp, swirl_model = pre_infer_obj(args.rl_exp_load, args.rl_model_load,
                                                   args.rl_env_load, db_conf=db_conf)

            for query in tqdm(query_list):
                if query[0][0] not in res[instance].keys():
                    res[instance][query[0][0]] = list()
                res[instance][query[0][0]].append(
                    get_swirl_res(args, query, swirl_exp, swirl_model))

        res_data = dict()
        for qno in res[instance].keys():
            res_data[qno] = list()
            for i in range(len(res[instance][qno])):
                r = dict()
                for ins in res.keys():
                    r[ins] = res[ins][qno][i]
                res_data[qno].append(r)

        args.res_save = f"/data1/wz/index/index_eab/eab_olap/bench_result/dsb/query_level/" \
                        f"dsb_1gb_template_53_multi_query_index_{agent}.json"
        if args.res_save is not None:
            if not os.path.exists(os.path.dirname(args.res_save)):
                os.makedirs(os.path.dirname(args.res_save))
            with open(args.res_save, "w") as wf:
                json.dump(res_data, wf, indent=2, cls=IndexEncoder)


def get_mcts_multi_query_res():
    parser = mcts_com.get_parser()
    args = parser.parse_args()

    gap, start, end = 3000, 1, 2
    args.work_file = "/data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_query_temp_multi_c100_n5174_test.json"
    args.res_save = f"/data1/wz/index/index_eab/eab_olap/bench_result/dsb/work_level/" \
                    f"dsb_1gb_template_53_multi_work_index_mcts{start * gap}-{end * gap}.json"
    # args.res_save = f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch/query_level/" \
    #                 f"dsb_1gb_template_53_multi_work_index_mcts.json"

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
        # query_list = json.load(rf)
        query_list = json.load(rf)[start * gap:end * gap]

    res_data = dict()
    for query in tqdm(query_list):
        if query[0][0] not in res_data.keys():
            res_data[query[0][0]] = list()
        res_data[query[0][0]].append(get_mcts_res(args, query))

        if args.res_save is not None:
            if not os.path.exists(os.path.dirname(args.res_save)):
                os.makedirs(os.path.dirname(args.res_save))
            with open(args.res_save, "w") as wf:
                json.dump(res_data, wf, indent=2, cls=MCTSEncoder)


def get_mab_multi_query_res():
    parser = mab_com.get_parser()
    args = parser.parse_args()

    gap, start, end = 3000, 1, 2
    args.work_file = "/data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_query_temp_multi_c100_n5174_test.json"
    args.res_save = f"/data1/wz/index/index_eab/eab_olap/bench_result/dsb/work_level/" \
                    f"dsb_1gb_template_53_multi_work_index_mab{start * gap}-{end * gap}.json"
    # args.res_save = f"/data1/wz/index/index_eab/eab_olap/bench_result/tpch/query_level/" \
    #                 f"dsb_1gb_template_53_multi_work_index_mab.json"

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
        # query_list = json.load(rf)
        query_list = json.load(rf)[start * gap:end * gap]

    res_data = dict()
    for query in tqdm(query_list):
        if query[0][0] not in res_data.keys():
            res_data[query[0][0]] = list()
        res_data[query[0][0]].append(get_mab_res(args, query))

        if args.res_save is not None:
            if not os.path.exists(os.path.dirname(args.res_save)):
                os.makedirs(os.path.dirname(args.res_save))
            with open(args.res_save, "w") as wf:
                json.dump(res_data, wf, indent=2, cls=IndexEncoder)


def remove_step_info():
    res_dir = "/data1/wz/index/index_eab/eab_olap/bench_result/dsb/query_level"
    for file in tqdm(os.listdir(res_dir)):
        if "dsb_1gb_template_53" not in file:
            continue

        try:
            with open(f"{res_dir}/{file}", "r") as rf:
                res_data = json.load(rf)
        except:
            print(file)

        for qno in res_data.keys():
            for item in res_data[qno]:
                for algo in item.keys():
                    if "step" in item[algo]["sel_info"].keys():
                        item[algo]["sel_info"].pop("step")

        with open(f"{res_dir}/{file.replace('.json', '_simple.json')}", "w") as wf:
            json.dump(res_data, wf, indent=2)


if __name__ == "__main__":
    # flat_multi_query()
    # get_heu_multi_query_res()
    # get_swirl_multi_query_res()

    # get_mcts_multi_query_res()
    get_mab_multi_query_res()

    # remove_step_info()
