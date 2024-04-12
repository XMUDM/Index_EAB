# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: isum_main
# @Author: Wei Zhou
# @Time: 2024/1/12 19:42

import math
import numpy as np

import json
from tqdm import tqdm

import copy
import configparser

import sys
sys.path.append("/data1/wz/index")

from index_eab.eab_other.isum_model.isum_workload import Workload
from index_eab.eab_other.isum_model.isum_utils import get_tbl_col_from_schema, read_row_query, traverse_plan, get_parser
from index_eab.eab_other.isum_model.isum_model import cal_work_feature, cal_utility, cal_query_feature, \
    upd_utility, upd_query_feature, ass_weight

from index_eab.eab_utils.postgres_dbms import PostgresDatabaseConnector


def greedy_all_pairs(w):
    max_benefit = -1
    max_benefit_query = None
    for q1 in w.queries:
        benefit = q1.utility
        for q2 in w.queries:
            if q1 == q2:
                continue

            feat_array = (q1.feature, q2.feature)
            benefit += (np.sum(np.min(feat_array, axis=0)) / np.sum(np.max(feat_array, axis=0)))

        if benefit > max_benefit:
            max_benefit = benefit
            max_benefit_query = q1

    return max_benefit_query


def greedy_summary(w):
    max_benefit = -1
    max_benefit_query = None

    cal_work_feature(w)
    total_utility = np.sum([q.utility for q in w.queries])
    for q in w.queries:
        if np.sum(q.feature) == 0.:
            continue

        contribution = q.feature
        reduced_total_utility = total_utility - q.utility

        v_ = (w.feature - contribution) * (total_utility / reduced_total_utility)

        feat_array = (q.feature, v_)
        benefit = q.utility + np.sum(np.min(feat_array, axis=0)) / np.max(np.max(feat_array, axis=0))

        if benefit > max_benefit:
            max_benefit = benefit
            max_benefit_query = q

    return max_benefit_query


def run_compress(w, k, conn, col_dict):
    for q in w.queries:
        plan = conn.get_plan(q.text)
        root = plan

        seq = list()
        terminal = list()
        traverse_plan(seq, terminal, root, None)

        cal_utility(q, root, seq)
        cal_query_feature(q, col_dict)

    w_t = copy.deepcopy(w)
    w_k = list()
    while len(w_k) < k:
        # q = greedy_all_pairs(w_t)
        # [Q9, Q19, Q12, Q10, Q3, Q6, Q14, Q13]
        # [Q9, Q19, Q6, Q3, Q21, Q10, Q13, Q11]
        q = greedy_summary(w_t)
        w_k.append(q)

        upd_utility(w_t, q)
        upd_query_feature(w_t, q, col_dict)
        w_t.queries.remove(q)

    ass_weight(w, w_k)

    return w_k


def isum_main(args):
    tables, columns = get_tbl_col_from_schema(args.schema_file)

    row_sum = sum([tbl.row for tbl in tables])
    for tbl in tables:
        tbl.weight = tbl.row / row_sum

    col_dict = dict()
    for no, col in enumerate(columns):
        col_dict[f"{col.table.name}.{col.name}"] = no

    with open(args.work_load, "r") as rf:
        workload = json.load(rf)

    db_config = configparser.ConfigParser()
    db_config.read(args.db_conf_file)

    conn = PostgresDatabaseConnector(db_config, autocommit=True, host=args.host, port=args.port,
                                     db_name=args.db_name, user=args.user, password=args.password)

    work_compressed = list()
    for work in tqdm(workload):
        workload = Workload(read_row_query(work, columns))

        k = int(2 * math.sqrt(len(workload.queries)))
        w_k = run_compress(workload, k, conn, col_dict)

        w_k_pre = list()
        for query in w_k:
            w_k_pre.append([query.nr, query.text, query.weight])
        work_compressed.append(w_k_pre)

    with open(args.work_save, "w") as wf:
        json.dump(work_compressed, wf, indent=2)


def get_res(work_load, compress_load, db_conf_file, varying_frequencies=False):
    db_config = configparser.ConfigParser()
    db_config.read(db_conf_file)

    host = "59.77.5.98"
    port = 5432
    db_name = "tpch_1gb103"
    user = "wz"
    password = "ai4db2021"
    conn = PostgresDatabaseConnector(db_config, autocommit=True, host=host, port=port,
                                     db_name=db_name, user=user, password=password)

    with open(work_load, "r") as rf:
        workload = json.load(rf)

    with open(compress_load, "r") as rf:
        index_compress = json.load(rf)

    for w, ic in zip(workload, index_compress):
        no_cost, ind_cost = list(), list()
        total_no_cost, total_ind_cost = 0, 0

        for algo in ic.keys():
            indexes = ic[algo]["indexes"]
            for query in w:
                if varying_frequencies:
                    no_cost_ = conn.get_ind_cost(query[1], "") * query[-1]
                else:
                    no_cost_ = conn.get_ind_cost(query[1], "")
                total_no_cost += no_cost_
                no_cost.append(no_cost_)

                if varying_frequencies:
                    ind_cost_ = conn.get_ind_cost(query[1], indexes) * query[-1]
                else:
                    ind_cost_ = conn.get_ind_cost(query[1], indexes)
                total_ind_cost += ind_cost_
                ind_cost.append(ind_cost_)

            ic[algo]["no_cost_original"] = copy.deepcopy(no_cost)
            ic[algo]["ind_cost_original"] = copy.deepcopy(ind_cost)
            ic[algo]["total_no_cost_original"] = total_no_cost
            ic[algo]["total_ind_cost_original"] = total_ind_cost

    with open(compress_load.replace(".json", "_original.json"), "w") as wf:
        json.dump(index_compress, wf, indent=2)


def ana_res():
    res_load1 = "/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_compress/" \
                "tpch_work_temp_multi_w18_n10_eval_compressed_index_original.json"
    with open(res_load1, "r") as rf:
        res1 = json.load(rf)

    res_load2 = "/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_compress/" \
                "tpch_work_temp_multi_w18_n10_eval_compressed_weight_index_original.json"
    with open(res_load2, "r") as rf:
        res2 = json.load(rf)

    res_load3 = "/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_compress/" \
                "tpch_work_temp_multi_w18_n10_eval_index.json"
    with open(res_load3, "r") as rf:
        res3 = json.load(rf)

    # [1 - item['anytime']['total_ind_cost_original'] / item['anytime']['total_no_cost_original'] for item in res1], [1 - item['anytime']['total_ind_cost_original'] / item['anytime']['total_no_cost_original'] for item in res2], [1 - item['anytime']['total_ind_cost'] / item['anytime']['total_no_cost'] for item in res3]
    # [item['anytime']['sel_info']['time_duration'] for item in res1], [item['anytime']['sel_info']['time_duration'] for item in res2], [item['anytime']['sel_info']['time_duration'] for item in res3]
    print(1)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # args.work_load = "/data1/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_n10_eval.json"
    # args.work_save = "/data1/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_n10_eval_compressed.json"
    #
    # args.schema_file = "/data1/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json"
    # args.db_conf_file = "/data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf"

    isum_main(args)

    compress_load = "/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_compress/" \
                    "tpch_work_temp_multi_w18_n10_eval_compressed_index.json"
    compress_load = "/data1/wz/index/index_eab/eab_olap/bench_result/tpch/work_compress/" \
                    "tpch_work_temp_multi_w18_n10_eval_compressed_weight_index.json"

    # get_res(work_load, compress_load, db_conf_file)

    # ana_res()
