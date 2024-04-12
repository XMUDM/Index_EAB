# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: pre_bench_data
# @Author: Wei Zhou
# @Time: 2023/7/7 12:03

import os
import re
import json
import configparser

import random
import numpy as np

import sqlparse
import mo_sql_parsing as mosqlparse

from heu_utils.workload import Workload
from heu_utils.selec_com import read_row_query_new, get_columns_from_schema
from heu_utils.candidate_generation import candidates_per_query, syntactically_relevant_indexes

from swirl_selection.swirl_utils.swirl_com import create_column_permutation_indexes, get_prom_index_candidates_original

excluded_qno = {"tpch": [15, 20, 17, 18],
                "tpch_skew": [15, 20, 17, 18],
                "tpcds": [2, 29, 36, 56, 87, 89, 95,
                          3, 34, 55, 73,
                          21, 25, 16,
                          6, 39],
                "dsb": [],
                "job": []}


def pre_temp():
    # tpcds template no.77
    temp_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_1gb_template_99_multi_query.json"
    with open(temp_load, "r") as rf:
        query_group = json.load(rf)

    temp_list = query_group["77"]

    pre_list = list()
    for temp in temp_list:
        pre_list.append(temp.replace("coalesce(returns, 0) returns", "coalesce(returns, 0) as returns"))

    query_group["77"] = pre_list

    temp_save = "/data/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_1gb_template_99_multi_query_pre.json"
    with open(temp_save, "w") as wf:
        json.dump(query_group, wf, indent=2)


def get_temp_list(bench):
    if bench == "tpch" or bench == "tpch_skew":
        temp_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_1gb_template_22_multi500_query.json"
        with open(temp_load, "r") as rf:
            query_group = json.load(rf)

    elif bench == "tpcds":
        temp_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_1gb_template_99_multi_query_pre.json"
        with open(temp_load, "r") as rf:
            query_group = json.load(rf)

    elif bench == "dsb":
        temp_load = "/data/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_template_multi_query_flat.json"
        with open(temp_load, "r") as rf:
            query_group = json.load(rf)

    elif bench == "job":
        data_dir = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_original"

        temp_group = dict()
        for file in os.listdir(data_dir):
            group = re.findall(r"([0-9]+)[a-z]", file)
            if len(group) == 0:
                continue

            group = group[0]
            if group not in temp_group.keys():
                temp_group[group] = list()
            temp_group[group].append(file)

        query_group = dict()
        for temp in temp_group.keys():
            query_group[temp] = list()
            for ins in temp_group[temp]:
                data_load = f"{data_dir}/{ins}"
                with open(data_load, "r") as rf:
                    sql = "".join(rf.readlines()).replace("\n", " ")
                    sql = mosqlparse.format(mosqlparse.parse(sql))
                query_group[temp].append(sql)

    return query_group


def pre_query_temp(bench, work_num, seed=666):
    np.random.seed(seed)

    query_group = get_temp_list(bench)

    temp_num = [len(query_group[group]) for group in query_group.keys()]
    assert work_num <= np.sum(temp_num), f"{work_num} vs. {np.sum(temp_num)}, can't generate the queries!"

    weights = np.array(temp_num) / np.sum(temp_num)

    work_list = list()
    work_temp_list = list()
    while len(work_list) < work_num:
        # 1. determine the class
        cls = np.random.choice(list(query_group.keys()), p=weights)

        if bench == "tpch" and cls == "15":
            continue

        # 2. determine the instance
        query = random.choice(query_group[cls])

        # 3. determine the frequency
        freq = random.randint(1, 1000)

        tup = [int(cls), query, freq]
        if [int(cls), query] in work_temp_list:
            continue

        work_list.append([tup])
        work_temp_list.append([int(cls), query])

    assert len(set([str(work) for work in work_temp_list])) == work_num, "Duplicate workload exists!"

    data_save = f"/data/wz/index/index_eab/eab_olap/bench_temp/{bench}/{bench}_query_temp_multi_n{work_num}.json"
    if not os.path.exists(os.path.dirname(data_save)):
        os.makedirs(os.path.dirname(data_save))
    with open(data_save, "w") as wf:
        json.dump(work_list, wf, indent=2)


def pre_query_temp_eval(bench, temp_num, seed=666):
    random.seed(seed)

    query_group = get_temp_list(bench)

    work_list = list()
    for qid in range(temp_num):
        if str(qid + 1) not in query_group.keys():
            continue
        # if str(qid + 1) not in query_group.keys() \
        #         or qid + 1 in excluded_qno_query[bench]:
        #     continue
        if bench in ["tpch", "tpch_skew"] and qid + 1 == 15:
            continue

        # 1. determine the instance
        query = random.choice(query_group[str(qid + 1)])
        # 2. determine the frequency
        freq = random.randint(1, 1000)

        tup = [qid + 1, query, freq]
        work_list.append([tup])

    data_save = f"/data/wz/index/index_eab/eab_olap/bench_temp/{bench}/{bench}_query_temp_multi_n{len(work_list)}_eval.json"
    if not os.path.exists(os.path.dirname(data_save)):
        os.makedirs(os.path.dirname(data_save))
    with open(data_save, "w") as wf:
        json.dump(work_list, wf, indent=2)


def pre_query_temp_test(bench, temp_num, instance_num, train_load, seed=666):
    random.seed(seed)

    query_group = get_temp_list(bench)
    with open(train_load, "r") as rf:
        train_data = json.load(rf)

    train_instances = list(set([item[0][1] for item in train_data]))

    work_list = list()
    for qid in range(temp_num):
        if str(qid + 1) not in query_group.keys():
            continue
        if bench in ["tpch", "tpch_skew"] and qid + 1 == 15:
            continue

        # 1. determine the instance
        if bench == "job":
            queries = query_group[str(qid + 1)]
        else:
            distinct = list(set(query_group[str(qid + 1)]).difference(set(train_instances)))
            if len(distinct) < instance_num:
                queries = random.sample(distinct, len(distinct))
            else:
                queries = random.sample(distinct, instance_num)

        for query in queries:
            # 2. determine the frequency
            freq = random.randint(1, 1000)

            tup = [qid + 1, query, freq]
            work_list.append([tup])

    data_save = f"/data/wz/index/index_eab/eab_olap/bench_temp/{bench}/{bench}_query_temp_multi_c{instance_num}_n{len(work_list)}_test.json"
    if not os.path.exists(os.path.dirname(data_save)):
        os.makedirs(os.path.dirname(data_save))
    with open(data_save, "w") as wf:
        json.dump(work_list, wf, indent=2)


def pre_work_temp(bench, work_num, work_size, temp_typ, seed=666):
    random.seed(seed)

    query_group = get_temp_list(bench)

    if bench == "tpch":
        query_group.pop("15")

    work_list = list()
    work_temp_list = list()
    while len(work_list) < work_num:
        sql_list = list()
        sql_temp_list = list()
        if temp_typ == "temp_unique":
            # 1. determine the class
            clses = random.sample(query_group.keys(), work_size)
            for cls in sorted(clses, key=lambda x: int(x)):
                # 2. determine the instance
                sql = random.choice(query_group[cls])

                # 3. determine the frequency
                freq = random.randint(1, 1000)

                tup = [int(cls), sql, freq]
                sql_list.append(tup)
                sql_temp_list.append([int(cls), sql])

        else:
            while len(sql_list) < work_size:
                # 1. determine the class
                cls = random.sample(query_group.keys(), 1)[0]

                # 2. determine the instance
                sql = random.choice(query_group[cls])

                # 3. determine the frequency
                freq = random.randint(1, 1000)

                tup = [int(cls), sql, freq]
                if tup in sql_temp_list:
                    continue

                sql_list.append(tup)
                sql_temp_list.append([int(cls), sql])

            sql_list = sorted(sql_list, key=lambda x: int(x[0]))
            sql_temp_list = sorted(sql_temp_list, key=lambda x: int(x[0]))

        work_list.append(sql_list)
        work_temp_list.append(sql_temp_list)

    assert len(set([str(work) for work in work_temp_list])) == work_num, "Duplicate workload exists!"

    data_save = f"/data/wz/index/index_eab/eab_olap/bench_temp/{bench}/{bench}_work_{temp_typ}_multi_w{work_size}_n{work_num}.json"
    if not os.path.exists(os.path.dirname(data_save)):
        os.makedirs(os.path.dirname(data_save))
    with open(data_save, "w") as wf:
        json.dump(work_list, wf, indent=2)


def pre_work_temp_eval(bench, temp_num, seed=666):
    random.seed(seed)

    query_group = get_temp_list(bench)

    work_list = list()
    for qid in range(temp_num):
        if qid + 1 in excluded_qno[bench]:
            continue

        # 1. determine the class
        cls = str(qid + 1)

        # 2. determine the instance
        sql = random.choice(query_group[cls])

        # 3. determine the frequency
        freq = random.randint(1, 1000)

        work_list.append([qid + 1, sql, freq])

    data_save = f"/data/wz/index/index_eab/eab_olap/bench_temp/{bench}/{bench}_work_temp_multi_w{len(work_list)}_n1_eval.json"
    with open(data_save, "w") as wf:
        json.dump(work_list, wf, indent=2)


def pre_work_temp_test(bench, temp_num, instance_num, train_load, seed=666):
    random.seed(seed)

    query_group = get_temp_list(bench)
    with open(train_load, "r") as rf:
        train_data = json.load(rf)

    train_instances = list()
    for item in train_data:
        train_instances.append("".join([i[1] for i in item]))

    work_list = list()
    while len(work_list) < instance_num:
        sql_list = list()
        for qid in range(temp_num):
            if qid + 1 in excluded_qno[bench]:
                continue

            # 1. determine the class
            cls = str(qid + 1)

            # 2. determine the instance
            sql = random.choice(query_group[cls])

            # 3. determine the frequency
            freq = random.randint(1, 1000)

            sql_list.append([qid + 1, sql, freq])

        if "".join([sql[1] for sql in sql_list]) not in train_instances:
            work_list.append(sql_list)

    data_save = f"/data/wz/index/index_eab/eab_olap/bench_temp/{bench}/{bench}_work_temp_multi_w{len(sql_list)}_n{instance_num}_test.json"
    with open(data_save, "w") as wf:
        json.dump(work_list, wf, indent=2)


def pre_work_temp_examp(bench, seed=666):
    random.seed(seed)

    query_group = get_temp_list(bench)

    if bench == "tpch":
        examp_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_template_21.sql"

    elif bench == "job":
        examp_load = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_template_33.sql"

    with open(examp_load, "r") as rf:
        examp_work = rf.readlines()

    data = list()
    for sql in examp_work:
        no = -1
        for qno in query_group.keys():
            if sql.strip("\n") in query_group[qno]:
                no = int(qno)
                break
        try:
            assert no != -1, "The query class is not correct!"
        except:
            print(1)

        freq = random.randint(1, 1000)
        data.append([no, sql, freq])

    data = sorted(data, key=lambda x: int(x[0]))

    examp_save = f"/data/wz/index/index_eab/eab_olap/bench_temp/{bench}/{bench}_temp_example_w{len(data)}_n1.json"
    with open(examp_save, "w") as wf:
        json.dump(data, wf, indent=2)


def ana_cand_num():
    schema_file = "/data/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json"
    _, columns = get_columns_from_schema(schema_file)

    work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_template_21.sql"
    with open(work_load, "r") as rf:
        query_texts = rf.readlines()
    workload = Workload(read_row_query_new(query_texts, columns))
    indexable_columns = workload.indexable_columns()

    max_index_width = 2
    candidates_v1 = candidates_per_query(workload, max_index_width,
                                         candidate_generator=syntactically_relevant_indexes)
    candidates_v1_pre = list()
    for cand in candidates_v1:
        candidates_v1_pre.extend(cand)
    candidates_v1_pre = list(set(candidates_v1_pre))

    candidates_v2 = create_column_permutation_indexes(indexable_columns, max_index_width)

    db_conf_file = "/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf"
    db_conf = configparser.ConfigParser()
    db_conf.read(db_conf_file)
    candidates_v3 = get_prom_index_candidates_original(db_conf, [query_texts],
                                                       None, indexable_columns)
    pass


def ana_multi_cost():
    pass


def ana_work_temp():
    work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_temp_unique_multi_w33_n1000.json"
    work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_temp_example_w33_n1.json"

    work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_multi_query_n1000.json"
    work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_multi_query_n1000_eval.json"

    work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch_skew/tpch_skew_multi_query_n17_eval.json"

    work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_temp_multi_query_n113.json"
    work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_temp_multi_query_n33_eval.json"

    work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_temp_multi_query_n3000.json"
    work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_temp_multi_query_n99_eval.json"

    work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_query_temp_multi_n5000.json"
    work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_query_temp_multi_n99_eval.json"

    # work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_temp_multi_query_n3000.json"
    # work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_temp_multi_query_n53_eval.json"

    with open(work_load, "r") as rf:
        data = json.load(rf)
    pass


if __name__ == "__main__":
    bench = "job"  # "tpch", "tpch_skew", "tpcds", "dsb", "job"

    work_num = {"tpch": 1000, "tpch_skew": 1000,
                "tpcds": 5000, "dsb": 3000,
                "job": 113}

    work_size = {"tpch": 18, "tpch_skew": 18,
                 "tpcds": 83, "dsb": 53,
                 "job": 33}

    temp_num = {"tpch": 22, "tpch_skew": 22,
                "tpcds": 99, "dsb": 53,
                "job": 33}

    seed = 666

    # todo: 0. preprocess the template
    # pre_temp()

    # todo: 1. query-level template data
    # pre_query_temp(bench, work_num[bench], seed)
    # pre_query_temp_eval(bench, temp_num[bench], seed)

    instance_num = 100
    train_load = f"/data/wz/index/index_eab/eab_olap/bench_temp/{bench}/{bench}_query_temp_multi_n{work_num[bench]}.json"
    pre_query_temp_test(bench, temp_num[bench], instance_num, train_load, seed)

    # todo: 2. workload-level template data
    # for temp_typ in ["temp_unique", "temp_duplicate"]:
    #     pre_work_temp(bench, work_num[bench], work_size[bench], temp_typ, seed)
    # pre_work_temp_eval(bench, temp_num[bench], seed)

    instance_num = 100
    temp_typ = "temp_duplicate"
    train_load = f"/data/wz/index/index_eab/eab_olap/bench_temp/{bench}/{bench}_work_{temp_typ}_multi_w{work_size[bench]}_n{work_num[bench]}.json"
    # pre_work_temp_test(bench, temp_num[bench], instance_num, train_load, seed=666)

    # todo: 3. workload statistic
    ana_cand_num()
    # ana_work_temp()
