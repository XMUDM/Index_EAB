# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: isum_utils
# @Author: Wei Zhou
# @Time: 2024/1/12 11:58

import json
import argparse
import configparser

from index_eab.eab_utils.postgres_dbms import PostgresDatabaseConnector
from index_eab.eab_other.isum_model.isum_workload import Query, Table, Column, Workload
from index_eab.eab_other.isum_model.isum_model import cal_utility, cal_work_feature, cal_influence, cal_query_feature


def get_parser():
    parser = argparse.ArgumentParser(
        description="A MODEL for workload compression.")

    parser.add_argument("--exp_id", type=str, default="new_exp_opt")

    # 1. tpch
    parser.add_argument("--work_load", type=str,
                        default="/data1/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_n10_eval.json")
    parser.add_argument("--work_save", type=str,
                        default="/data1/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_n10_eval_compressed.json")

    parser.add_argument("--schema_file", type=str,
                        default="/data1/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json")
    parser.add_argument("--db_conf_file", type=str,
                        default="/data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf")

    # 2. tpcds

    # 3. job

    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--db_name", type=str, default=None)

    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)

    return parser


def get_tbl_col_from_schema(schema_file):
    tables, columns = list(), list()
    with open(schema_file, "r") as rf:
        db_schema = json.load(rf)

    for item in db_schema:
        table_object = Table(item["table"], row=item["rows"])
        tables.append(table_object)
        for col_info in item["columns"]:
            column_object = Column(col_info["name"])
            table_object.add_column(column_object)
            columns.append(column_object)

    return tables, columns


def read_row_query(sql_list, columns):
    workload = list()

    if isinstance(sql_list[0], list):
        query_ids = [item[0] for item in sql_list]
        query_texts = [item[1] for item in sql_list]
        query_freqs = [item[2] for item in sql_list]
    else:
        query_ids = [i for i in range(len(sql_list))]
        query_texts = sql_list
        query_freqs = [1 for _ in range(len(sql_list))]

    for query_id, query_text, query_freq in zip(query_ids, query_texts, query_freqs):
        qcols, qname = list(), dict()
        for column in columns:
            if column.name in query_text.lower() and \
                    f"{column.table.name}" in query_text.lower():
                qcols.append(column)

                # if column.name not in qname.keys():
                #     qname[column.name] = 0
                # qname[column.name] += 1

        # todo(0113): alias
        # qcols_pre = list()
        # for column in qcols:
        #     if qname[column.name] == 1:
        #         qcols_pre.append(column)
        #     else:
        #         if f"{column.name}.{column.table.name}" in query_text.lower():
        #             qcols_pre.append(column)

        query = Query(query_id, query_text, columns=qcols, frequency=query_freq)
        workload.append(query)

    return workload


def traverse_plan(seq, terminal, node, parent):
    node["parent"] = parent
    node["ID"] = len(seq)

    seq.append(node)

    if "Plans" in node.keys():
        for i, n in enumerate(node["Plans"]):
            traverse_plan(seq, terminal, n, node["ID"])
            if i == 0:
                node["left"] = n["ID"]
            elif i == 1:
                node["right"] = n["ID"]
    else:
        terminal.append(node["ID"])


if __name__ == "__main__":
    schema_file = "/data1/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json"
    tables, columns = get_tbl_col_from_schema(schema_file)

    row_sum = sum([tbl.row for tbl in tables])
    for tbl in tables:
        tbl.weight = tbl.row / row_sum

    col_dict = dict()
    for no, col in enumerate(columns):
        col_dict[f"{col.table.name}.{col.name}"] = no

    work_load = "/data1/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_n10_eval.json"
    with open(work_load, "r") as rf:
        workload = json.load(rf)

    workload = Workload(read_row_query([workload[0][14]], columns))

    db_conf_file = "/data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf"
    db_config = configparser.ConfigParser()
    db_config.read(db_conf_file)

    host = "59.77.5.98"
    port = 5432
    db_name = "tpch_1gb103"
    user = "wz"
    password = "ai4db2021"
    conn = PostgresDatabaseConnector(db_config, autocommit=True, host=host, port=port,
                                     db_name=db_name, user=user, password=password)

    for query in workload.queries:
        plan = conn.get_plan(query.text)
        root = plan

        # 1. tag the node in the plan
        seq = list()
        terminal = list()
        traverse_plan(seq, terminal, root, None)

        cal_utility(query, root, seq)
        cal_query_feature(query, col_dict)

    cal_work_feature(workload, col_dict)

    for query in workload.queries:
        cal_influence(workload, query)
