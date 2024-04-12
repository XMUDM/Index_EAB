# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: pre_cand_data
# @Author: Wei Zhou
# @Time: 2023/11/25 23:37

import json


def gen_cand_work():
    data_load = "/data1/wz/index/index_eab/eab_olap/bench_random/tpch/tpch_work_multi_w18_n100_test.json"
    with open(data_load, "r") as rf:
        data = json.load(rf)

    pass


if __name__ == "__main__":
    gen_cand_work()
