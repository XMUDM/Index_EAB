# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: pre_cost_data
# @Author: Wei Zhou
# @Time: 2023/10/2 16:45

import os
import json

import random
import numpy as np

import shutil
from tqdm import tqdm

timeout = 180000
bench = ["tpch", "tpcds", "job"]


def pre_cost_file():
    file_load = "/data/wz/index/index_eab/eab_data/act_info_data.txt"
    with open(file_load, "r") as rf:
        data = rf.readlines()

    for ben in bench:
        file_save = f"/data/wz/index/index_eab/eab_data/act_info_{ben}_data.txt"

        with open(file_save, "w") as wf:
            for d in [dat for dat in data if "_random_" in dat and ben in dat]:
                wf.writelines(d)


def organize_cost_file():
    for ben in bench:
        file_load = f"/data/wz/index/index_eab/eab_data/act_info_{ben}_data.txt"
        with open(file_load, "r") as rf:
            data = rf.readlines()

        save_dir = "/data/wz/index/index_eab/eab_other/cost_data"
        for from_path in tqdm(data):
            from_path = from_path.strip("\n")
            dest_path = f"{save_dir}/{ben}/{from_path.split('/')[-3]}_{from_path.split('/')[-1]}"
            if not os.path.exists(os.path.dirname(dest_path)):
                os.makedirs(os.path.dirname(dest_path))

            shutil.copy(from_path, dest_path)


def pre_corr_data():
    # tpch: 3200, tpcds: 2853, job: 2000
    # src_est_hypo_cost, src_act_not_hypo_cost
    for ben in bench:
        load_dir = f"/data/wz/index/index_eab/eab_other/cost_data/{ben}"

        data_sub = list()
        est_data, act_data = list(), list()
        for file in os.listdir(load_dir):
            with open(f"{load_dir}/{file}", "r") as rf:
                data = json.load(rf)

            # if "src_act_not_hypo_cost" in data[0].keys():
            #     est_data.extend([dat["src_est_hypo_cost"] for dat in data])
            #     act_data.extend([dat["src_act_not_hypo_cost"] for dat in data])
            #     data_sub.extend(data)

            est_data.extend([dat["tgt_est_hypo_cost"] for dat in data if dat["tgt_act_not_hypo_cost"] != timeout])
            act_data.extend([dat["tgt_act_not_hypo_cost"] for dat in data if dat["tgt_act_not_hypo_cost"] != timeout])
            data_sub.extend([dat for dat in data if dat["tgt_act_not_hypo_cost"] != timeout])

        data_save = f"/data/wz/index/index_eab/eab_other/cost_data/{ben}_corr_cost_data_tgt.json"
        if not os.path.exists(os.path.dirname(data_save)):
            os.makedirs(os.path.dirname(data_save))
        with open(data_save, "w") as wf:
            json.dump(data_sub, wf, indent=2)


def ana_corr_data():
    for ben in ["job"]:  # bench
        data_load = f"/data/wz/index/index_eab/eab_other/cost_data/{ben}_corr_cost_data_tgt.json"
        with open(data_load, "r") as rf:
            data_sub = json.load(rf)

        est_data = [dat["tgt_est_hypo_cost"] for dat in data_sub]
        # act_data = [dat["tgt_est_not_hypo_cost"] for dat in data_sub]
        act_data = [dat["tgt_act_not_hypo_cost"] for dat in data_sub]

        # correlation_coefficient = np.corrcoef(np.log(est_data), np.log(act_data))[0, 1]

        est_data = np.log(est_data)
        act_data = np.log(act_data)

        est_data_pre = [e for e, a in zip(est_data, act_data) if e != -np.inf and a != -np.inf]
        act_data_pre = [a for e, a in zip(est_data, act_data) if e != -np.inf and a != -np.inf]
        correlation_coefficient = np.corrcoef(est_data_pre, act_data_pre)[0, 1]

        print("Pearson correlation coefficient:", correlation_coefficient)


def pre_cost_data():
    # src_sql, src_inds
    # src_est_wo_cost, src_est_wo_plan, src_act_wo_cost
    # src_est_hypo_cost, src_est_hypo_plan, src_act_not_hypo_cost

    for typ in ["src"]:  # ["src", "tgt"]
        for ben in ["tpch"]:  # tqdm(bench)
            data_load = f"/data/wz/index/index_eab/eab_other/cost_data/{ben}_corr_cost_data_{typ}.json"
            with open(data_load, "r") as rf:
                data = json.load(rf)

            data_pre = list()
            for item in data:
                if item[f"{typ}_act_wo_cost"] == timeout or \
                        item[f"{typ}_act_not_hypo_cost"] == timeout:
                    continue

                if np.log(item[f"{typ}_act_wo_cost"]) == -np.inf or \
                        np.log(item[f"{typ}_act_not_hypo_cost"]) == -np.inf:
                    continue

                data_pre.append({"sql": item[f"{typ}_sql"], "indexes": item[f"{typ}_inds"],
                                 "w/o estimated cost": item[f"{typ}_est_wo_cost"],
                                 "w/ estimated cost": item[f"{typ}_est_hypo_cost"],
                                 "w/o actual cost": item[f"{typ}_act_wo_cost"],
                                 "w/ actual cost": item[f"{typ}_act_not_hypo_cost"],
                                 "w/o plan": item[f"{typ}_est_wo_plan"],
                                 "w/ plan": item[f"{typ}_est_hypo_plan"]})

            data_save = f"/data/wz/index/index_eab/eab_benefit/cost_data/{ben}_cost_data_{typ}.json"
            with open(data_save, "w") as wf:
                json.dump(data_pre, wf, indent=2)


def split_cost_data(seed=666):
    random.seed(seed)

    for typ in ["src", "tgt"]:  # ["src", "tgt"]
        for ben in tqdm(bench):  # tqdm(bench)
            data_load = f"/data/wz/index/index_eab/eab_benefit/cost_data/{ben}/{ben}_cost_data_{typ}.json"
            with open(data_load, "r") as rf:
                data = json.load(rf)

            train_num, valid_num = int(0.7 * len(data)), int(0.2 * len(data))

            random.shuffle(data)

            data_save = f"/data/wz/index/index_eab/eab_benefit/cost_data/{ben}/{ben}_cost_data_{typ}_train.json"
            with open(data_save, "w") as wf:
                json.dump(data[:train_num], wf, indent=2)

            data_save = f"/data/wz/index/index_eab/eab_benefit/cost_data/{ben}/{ben}_cost_data_{typ}_valid.json"
            with open(data_save, "w") as wf:
                json.dump(data[train_num:train_num + valid_num], wf, indent=2)

            data_save = f"/data/wz/index/index_eab/eab_benefit/cost_data/{ben}/{ben}_cost_data_{typ}_test.json"
            with open(data_save, "w") as wf:
                json.dump(data[train_num + valid_num:], wf, indent=2)


if __name__ == "__main__":
    # pre_cost_file()
    # organize_cost_file()
    # pre_corr_data()
    # ana_corr_data()

    # pre_cost_data()
    split_cost_data()
