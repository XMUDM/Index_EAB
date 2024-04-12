# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: former_infer
# @Author: Wei Zhou
# @Time: 2023/10/7 11:06

import os
import json
import argparse
import logging

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append("/data/wz/index/")

from index_eab.eab_benefit.query_former.former_train import get_parser
from index_eab.eab_benefit.query_former.model.model import QueryFormer

from index_eab.eab_benefit.query_former.model import util
from index_eab.eab_benefit.query_former.model.util import Normalizer, set_logger
from index_eab.eab_benefit.query_former.model.util import seed_everything

from index_eab.eab_benefit.query_former.model.dataset import PlanTreeDataset
from index_eab.eab_benefit.query_former.model.trainer import train, evaluate
from index_eab.eab_benefit.query_former.model.database_util import get_hist_file, get_job_table_sample
from index_eab.eab_benefit.benefit_utils.benefit_const import alias2table_tpch, alias2table_job, alias2table_tpcds


def eval_work(workload, methods):

    ds = PlanTreeDataset(plan_df, workload_csv,
                         methods['encoding'], methods['hist_file'], methods['cost_norm'],
                         methods['cost_norm'], 'cost', table_sample)

    eval_score = evaluate(methods['model'], ds, methods['bs'], methods['cost_norm'], methods['device'], True)
    return eval_score, ds


def get_est_res(args):
    device = torch.device(f"cuda:{args.gpu_no}" if torch.cuda.is_available() else "cpu")

    encoding = torch.load(args.encoding_load.replace(".pt", "_v2.pt"))
    # encoding_ckpt = torch.load("checkpoints/encoding.pt")
    # encoding = encoding_ckpt["encoding"]

    cost_norm = Normalizer()
    card_norm = Normalizer()

    model = QueryFormer(emb_size=args.embed_size, ffn_dim=args.ffn_dim, head_size=args.head_size,
                        dropout=args.dropout, n_layers=args.n_layers, encoding=encoding,
                        use_sample=args.use_sample, use_hist=args.use_hist, pred_hid=args.pred_hid)
    args.model_load = "/data/wz/index/index_eab/eab_benefit/query_former/exp_res/" \
                      "exp_former_tpch_tgt_ep500_bat1024/model/former_-8663975484835275719.pt.pt"
    if os.path.exists(args.model_load):
        # "checkpoints/cost_model.pt"
        checkpoint = torch.load(args.model_load, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    methods = {
        'get_sample': get_job_table_sample,
        'encoding': encoding,
        'cost_norm': cost_norm,
        'hist_file': None,
        'model': model,
        'device': device,
        'bs': 512,
    }

    _ = eval_work('job-light', methods)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    get_est_res(args)
