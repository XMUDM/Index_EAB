# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: tree_cost_infer
# @Author: Wei Zhou
# @Time: 2022/8/18 9:58

import random
import numpy as np

import torch
from torch.utils.data import random_split

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

from index_eab.eab_benefit.tree_model.tree_cost_model import MLP, XGBoost, LightGBM, RandomForest, RegLSTM
from index_eab.eab_benefit.tree_model.tree_cost_utils.tree_cost_loss import QError, xgb_QError, lgb_QError, cal_mape
from index_eab.eab_benefit.tree_model.tree_cost_dataset import PlanPairDataset, unnormalize, normalize

# todo: hyper-params.
data_type = "syntheti"
label_type = "ratio"  # ["ratio", "diff_ratio", "cla", "raw"]
plan_num = 2
feat_conn = "concat"
feat_chan = "cost_row"  # ["cost", "row", "cost_row", "tra_order", "seq_ind"]
cla_min_ratio = 0.2

# data_load = "/data/wz/index/attack/data_resource/visrew_qplan/server103/" \
#             "103prenc_pgs200_plan4_filter20cr_split_vec_ran2w_res4755.pt"
# data_load = "/data/wz/index/attack/data_resource/visrew_qplan/server103/" \
#             "103prenc_pgs200_plan2_filter_split_format_vec_woindex_res4755.pt"

feat_id = "order"
scale_load = "/data/wz/index/attack/visual_rewrite/cost_exp_res/" \
             f"{feat_id}_pl2_con_ratio_reg_mlp/data/train_scale_data.pt"

data_load = "/data/wz/index/attack/data_resource/visrew_qplan/server103_all/plan2_103prenc_pgs200_nopre_pgd200_res4755.pt"
scale_load = "/data/wz/index/attack/visual_rewrite/cost_exp_res/new_exp_mse_minmax/data/train_scale_data.pt"
scale_load = "/data/wz/index/attack/visual_rewrite/cost_exp_res/new_exp_log_minmax/data/train_scale_data.pt"

data_load = "/data/wz/index/attack/visual_rewrite/cost_exp_res/cost_row_pl2_con_ratio_reg_lgb/data/train_data.pt"
scale_load = "/data/wz/index/attack/visual_rewrite/cost_exp_res/cost_row_pl2_con_ratio_reg_lgb/data/train_scale_data.pt"

model_type = "LightGBM"  # ["Optimizer", "MLP", "RegGCN", "RegLSTM", "XGBoost", "LightGBM", "RandomForest"]
if model_type == "Optimizer":
    model_load = "/data/wz/index/attack/visual_rewrite/cost_exp_res/" \
                 "103prenc_pgs200_valid_plan4_filter20cr_split_vec_ran2w_res4755.pt"
    model_load = "/data/wz/index/attack/data_resource/visrew_qplan/server103/" \
                 "103prenc_pgs200_plan2_filter_split_format_vec_woindex_res4755.pt"
if model_type == "MLP":
    model_load = "/data/wz/index/attack/visual_rewrite/cost_exp_res/" \
                 f"{feat_id}_pl2_con_ratio_reg_mlp/model/cost_train_200.pt"
    model_load = "/data/wz/index/attack/visual_rewrite/cost_exp_res/new_exp_mse_minmax/model/cost_train_100.pt"
    model_load = "/data/wz/index/attack/visual_rewrite/cost_exp_res/new_exp_log_minmax/model/cost_train_100.pt"
elif model_type == "LSTM":
    model_load = "/data/wz/index/attack/visual_rewrite/cost_exp_res/" \
                 f"{feat_id}_pl2_con_ratio_reg_lstm/model/cost_train_200.pt"
elif model_type == "XGBoost":
    model_load = "/data/wz/index/attack/visual_rewrite/cost_exp_res/" \
                 f"{feat_id}_pl2_con_ratio_reg_xgb/model/"
elif model_type == "LightGBM":
    model_load = "/data/wz/index/attack/visual_rewrite/cost_exp_res/" \
                 f"{feat_id}_pl2_con_ratio_reg_lgb/model/"
    model_load = "/data/wz/index/attack/visual_rewrite/cost_exp_res/" \
                 "cost_row_pl2_con_ratio_reg_lgb/model/"

# label_type: ratio
if label_type == "ratio":
    min_card_log, max_card_log = -16.997074, 2.9952054
# label_type: raw
elif label_type == "raw":
    min_card_log, max_card_log = -5.809143, 12.096035

# todo: 4. set the torch random_seed (consistent with args!!!).
seed = 666
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# todo: 1. Data preparation (unnormalized).
if data_type == "synthetic":
    x = torch.unsqueeze(torch.linspace(-1, 1, 5000), dim=1)
    y = (x.pow(2) + 0.2 * torch.rand(x.size())).squeeze(-1).numpy()
    X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)

    X_train = np.array(X_train).reshape(-1, 1)
    X_valid = np.array(X_valid).reshape(-1, 1)
else:
    data = torch.load(data_load)

    # dataset = PlanPairDataset(data, plan_num=plan_num, feat_chan=feat_chan, feat_conn=feat_conn,
    #                           label_type=label_type, cla_min_ratio=cla_min_ratio)

    dataset = data

    train_set, valid_set = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    train_set, valid_set = list(train_set), list(valid_set)

    X_train = [sample[0] for sample in train_set]
    y_train = [sample[1] for sample in train_set]
    X_valid = [sample[0] for sample in valid_set]
    y_valid = [sample[1] for sample in valid_set]

    # Scale features of X according to feature_range
    # if feat_chan in ["cost", "row", "cost_row"]:
    #     scaler = torch.load(scale_load)
    #     X_train = np.array(scaler.transform(X_train), dtype=np.float32)
    #     X_valid = np.array(scaler.transform(X_valid), dtype=np.float32)

    train_set, valid_set = list(zip(X_train, y_train)), list(zip(X_valid, y_valid))

# todo: 2. Model load.
if model_type == "Optimizer":
    data = torch.load(model_load)
    y_pred = [dat[1]["est_cost"] / dat[0]["est_cost"] for dat in data]
    y_act = [dat[1]["act_cost"] / dat[0]["act_cost"] for dat in data]

    qerror = QError()(torch.from_numpy(np.array(y_pred)),
                      torch.from_numpy(np.array(y_valid)),
                      out="raw", min_val=None, max_val=None)

    metric = [torch.mean(qerror).item(), torch.median(qerror).item(), torch.quantile(qerror, 0.9).item(),
              torch.quantile(qerror, 0.95).item(), torch.quantile(qerror, 0.99).item(), torch.max(qerror).item()]
    print("\t".join(map(str, metric)))

elif model_type == "MLP":
    inp_dim = len(train_set[0][0])
    model = MLP(inp_dim=inp_dim, hid_dim=128)

    model_source = torch.load(model_load,
                              map_location=lambda storage, loc: storage)
    model.load_state_dict(model_source["model"])
    model = model.cpu()
    model.eval()

    X_valid = torch.from_numpy(np.array(X_valid))
    y_pred = model(X_valid).squeeze(-1).detach().numpy()

    y_valid_norm = [normalize(y, min_card_log, max_card_log) for y in y_valid]
    y_pred_unnorm = [unnormalize(y, min_card_log, max_card_log) for y in y_pred]

    # qerror = QError()(torch.from_numpy(np.array(y_pred)),
    #                   torch.from_numpy(np.array(y_valid_norm)),
    #                   out="raw", min_val=min_card_log, max_val=max_card_log)

    qerror = QError()(torch.from_numpy(np.array(y_pred_unnorm)),
                      torch.from_numpy(np.array(y_valid)),
                      out="raw", min_val=None, max_val=None)

    # tensor(111.3453)
    # torch.mean(QError()(torch.from_numpy(np.array([np.exp(pred) for pred in y_pred])),
    #                     torch.from_numpy(np.array(y_valid)),
    #                     out="raw", min_val=None, max_val=None))

    metric = [torch.mean(qerror).item(), torch.median(qerror).item(), torch.quantile(qerror, 0.9).item(),
              torch.quantile(qerror, 0.95).item(), torch.quantile(qerror, 0.99).item(), torch.max(qerror).item()]
    print("\t".join(map(str, metric)))
    pass


elif model_type == "LSTM":
    inp_dim = len(train_set[0][0])
    model = RegLSTM(input_dim=inp_dim, hidden_dim=128)

    model_source = torch.load(model_load,
                              map_location=lambda storage, loc: storage)
    model.load_state_dict(model_source["model"])
    model = model.cpu()
    model.eval()

    X_valid = torch.from_numpy(np.array(X_valid))
    y_pred = model(X_valid).squeeze(-1).detach().numpy()

    y_valid_norm = [normalize(y, min_card_log, max_card_log) for y in y_valid]
    y_pred_unnorm = [unnormalize(y, min_card_log, max_card_log) for y in y_pred]

    qerror = QError()(torch.from_numpy(np.array(y_pred)),
                      torch.from_numpy(np.array(y_valid_norm)),
                      out="raw", min_val=min_card_log, max_val=max_card_log)

    metric = [torch.mean(qerror).item(), torch.median(qerror).item(), torch.quantile(qerror, 0.9).item(),
              torch.quantile(qerror, 0.95).item(), torch.quantile(qerror, 0.99).item(), torch.max(qerror).item()]
    print("\t".join(map(str, metric)))
    pass


elif model_type == "XGBoost":
    model_path = model_load + "reg_xgb_cost.xgb.model"
    xgb = XGBoost(path=model_path)
    y_pred = xgb.estimate(X_valid)

    y_valid_norm = [normalize(y, min_card_log, max_card_log) for y in y_valid]
    y_pred_unnorm = [unnormalize(y, min_card_log, max_card_log) for y in y_pred]

    # qerror = QError()(torch.from_numpy(np.array(y_pred)),
    #                   torch.from_numpy(np.array(y_valid_norm)),
    #                   out="raw", min_val=min_card_log, max_val=max_card_log)

    qerror = QError()(torch.from_numpy(np.array(y_pred_unnorm)),
                      torch.from_numpy(np.array(y_valid)),
                      out="raw", min_val=None, max_val=None)

    metric = [torch.mean(qerror).item(), torch.median(qerror).item(), torch.quantile(qerror, 0.9).item(),
              torch.quantile(qerror, 0.95).item(), torch.quantile(qerror, 0.99).item(), torch.max(qerror).item()]
    print("\t".join(map(str, metric)))
    pass

elif model_type == "LightGBM":
    model_path = model_load + "reg_lgb_cost.lgb.model"
    lgb = LightGBM(model_path)
    y_pred = lgb.estimate(X_valid)
    # (y_true - y_pred) ** 2

    y_valid_norm = [normalize(y, min_card_log, max_card_log) for y in y_valid]
    y_pred_unnorm = [unnormalize(y, min_card_log, max_card_log) for y in y_pred]

    # qerror = QError()(torch.from_numpy(np.array(y_pred)),
    #                   torch.from_numpy(np.array(y_valid_norm)),
    #                   out="raw", min_val=min_card_log, max_val=max_card_log)

    qerror = QError()(torch.from_numpy(np.array(y_pred_unnorm)),
                      torch.from_numpy(np.array(y_valid)),
                      out="raw", min_val=None, max_val=None)

    metric = [torch.mean(qerror).item(), torch.median(qerror).item(), torch.quantile(qerror, 0.9).item(),
              torch.quantile(qerror, 0.95).item(), torch.quantile(qerror, 0.99).item(), torch.max(qerror).item()]
    print("\t".join(map(str, metric)))
    pass

elif model_type == "RandomForest":
    model_path = model_load + "reg_rf_cost.rf.model"
    rf = RandomForest(model_path)
    y_pred = rf.estimate(X_valid)

    qerror = QError()(torch.from_numpy(np.array(y_pred)),
                      torch.from_numpy(np.array(y_valid)),
                      out="raw")
    pass

# todo: 3. Metric calculation.
# np.sum(model(torch.from_numpy(np.array(X_valid))).squeeze(-1).detach().numpy() == y_valid) / len(y_valid)
accuracy = accuracy_score(y_pred, y_valid)
precision = precision_score(y_pred, y_valid)
recall = recall_score(y_pred, y_valid)
f1 = f1_score(y_pred, y_valid)

print(1)
