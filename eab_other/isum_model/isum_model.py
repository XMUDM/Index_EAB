# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: isum_model
# @Author: Wei Zhou
# @Time: 2024/1/12 21:33

import re

import math
import numpy as np

import itertools

from index_eab.eab_other.isum_model.isum_workload import Workload
from index_eab.eab_other.isum_model.isum_const import ops_join_dict, ops_scan_dict, ops_sort_dict, ops_group_dict

MAX_INDEX_WIDTH = 2


def cal_utility(query, root, seq):
    selectivity = dict()
    scan = [s for s in seq if s["Node Type"] in ops_scan_dict.keys()]
    for sc in scan:
        sc_str = [sc[key] for key in sc.keys() if key != "Plans"]
        pred_num = 1 + len(re.findall(r" and ", str(sc))) + len(re.findall(r" or ", str(sc)))
        for col in query.columns:
            if col.name in str(sc_str) and col.table.name in str(sc_str):
                # if "Relation Name" not in sc.keys() and "Alias" not in sc.keys():
                #     continue
                #
                # if sc["Relation Name"] in row.keys():
                #     select = sc["Plan Rows"] / row[sc["Relation Name"].lower()]
                # elif sc["Alias"] in row.keys():
                #     select = sc["Plan Rows"] / row[sc["Alias"].lower()]

                select = sc["Plan Rows"] / col.table.row

                select = math.pow(select, 1 / pred_num)

                if f"{col.table.name}.{col.name}" not in selectivity:
                    selectivity[f"{col.table.name}.{col.name}"] = 0
                selectivity[f"{col.table.name}.{col.name}"] += (1 - select)

                if col.table.name not in query.selection.keys():
                    query.selection[col.table.name] = set()
                query.selection[col.table.name].add(col)

    join = [s for s in seq if s["Node Type"] in ops_join_dict.keys()]
    for jo in join:
        jo_str = [jo[key] for key in jo.keys() if key != "Plans"]
        for col in query.columns:
            if col.name in str(jo_str) and col.table.name in str(jo_str):
                select = jo["Plan Rows"] / (seq[jo["left"]]["Plan Rows"] * seq[jo["right"]]["Plan Rows"])
                select = math.sqrt(select)

                if f"{col.table.name}.{col.name}" not in selectivity:
                    selectivity[f"{col.table.name}.{col.name}"] = 0
                selectivity[f"{col.table.name}.{col.name}"] += (1 - select)

                if col.table.name not in query.join.keys():
                    query.join[col.table.name] = set()
                query.join[col.table.name].add(col)

    selectivity = sum(selectivity.values()) / len(selectivity.keys())
    utility = selectivity * root["Total Cost"]

    query.utility = utility

    sort = [s for s in seq if s["Node Type"] in ops_sort_dict.keys()]
    for so in sort:
        so_str = [so[key] for key in so.keys() if key != "Plans"]
        for col in query.columns:
            if col.name in str(so_str):
                if col.table.name not in query.orderby.keys():
                    query.orderby[col.table.name] = set()
                query.orderby[col.table.name].add(col)

    group = [s for s in seq if s["Node Type"] in ops_group_dict.keys()]
    for gr in group:
        gr_str = [gr[key] for key in gr.keys() if key != "Plans"]
        for col in query.columns:
            if col.name in str(gr_str):
                if col.table.name not in query.groupby.keys():
                    query.groupby[col.table.name] = set()
                query.groupby[col.table.name].add(col)


def cal_query_feature(query, col_dict):
    feature = np.zeros((len(col_dict, )))

    candidates = dict()
    # R1: selection
    for tbl in query.selection.keys():
        for width in range(1, MAX_INDEX_WIDTH + 1):
            if len(query.selection[tbl]) < width:
                break

            cands = itertools.permutations(query.selection[tbl], width)

            if tbl not in candidates.keys():
                candidates[tbl] = set()
            candidates[tbl] = candidates[tbl].union(cands)

    # R2: join
    for tbl in query.join.keys():
        for width in range(1, MAX_INDEX_WIDTH + 1):
            if len(query.join[tbl]) < width:
                break

            cands = itertools.permutations(query.join[tbl], width)

            if tbl not in candidates.keys():
                candidates[tbl] = set()
            candidates[tbl] = candidates[tbl].union(cands)

    # R3: selection + join
    for tbl in query.selection.keys():
        for width1 in range(1, min(MAX_INDEX_WIDTH, len(query.selection[tbl])) + 1):
            for prefix in itertools.permutations(query.selection[tbl], width1):
                for width2 in range(1, MAX_INDEX_WIDTH + 1 - width1):
                    if tbl not in query.join.keys() or len(query.join[tbl]) < width2:
                        break

                    for cands in itertools.permutations(query.join[tbl], width2):
                        if tbl not in candidates.keys():
                            candidates[tbl] = set()
                        candidates[tbl].add(prefix + cands)

    # R4: join + selection
    for tbl in query.join.keys():
        if len(query.join[tbl]) >= MAX_INDEX_WIDTH:
            continue
        else:
            for width1 in range(1, min(MAX_INDEX_WIDTH, len(query.join[tbl])) + 1):
                for prefix in itertools.permutations(query.join[tbl], width1):
                    for width2 in range(1, MAX_INDEX_WIDTH + 1 - width1):
                        if tbl not in query.selection.keys() or len(query.selection[tbl]) < width2:
                            break

                        for cands in itertools.permutations(query.selection[tbl], width2):
                            if tbl not in candidates.keys():
                                candidates[tbl] = set()
                            candidates[tbl].add(prefix + cands)

    # R5: orderby + selection + join
    for tbl in query.orderby.keys():
        for prefix in itertools.permutations(query.orderby[tbl],
                                             min(MAX_INDEX_WIDTH, len(query.orderby[tbl]))):
            if len(prefix) == MAX_INDEX_WIDTH:
                if tbl not in candidates.keys():
                    candidates[tbl] = set()
                candidates[tbl].add(prefix)
            else:
                if tbl in query.selection.keys():
                    for width1 in range(1, MAX_INDEX_WIDTH + 1 - len(query.orderby[tbl])):
                        for suffix in itertools.permutations(query.selection[tbl], width1):
                            if len(suffix) == MAX_INDEX_WIDTH - len(query.orderby[tbl]) \
                                    or tbl not in query.join.keys():
                                if tbl not in candidates.keys():
                                    candidates[tbl] = set()
                                candidates[tbl].add(prefix + suffix)
                            else:
                                for width2 in range(1, MAX_INDEX_WIDTH + 1 - len(query.orderby[tbl]) - len(
                                        query.selection[tbl])):
                                    for cands in itertools.permutations(query.join[tbl], width2):
                                        if tbl not in candidates.keys():
                                            candidates[tbl] = set()
                                        candidates[tbl].add(prefix + suffix + cands)

    # R6: groupby + selection + join
    for tbl in query.groupby.keys():
        for prefix in itertools.permutations(query.groupby[tbl],
                                             min(MAX_INDEX_WIDTH, len(query.groupby[tbl]))):
            if len(prefix) == MAX_INDEX_WIDTH:
                if tbl not in candidates.keys():
                    candidates[tbl] = set()
                candidates[tbl].add(prefix)
            else:
                if tbl in query.selection.keys():
                    for width1 in range(1, MAX_INDEX_WIDTH + 1 - len(query.groupby[tbl])):
                        for suffix in itertools.permutations(query.selection[tbl], width1):
                            if len(suffix) == MAX_INDEX_WIDTH - len(query.groupby[tbl]) \
                                    or tbl not in query.join.keys():
                                if tbl not in candidates.keys():
                                    candidates[tbl] = set()
                                candidates[tbl].add(prefix + suffix)
                            else:
                                if tbl in query.join.keys():
                                    for width2 in range(1, MAX_INDEX_WIDTH + 1 - len(query.groupby[tbl]) - len(
                                            query.selection[tbl])):
                                        for cands in itertools.permutations(query.join[tbl], width2):
                                            if tbl not in candidates.keys():
                                                candidates[tbl] = set()
                                            candidates[tbl].add(prefix + suffix + cands)

    # R7: orderby + join + selection
    for tbl in query.orderby.keys():
        for prefix in itertools.permutations(query.orderby[tbl],
                                             min(MAX_INDEX_WIDTH, len(query.orderby[tbl]))):
            if len(prefix) == MAX_INDEX_WIDTH:
                break
            else:
                if tbl in query.join.keys():
                    for width1 in range(1, MAX_INDEX_WIDTH + 1 - len(query.orderby[tbl])):
                        for suffix in itertools.permutations(query.join[tbl], width1):
                            if len(suffix) == MAX_INDEX_WIDTH - len(query.orderby[tbl]) \
                                    or tbl not in query.selection.keys():
                                if tbl not in candidates.keys():
                                    candidates[tbl] = set()
                                candidates[tbl].add(prefix + suffix)
                            else:
                                for width2 in range(1, MAX_INDEX_WIDTH + 1 - len(query.orderby[tbl]) - len(
                                        query.join[tbl])):
                                    for cands in itertools.permutations(query.selection[tbl], width2):
                                        if tbl not in candidates.keys():
                                            candidates[tbl] = set()
                                        candidates[tbl].add(prefix + suffix + cands)

    # R8: groupby + join + selection
    for tbl in query.groupby.keys():
        for prefix in itertools.permutations(query.groupby[tbl],
                                             min(MAX_INDEX_WIDTH, len(query.groupby[tbl]))):
            if len(prefix) == MAX_INDEX_WIDTH:
                break
            else:
                if tbl in query.join.keys():
                    for width1 in range(1, MAX_INDEX_WIDTH + 1 - len(query.groupby[tbl])):
                        for suffix in itertools.permutations(query.join[tbl], width1):
                            if len(suffix) == MAX_INDEX_WIDTH - len(query.groupby[tbl]) \
                                    or tbl not in query.selection.keys():
                                if tbl not in candidates.keys():
                                    candidates[tbl] = set()
                                candidates[tbl].add(prefix + suffix)
                            else:
                                for width2 in range(1, MAX_INDEX_WIDTH + 1 - len(query.groupby[tbl]) - len(
                                        query.join[tbl])):
                                    for cands in itertools.permutations(query.selection[tbl], width2):
                                        if tbl not in candidates.keys():
                                            candidates[tbl] = set()
                                        candidates[tbl].add(prefix + suffix + cands)

    for c in query.columns:
        if c.table.name not in candidates.keys():
            continue
        d_t_c = len([cand for cand in candidates[c.table.name] if c in cand])
        feature[col_dict[f"{c.table.name}.{c.name}"]] = (d_t_c / len(candidates[c.table.name])) * c.table.weight
    query.feature = feature


def cal_influence(workload, query):
    influence = 0
    for q in workload:
        if q == query:
            continue

        feat_array = np.array((query.feature, q.feature))
        s = np.sum(np.min(feat_array, axis=0)) / np.sum(np.max(feat_array, axis=0))

        influence += (s * q.utility)

    query.influence = influence


def upd_utility(workload, query):
    for q in workload.queries:
        if q == query:
            # query.utility = -1
            continue
        else:
            feat_array = np.array((query.feature, q.feature))
            s = np.sum(np.min(feat_array, axis=0)) / np.sum(np.max(feat_array, axis=0))

            q.utility -= (s * q.utility)


def upd_query_feature(workload, query, col_dict):
    for q in workload.queries:
        for c in query.columns:
            q.feature[col_dict[f"{c.table.name}.{c.name}"]] = 0.


def cal_work_feature(workload):
    feature = np.zeros((len(workload.queries[0].feature),))
    for query in workload.queries:
        feature = np.sum((feature, query.utility * query.feature), axis=0)

    workload.feature = feature


def ass_weight(w, w_k):
    w_u = Workload([q for q in w.queries if q not in w_k])
    cal_work_feature(w_u)

    remove = list()
    total_benefits = 0
    while len(remove) < len(w_k):
        max_benefit = -1
        max_benefit_query = None
        for q in w_k:
            if q in remove:
                continue

            feat_array = np.array((q.feature, w_u.feature))
            benefit = q.utility + np.sum(np.min(feat_array)) / np.sum(np.max(feat_array))

            if benefit > max_benefit:
                max_benefit = benefit
                max_benefit_query = q

        total_benefits += max_benefit
        max_benefit_query.weight = max_benefit

        w_u.queries.append(max_benefit_query)
        remove.append(max_benefit_query)

    for q in w_k:
        q.weight = q.weight / total_benefits
