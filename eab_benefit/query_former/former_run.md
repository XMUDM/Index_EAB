
## TPC-H

## Train

```
python former_train.py

--exp_id exp_former_tpch_tgt_ep500_bat1024 
--gpu_no 0 --seed 666

--epoch_num 500 --batch_size 1024

--train_data_file /data/wz/index/index_eab/eab_benefit/cost_data/tpch/tpch_cost_data_tgt_train.json
--valid_data_file /data/wz/index/index_eab/eab_benefit/cost_data/tpch/tpch_cost_data_tgt_valid.json

--encoding_load /data/wz/index/index_eab/eab_benefit/query_former/data/tpch/encoding_tpch.pt

```

## Infer

```
```

## TPC-DS

## Train

```
python former_train.py

--exp_id exp_former_tpcds_tgt_ep500_bat1024 
--gpu_no 5 --seed 666

--epoch_num 500 --batch_size 1024

--train_data_file /data1/wz/index/index_eab/eab_benefit/cost_data/tpcds/tpcds_cost_data_tgt_train.json
--valid_data_file /data1/wz/index/index_eab/eab_benefit/cost_data/tpcds/tpcds_cost_data_tgt_valid.json

--encoding_load /data1/wz/index/index_eab/eab_benefit/query_former/data/tpcds/encoding_tpcds.pt

```

## Infer

```
```

## JOB

## Train

```
python former_train.py

--exp_id exp_former_job_tgt_ep500_bat1024 
--gpu_no 0 --seed 666

--epoch_num 500 --batch_size 1024

--train_data_file /data2/wz/index/index_eab/eab_benefit/cost_data/job/job_cost_data_tgt_train.json
--valid_data_file /data2/wz/index/index_eab/eab_benefit/cost_data/job/job_cost_data_tgt_valid.json

--encoding_load /data2/wz/index/index_eab/eab_benefit/query_former/data/job/encoding_job.pt

```

## Infer

```
```
