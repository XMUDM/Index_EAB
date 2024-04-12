
## TPC-H

```
python isum_main.py

--work_load /data1/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_n10_eval.json
--work_save /data1/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_n10_eval_compressed.json

--schema_file /data1/wz/index/index_eab/eab_data/db_info_conf/schema_tpcds_1gb.json
--db_conf_file /data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf

--host 59.77.5.98
--port 5432
--db_name tpch_1gb103

--user wz
--password ai4db2021

```

## TPC-DS

```
python isum_main.py

--work_load /data1/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_work_temp_multi_w79_n10_eval.json
--work_save /data1/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_work_temp_multi_w79_n10_eval_compressed.json

--schema_file /data1/wz/index/index_eab/eab_data/db_info_conf/schema_tpcds_1gb.json
--db_conf_file /data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf

--host 59.77.5.98
--port 5432
--db_name tpcds_1gb103

--user wz
--password ai4db2021

```