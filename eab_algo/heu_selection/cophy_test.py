# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: cophy_test
# @Author: Wei Zhou
# @Time: 2023/12/11 14:37


import amplpy as ampl

# Create an AMPL instance
ampl_env = ampl.Environment(binary_directory="/data1/wz/ampl.linux-intel64")
ampl_instance = ampl.AMPL(environment=ampl_env)

# Load your AMPL model file
ampl_instance.read("/data1/wz/index/index_eab/eab_data/heu_run_conf/cophy_ampl_model.mod")

# Optionally, load data if your model requires it
ampl_instance.readData("/data1/wz/index/index_eab/eab_olap/benchmark_results/cophy/cophy_input__width2__per_query1.txt")

ampl_instance.option["solver"] = "highs"

# Solve the optimization problem
ampl_instance.solve()
# ampl_instance.get_value("solve_result")
# ampl_instance.get_objective("overall_costs_of_queries").value()
# ampl_instance.get_variable("x").to_pandas().values

pass
