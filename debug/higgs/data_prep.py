# Databricks notebook source
# MAGIC %sh
# MAGIC
# MAGIC git clone https://github.com/guolinke/boosting_tree_benchmarks.git
# MAGIC cd boosting_tree_benchmarks/data
# MAGIC wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
# MAGIC gunzip -fv HIGGS.csv.gz
# MAGIC python higgs2libsvm.py
# MAGIC cd ../..
# MAGIC ln -s boosting_tree_benchmarks/data/higgs.train
# MAGIC ln -s boosting_tree_benchmarks/data/higgs.test

# COMMAND ----------

# MAGIC %sh 
# MAGIC
# MAGIC ls -l 

# COMMAND ----------


