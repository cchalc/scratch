# Databricks notebook source
# MAGIC %md ### HIGSS dataset

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC sudo apt-get -y install --no-install-recommends libboost-filesystem-dev

# COMMAND ----------

from sklearn.datasets import load_svmlight_file

X_train, y_train = load_svmlight_file("higgs.train")
X_test, y_test = load_svmlight_file("higgs.test")

# COMMAND ----------


