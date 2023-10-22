# Databricks notebook source
# MAGIC
# MAGIC %pip install kaggle

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

kaggle_key = dbutils.secrets.get("tokens", "kaggle_key")

# COMMAND ----------

import os
os.environ['KAGGLE_USERNAME']="cchalc"
os.environ['KAGGLE_KEY']=kaggle_key

# COMMAND ----------

base_path = "/Users/christopher.chalcraft@databricks.com/zeroshot"

# COMMAND ----------

dbutils.fs.mkdirs("/Users/christopher.chalcraft@databricks.com/zeroshot")

# COMMAND ----------

# %sh 
# kaggle competitions download -c learn-ai-bbc -p /dbfs/Users/christopher.chalcraft@databricks.com/zeroshot

# COMMAND ----------

# %sh
# cd /dbfs/Users/christopher.chalcraft@databricks.com/zeroshot &&  unzip /dbfs/Users/christopher.chalcraft@databricks.com/zeroshot/learn-ai-bbc.zip 

# COMMAND ----------

# MAGIC %sh ls /dbfs/Users/christopher.chalcraft@databricks.com/zeroshot/

# COMMAND ----------

train_df = (spark
            .read
            .option("header", True)
            .csv("/Users/christopher.chalcraft@databricks.com/zeroshot/BBC News Train.csv")
)

# COMMAND ----------

display(train_df)

# COMMAND ----------

test_df = (spark
            .read
            .option("header", True)
            .csv("/Users/christopher.chalcraft@databricks.com/zeroshot/BBC News Test.csv")
)

# COMMAND ----------

display(test_df)

# COMMAND ----------

sample_df = (spark
             .read
             .option("header", True)
             .csv("/Users/christopher.chalcraft@databricks.com/zeroshot/BBC News Sample Solution.csv")
)

# COMMAND ----------

display(sample_df)

# COMMAND ----------

# make sure these are created beforehand
spark.sql("use catalog cjc")
spark.sql("use schema scratch")

# COMMAND ----------

train_df.write.saveAsTable("bbc_news_train")

# COMMAND ----------

test_df.write.saveAsTable("bbc_news_test")

# COMMAND ----------

sample_df.write.saveAsTable("bbc_news_sample_solution")

# COMMAND ----------


