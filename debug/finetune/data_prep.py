# Databricks notebook source
# MAGIC %fs ls /databricks-datasets

# COMMAND ----------

# MAGIC
# MAGIC %pip install kaggle

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# create a scope or use an existing scope to put keys
# databricks secrets put --scope tokens --key kaggle_key --string-value <string value>
kaggle_key = dbutils.secrets.get("tokens", "kaggle_key")
project_name = 'finetune'

# COMMAND ----------

import os
os.environ['KAGGLE_USERNAME']="cchalc"
os.environ['KAGGLE_KEY']=kaggle_key

# COMMAND ----------

base_path = f"/Users/christopher.chalcraft@databricks.com/{project_name}"

# COMMAND ----------

dbutils.fs.mkdirs(f"/Users/christopher.chalcraft@databricks.com/{project_name}")

# COMMAND ----------

# MAGIC %sh 
# MAGIC kaggle competitions download -c learn-ai-bbc -p /dbfs/Users/christopher.chalcraft@databricks.com/finetune

# COMMAND ----------

# MAGIC %fs ls dbfs:/Users/christopher.chalcraft@databricks.com/finetune

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /dbfs/Users/christopher.chalcraft@databricks.com/finetune &&  unzip -o /dbfs/Users/christopher.chalcraft@databricks.com/finetune/learn-ai-bbc.zip 

# COMMAND ----------

# MAGIC %sh ls /dbfs/Users/christopher.chalcraft@databricks.com/finetune/

# COMMAND ----------

train_df = (spark
            .read
            .option("header", True)
            .csv(f"/Users/christopher.chalcraft@databricks.com/{project_name}/BBC News Train.csv")
)

# COMMAND ----------

display(train_df)

# COMMAND ----------

test_df = (spark
            .read
            .option("header", True)
            .csv(f"/Users/christopher.chalcraft@databricks.com/{project_name}/BBC News Test.csv")
)

# COMMAND ----------

display(test_df)

# COMMAND ----------

sample_df = (spark
             .read
             .option("header", True)
             .csv(f"/Users/christopher.chalcraft@databricks.com/{project_name}/BBC News Sample Solution.csv")
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


