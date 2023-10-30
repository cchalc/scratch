# Databricks notebook source
# MAGIC %md ### Read in data

# COMMAND ----------

spark.sql("use catalog cjc")
spark.sql("use schema scratch")

# COMMAND ----------

tables = spark.catalog.listTables()
tables_in_schema = [t.name for t in tables]
print(tables_in_schema)



# COMMAND ----------

sdf_walmart = spark.read.table("m5_final_cleaned_filtered")
display(sdf_walmart)

# COMMAND ----------

# MAGIC %md ### Create random time-series per model for the Demo

# COMMAND ----------

# Experiment setup - make sure num_items and items_per_model are divisible
num_items = 200  # Max number of item time series to load, full dataset has 30490 which is overkill
items_per_model = 100  # Number of item time series per model
num_batches = 1  # num trials = max_concurrent_trials * num_batches

# COMMAND ----------

from delta.tables import DeltaTable
from pyspark.sql.functions import col
from pyspark.sql import Window
import pyspark.sql.functions as F

window_spec = Window.orderBy('state_id', 'store_id', 'cat_id', 'dept_id', 'item_id')
sdf_walmart_with_model_num = sdf_walmart.withColumn("item_num", F.dense_rank().over(window_spec))  # A unique item number based on the window
sdf_walmart_with_model_num = sdf_walmart_with_model_num.filter(sdf_walmart_with_model_num.item_num <= num_items)
sdf_walmart_with_model_num = sdf_walmart_with_model_num.withColumn("model_num", F.ceil(F.col("item_num") / items_per_model))
sdf_walmart_with_model_num = sdf_walmart_with_model_num.withColumn('y', F.col('sell_price')*F.col('sale_quantity'))
sdf_walmart_with_model_num.cache()
print(sdf_walmart.count())
sdf_walmart_with_model_num.display()

# COMMAND ----------



# COMMAND ----------


