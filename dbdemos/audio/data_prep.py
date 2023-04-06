# Databricks notebook source
# MAGIC %sql CREATE CATALOG IF NOT EXISTS cjc;
# MAGIC USE CATALOG cjc

# COMMAND ----------

# %sql 
# DROP SCHEMA IF EXISTS libri CASCADE;
# CREATE SCHEMA IF NOT EXISTS libri;

# COMMAND ----------

# MAGIC %sh 
# MAGIC wget https://www.openslr.org/resources/12/dev-clean.tar.gz

# COMMAND ----------

# MAGIC %sh
# MAGIC tar -zxvf dev-clean.tar.gz

# COMMAND ----------

# %fs ls /dbfs/Users/christopher.chalcraft@databricks.com

# COMMAND ----------

# MAGIC %sh
# MAGIC cp -r LibriSpeech/ /dbfs/tutorials/

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/tutorials/LibriSpeech/dev-clean/*/*/*.flac

# COMMAND ----------

import os
from glob import glob
audio_files = [y for x in os.walk("/dbfs/tutorials/LibriSpeech/dev-clean/") for y in glob(os.path.join(x[0], '*.flac'))]


# COMMAND ----------

print(audio_files)

# COMMAND ----------

import pandas as pd
pandas_df = pd.DataFrame(pd.Series(audio_files),columns=["path"])

# COMMAND ----------

df = spark.createDataFrame(pandas_df)

# COMMAND ----------

display(df)

# COMMAND ----------

df_with_ids = df.selectExpr("path", "uuid() as id")

# COMMAND ----------

df_with_ids.write.saveAsTable("cjc.libri.paths_with_ids")

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM cjc.libri.paths_with_ids

# COMMAND ----------

# MAGIC %md
# MAGIC # Create binary table

# COMMAND ----------

binary_df = spark.readStream.format("cloudFiles") \
  .option("cloudFiles.format", "binaryFile") \
  .option("recursiveFileLookup", "true") \
  .option("pathGlobFilter", "*.flac") \
  .load("/tutorials/LibriSpeech/dev-clean") \
  .repartition(32)

# COMMAND ----------

binary_df = binary_df.selectExpr("*", "uuid() as id")

# COMMAND ----------

binary_df.writeStream.format("delta")\
  .option("checkpointLocation", "/tmp/delta/tutorials/librispeech")\
  .trigger(once=True)\
  .toTable("cjc.libri.binary_audio_with_ids")

# COMMAND ----------

df = spark.read.table("cjc.libri.binary_audio_with_ids")
display(df)

# COMMAND ----------


