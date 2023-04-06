# Databricks notebook source
# MAGIC %md
# MAGIC #Requirements
# MAGIC - GPU cluster
# MAGIC - ffmpeg installed (via init script at `dbfs:/tutorials/LibriSpeech/install_ffmpeg.sh`)

# COMMAND ----------

# MAGIC %pip install --upgrade transformers

# COMMAND ----------

# MAGIC %pip install whisper

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Load data

# COMMAND ----------

df = spark.read.table("cjc.libri.paths_with_ids")

# COMMAND ----------

pandas_df = df.toPandas()

# COMMAND ----------

pandas_df

# COMMAND ----------

pandas_df["path"][0]

# COMMAND ----------

# Imports
import os
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
import pandas as pd
# os.path.getsize("/dbfs/tutorials/LibriSpeech/dev-clean/251/118436/251-118436-0012.flac")

def get_size(row):
  return os.path.getsize(pandas_df["path"][row])

get_size(2)
# pandas_df['filesize'] = pandas_df.apply(lambda row: get_size(row), axis=1)

# COMMAND ----------

pandas_df['filesize'] = pandas_df.apply(lambda row: os.path.getsize(row["path"]), axis=1)

# COMMAND ----------

display(pandas_df)

# COMMAND ----------

pandas_df.filesize.sum() / 1000000

# COMMAND ----------

# MAGIC %md
# MAGIC # Load model pipeline

# COMMAND ----------

import torch
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-medium",
  chunk_length_s=30,
  device=device,
)


# COMMAND ----------


sample = pandas_df["path"][0]

prediction = pipe(sample)
print(prediction)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create udf

# COMMAND ----------

broadcast_pipeline = spark.sparkContext.broadcast(pipe)

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import pandas_udf

@pandas_udf("string")
def transcribe_udf(paths: pd.Series) -> pd.Series:
  pipe = broadcast_pipeline.value
  transcriptions = [result['text'] for result in pipe(paths.to_list(), batch_size=1)]
  return pd.Series(transcriptions)

# COMMAND ----------

# MAGIC %md
# MAGIC # Run on sample

# COMMAND ----------

sc.setJobDescription("single transription")
transcribed = df.limit(10).cache().select(df.path, df.id, transcribe_udf(df.path))
transcribed.cache()

# COMMAND ----------

display(transcribed)

# COMMAND ----------

# MAGIC %md
# MAGIC # Run on full data

# COMMAND ----------

transcribed = df.repartition(16).select(df.path, df.id, transcribe_udf(df.path).alias('transcription'))
transcribed.cache()

# COMMAND ----------

sc.setJobDescription("full transription")
transcribed.write.saveAsTable("cjc.libri.transcriptions")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM cjc.libri.transcriptions

# COMMAND ----------

# MAGIC %md
# MAGIC # Using binary columns instead of references

# COMMAND ----------

binary_df = spark.read.table("cjc.libri.binary_audio_with_ids")


# COMMAND ----------

one_df = binary_df.limit(1).cache()

# COMMAND ----------

display(one_df)

# COMMAND ----------

one_transcription = one_df.select(one_df.path, one_df.id, transcribe_udf(one_df.content))
display(one_transcription)
