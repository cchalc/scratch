# Databricks notebook source
# MAGIC %md #### Refrences
# MAGIC - https://blog.vlgdata.io/post/nlp_hugging_face/
# MAGIC - https://github.com/datatrigger/nlp_hugging_face/blob/main/bbc_news.ipynb
# MAGIC - https://www.kaggle.com/competitions/learn-ai-bbc/data

# COMMAND ----------

# MAGIC %md ### Set up data

# COMMAND ----------

spark.sql("use catalog cjc")
spark.sql("use schema scratch")

# COMMAND ----------

import pandas as pd
import os

os.environ['TRANSFORMERS_CACHE'] = '/Volumes/cjc/scratch/llm_cache'

# COMMAND ----------

df = spark.read.table("bbc_news_train")
display(df)

# COMMAND ----------

df.select('Category').distinct().show()

# COMMAND ----------

from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

# COMMAND ----------

tokenizer.max_model_input_sizes

# COMMAND ----------

# Convert the Spark DataFrame column to a list
text_list = df.select('Text').rdd.flatMap(lambda x: x).collect()

# Tokenize the list of strings
tokenized_articles_lengths = pd.DataFrame({'length': list(map(len, tokenizer(text_list, truncation=False, padding=False)['input_ids']))})

# Print summary statistics of the resulting DataFrame
tokenized_articles_lengths.describe()

# COMMAND ----------

candidate_labels = df.select('Category').distinct().rdd.flatMap(lambda x: x).collect()
# print(labels)
# ['sport', 'politics', 'entertainment', 'business', 'tech']

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, StringType
from transformers import pipeline
import pandas as pd
import torch

# COMMAND ----------

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1

# Define the zero-shot classification model
zero_shot_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

# COMMAND ----------

broadcast_pipeline = spark.sparkContext.broadcast(zero_shot_classifier)

# COMMAND ----------

# Define a Pandas UDF to perform zero-shot classification
@pandas_udf('string')
def classify_text(texts: pd.Series) -> pd.Series:
    pipe = broadcast_pipeline.value(
      ("classify: " + texts).to_list()
                      ,candidate_labels
                      ,batch_size=1)
    labels = [result['labels'][0] for result in pipe]
    return pd.Series(labels)

# COMMAND ----------

sc.setJobDescription("single classification")
result_df = (df
             .limit(1)
             .cache()
             .withColumn(
               "label_pred_zero_shot", classify_text(df.Text)
               )
             )

# COMMAND ----------

display(result_df)

# COMMAND ----------

# MAGIC %md ### Run on small sample

# COMMAND ----------

sc.setJobDescription("small sample classification")
result_df = (df
             .limit(10)
             .cache()
             .withColumn(
               "label_pred_zero_shot", classify_text(df.Text)
               )
             )
display(result_df)

# COMMAND ----------

# MAGIC %md ### Run on larger dataset and modify batch size

# COMMAND ----------

# Define a Pandas UDF to perform zero-shot classification
@pandas_udf('string')
def classify_text(texts: pd.Series) -> pd.Series:
    pipe = broadcast_pipeline.value(
      ("classify: " + texts).to_list()
                      ,candidate_labels
                      ,batch_size=8)
    labels = [result['labels'][0] for result in pipe]
    return pd.Series(labels)

# COMMAND ----------

sc.setJobDescription("large dataset classification")
result_df = (df
             .limit(200)
             .cache()
             .withColumn(
               "label_pred_zero_shot", classify_text(df.Text)
               )
             )
display(result_df)

# COMMAND ----------

# MAGIC %md ### Full dataset

# COMMAND ----------

sc.setJobDescription("full dataset classification")
result_df = (df
             .cache()
             .withColumn(
               "label_pred_zero_shot", classify_text(df.Text)
               )
             )
display(result_df)

# COMMAND ----------

# MAGIC %md ### Summary
# MAGIC #### Single node GPU
# MAGIC **Specs**
# MAGIC ```
# MAGIC Summary
# MAGIC 1 Driver 16 GB Memory, 4 Cores
# MAGIC Runtime 14.1.x-gpu-ml-scala2.12
# MAGIC Unity Catalog g4dn.xlarge 0.71 DBU/h
# MAGIC ```
# MAGIC
# MAGIC **Runs**
# MAGIC - Single Classification (1): Command took 32.20 seconds
# MAGIC - Small Sample (10): Command took 37.62 seconds
# MAGIC - Larger Sample (200, batch size 8): Command took 3.76 minutes
# MAGIC - Full dataset (1490, batch size 8): Command took 25.26 minutes

# COMMAND ----------


