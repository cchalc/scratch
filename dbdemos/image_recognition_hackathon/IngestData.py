# Databricks notebook source
dbutils.fs.ls("/mnt/images/")

# COMMAND ----------

dbutils.fs.ls("/mnt/images/labels")

# COMMAND ----------

csv_path = "/mnt/images/labels/labeledDatapoints.csv"

label_df = spark.read.format("csv") \
  .option("header", True) \
  .option("inferSchema", True) \
  .load(csv_path)

display(label_df)

# COMMAND ----------

csv_path = "/mnt/images/labels/abandonedGarbagewithlabels-50.csv" 

garbage_df = spark.read.format("csv") \
  .option("header", True) \
  .option("inferSchema", True) \
  .load(csv_path)

display(garbage_df)

# COMMAND ----------

labels = ['furniture', 'mattress', 'large_appliance', 'other']

# COMMAND ----------

label_df.write.mode("overwrite").saveAsTable("labels")

# COMMAND ----------

garbage_df.write.mode("overwrite").saveAsTable("garbage_info")

# COMMAND ----------

dbutils.fs.mkdirs("/mnt/images/garbage_checkpoint")

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql import *

# COMMAND ----------

display(spark.read.format("binaryFile").load(f"/mnt/images/AbGarbageImages/"))

# COMMAND ----------

spark.read.format("binaryFile").load(f"/mnt/images/AbGarbageImages/").write.saveAsTable("garbage_images")

# COMMAND ----------

display(spark.read.table("garbage_images"))

# COMMAND ----------

display(label_df)

# COMMAND ----------

# Parsing the label to only include Furniture, Mattress, Large Appliances, Others

# COMMAND ----------

label_df.createOrReplaceTempView("garbage_labels")

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct Label from garbage_labels;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE garbage_label AS
# MAGIC (select replace(Url, 'AmlDatastore://waste_mngmt_images', 'dbfs:/mnt/images') as path, case when contains(Label, 'Furniture') then 'furniture' when contains(Label, 'Mattress') then 'mattress' when contains(Label, 'Large_Appl') then 'large_appliances' else 'others' end as label from garbage_labels)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from hive_metastore.default.garbage_label
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE garbage_training_dataset AS 
# MAGIC   (SELECT 
# MAGIC     *, CASE WHEN label = 'furniture' THEN 'furniture' ELSE 'non-furniture' END as labels
# MAGIC     FROM garbage_images INNER JOIN garbage_label USING (path)) ;
# MAGIC
# MAGIC SELECT * FROM garbage_training_dataset LIMIT 10;

# COMMAND ----------

# Crop and resize our images
from PIL import Image
import io
from pyspark.sql.functions import pandas_udf
IMAGE_RESIZE = 256

#Resize UDF function
@pandas_udf("binary")
def resize_image_udf(content_series):
  def resize_image(content):
    """resize image and serialize back as jpeg"""
    #Load the PIL image
    image = Image.open(io.BytesIO(content))
    width, height = image.size   # Get dimensions
    new_size = min(width, height)
    # Crop the center of the image
    image = image.crop(((width - new_size)/2, (height - new_size)/2, (width + new_size)/2, (height + new_size)/2))
    #Resize to the new resolution
    image = image.resize((IMAGE_RESIZE, IMAGE_RESIZE), Image.NEAREST)
    #Save back as jpeg
    output = io.BytesIO()
    image.save(output, format='JPEG')
    return output.getvalue()
  return content_series.apply(resize_image)


# add the metadata to enable the image preview
image_meta = {"spark.contentAnnotation" : '{"mimeType": "image/jpeg"}'}

df = (spark.table("garbage_training_dataset")
      .withColumn("sort", F.rand()).orderBy("sort").drop('sort') #shuffle the DF
      .withColumn("content", resize_image_udf(col("content")).alias("content", metadata=image_meta)))
#df.write.mode('overwrite').saveAsTable("garbage_training_dataset_augmented")

# COMMAND ----------

display(df)
