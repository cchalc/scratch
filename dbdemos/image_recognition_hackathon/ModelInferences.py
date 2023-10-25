# Databricks notebook source
#Batch/ streaming model scoring
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
import os
requirements_path = ModelsArtifactRepository("models:/FurnitureGarbage/Production").download_artifacts(artifact_path="requirements.txt") # download model requirement from remote registry

if not os.path.exists(requirements_path):
  dbutils.fs.put("file:" + requirements_path, "", True)

# COMMAND ----------

# MAGIC %pip install -r $requirements_path

# COMMAND ----------

import io
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

#Call the pipeline and returns the main class with its probability
def predict_byte_series(content_as_byte_series, pipeline):
  #Transform as a list of PIL Images for our huggingface pipeline:
  image_list = content_as_byte_series.apply(lambda b: Image.open(io.BytesIO(b))).to_list()
  #the pipeline returns the probability for all the class
  predictions = pipeline.predict(image_list)
  #Filter & returns only the class with the highest score [{'score': 0.999038815498352, 'label': 'normal'}, ...]
  return pd.DataFrame([max(r, key=lambda x: x['score']) for r in predictions])  


df = spark.read.table("garbage_training_dataset").limit(50)
#Switch our model in inference mode
pipeline.model.eval()
with torch.set_grad_enabled(False):
  predictions = predict_byte_series(df.limit(10).toPandas()['content'], pipeline)
display(predictions)
