# **Procedure**

Follow General Steps from : https://cloud.google.com/tpu/docs/tutorials/resnet-2.x with the following changes :

1. Use the following local variables on the TPU VM

`export TPU_NAME=local`

`export STORAGE_BUCKET=gs://longtail_imagenet`

`export MODEL_DIR=${STORAGE_BUCKET}/resnet-2x`

`export DATA_DIR=${STORAGE_BUCKET}/IX_IMAGENET_DIR`

2. Replace the respective files in the folder **/usr/share/tpu/models/official/vision/image_classification/resnet** with the ones in this folder.
