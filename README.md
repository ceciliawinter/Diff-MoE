# Diff-MoE: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching

## Setup

```bash

# clone the repo
git clone --recursive https://github.com/ceciliawinter/Diff-MoE.git

# Starting from the official container
docker run -ti --gpus all --shm-size 5g --name diff-moe -v ${DATA_PATH}:/data nvcr.io/nvidia/pytorch:22.09-py3 bash



# build on H200
mkdir -p FasterTransformer/build
cd FasterTransformer/build
cmake -DSM=90 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
make -j
```
* Note: Replace `${DATA_PATH}` with path on host.
* Note: The `xx` of `-DSM=xx` in the scripts above means the compute capability of your GPU. The following table shows the compute capability of common GPUs.

|  GPU  | compute capacity |
| :---: | :--------------: |
|  P40  |        60        |
|  P4   |        61        |
| V100  |        70        |
|  T4   |        75        |
| A100  |        80        |
|  A30  |        80        |
|  A10  |        86        |
|  H200 |        90        |

```
# Python dependencies
pip install -r ../requirement.txt
```


## Prepare models

You may fine-tune the model on downstream tasks to generate adapter files. Name each adapter file as `${model}-${dataset}` and place them in the `/data` directory.

```bash
mkdir /data/ft
cd /workspace/FasterTransformer/

# for base models (e.g., Switch-Base)
./scripts/convert.sh
# for finetuned models
./scripts/convert_finetuned.sh
```



## Evaluation

logs will be output to the `/logs/${model}` directory.

```bash
cd /workspace/FasterTransformer/
python scripts/eval_cache.py
```
