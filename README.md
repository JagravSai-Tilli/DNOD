
# DNOD : Deformable Neural Operators for Object Detection in SAR Images
### [OpenReview](https://openreview.net/forum?id=tjBqPJdQ72)


## Installation

<details>
  <summary>Installation</summary>

   ```sh
   pip install -r requirements.txt
   ```

</details>



## Data

<details>
  <summary>Data</summary>

Please download [SAR_DET100K](https://github.com/zcablii/SARDet_100K) dataset and organize them as following:
```

SAR_DET100K/
  ├── Annotations/
  	├── train.json
    ├── val.json
  	└── test.json

  └── images/
  	├── train
    ├── val
  	└── test
```
Change the data_path variable in the config file (config/cfg_dnod.py)

</details>


## Run
<details>
  <summary>Eval our pretrianed models</summary>

  ```sh
  python3 main.py --eval True
  ```
</details>
<details>
  <summary> Train our model from scratch </summary>

```sh
python3 main.py
```
</details>


## Distributed Run
<details>
  <summary>Eval our pretrianed models</summary>
  
  ```sh
  torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py --eval True
  ```
</details>
<details>
  <summary> Train our model from scratch </summary>

```sh
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py
```
</details>


# Acknowledgement



<details>
  <summary>Code Acknowledgement</summary>

  Many parts of code are inspired and modified from following repositories
  ```sh
   https://github.com/IDEA-Research/DINO.git
   https://github.com/NVlabs/AFNO-transformer.git
   https://github.com/Atten4Vis/MS-DETR.git
   ```

</details>

### This code is for the paper DNOD (TMLR 2025)
