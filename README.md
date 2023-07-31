# BC Policy Learning + sim-to-real transfer

## Installation

### Prerequisites: Install MUSE Simulator
Follow instructions in [MUSE](https://github.com/deepmind/mujoco) repository to install the simulator.

### Install Robolearn

Activate your conda environment. Then, use pip to install robolearn requirements:

```
pip install -r requirements.txt
```

Then, install robolearn:

```
pip install -e .
```


## Data Collection

To collect data for the ***proxy localization*** task run the following command:

```
python -m robolearn.collect.poses --output-dir $DATASET/robolearn/dr_sim_pose/train/ --poses 20000 --dr
```

Note that `--dr` flag controls if domain randomization is applied to the data.

To collect demonstrations for a ***manipulation task*** e.g., Stack, run the following command:

```
python -m robolearn.collect.demos --output-dir $DATASET/robolearn/dr_stack/train/ --episodes 2000 --seed 0 --env-name DR-Stack-v0
```


## Training

To train a policy using Behavioural Cloning run:

```
python -m robolearn.train --task bc --log-dir $OUTPUT_DIR --train-path $DATASET/robolearn/dr_stack/train/ --eval-env DR-Stack-v0 --data-aug iros23_s2r
```

## Evaluation
To evaluate a trained policy run:

```
python -m robolearn.evaluate.policy --checkpoint $OUTPUT_DIR/checkpoint.pth --env-name Stack-v0 --episodes 250 --checkpoint $OUTPUT_DIR  --record
```
Note that `--record` flag will record videos of the policy run for evaluation episodes.

## Real Robot Datasets

You can download real robot datasets aligned with simulation datasets for the proxy task and stacking task using the following links:

[Stacking dataset](https://drive.google.com/file/d/1l0p45EC3ZlFWctf0EvOpmD70neglqQoR/view?usp=sharing)

[Localization training dataset - default](https://drive.google.com/file/d/1ZhG2Dy8NQcasQ5SpCgtis4M4F3j28xwa/view?usp=sharing)

[Localization validation dataset - default](https://drive.google.com/file/d/1wfSFlpKqeKdsVPmsilvGQegFmeYnJLXj/view?usp=sharing)

[Localization validation dataset - textured table cloth](https://drive.google.com/file/d/1ZtXmJUkOd1qIPU10Cdox5WfsecfN5zH9/view?usp=sharing)

[Localization validation dataset - low lighting](https://drive.google.com/file/d/1kamPnEsRuPdxICTqaON3Vy51cN3mSzfL/view?usp=sharing)

[Localization validation dataset - multicolor lighting](https://drive.google.com/file/d/1D6Og4Qt7KsSAaywztCVO7GX-maeiBEaH/view?usp=sharing)

[Localization validation dataset - object colors variation](https://drive.google.com/file/d/1aGHbjqicZ3Yxh0FtfyGIwYGFKZV7cLNC/view?usp=sharing)

[Localization validation dataset - camera variation](https://drive.google.com/file/d/1XQARMlfynxML6fGQFoLb2zcnGEufdLlB/view?usp=sharing)

## Pretrained models

Coming soon...

## Citations

Please, if you use this repository in you research project, think about properly citing our work:

```
 @article{garcia2023,
    author    = {Ricardo Garcia and Robin Strudel and Shizhe Chen and Etienne Arlaud and Ivan Laptev and Cordelia Schmid},
    title     = {Robust visual sim-to-real transfer for robotic manipulation},
    journal   = {International Conference on Intelligent Robots and Systems (IROS)},
    year      = {2023}
}    
```
