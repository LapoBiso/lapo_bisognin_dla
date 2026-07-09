# Deep Learning Applications laboratories
Author: Lapo Bisognin

## Setup the environment
I used `uv` to manage the project environment. To install all the necessary
dependencies, from the terminal:
```bash
uv sync
source .venv/bin/activate
```
For the paths used inside the code to work correctly, commands must be launched from the repository root that contains all the lab folders, so that paths like `DLA-Lab2/.../...` don't cause issues.

### ++ IMPORTANT: for complete implementation and experimental analysis, read the pdf files for each laboratory, found in the folder with the same name (e.g. \DLA-Lab1\DLA_Lab_1.pdf)++


# Lab. 1: From Pixels to Semantics


This is the only laboratory that uses hydra for configuration management. **Why hydra?** Purely for educational purposes, I wanted to test different methodologies for managing configurations. The default config file structure is therefore:

```bash
defaults:
  - model: resnet50
  - opt: adam
  - loss: cross_entropy
  - dataset: GTSRB
  - train: default
  - wandb: default
  - experiments: experiment_3_2
  - _self_

hydra:
  run:
    dir: .
  output_subdir: null
```

Which refers to the following file structure:

```
configs/
├── dataset/
│   └── GTSRB.yaml
├── experiments/
│   ├── experiment_1_1.yaml
│   └── experiment_3_2.yaml
├── loss/
│   └── cross_entropy.yaml
│   └── multi_margin.yaml
├── model/
│   ├── convnext.yaml
│   ├── inception.yaml
│   ├── mobilenet_v2.yaml
│   ├── mobilenet_v3.yaml
│   ├── resnet18.yaml
│   ├── resnet50.yaml
│   └── resnet101.yaml
├── opt/
│   └── adam.yaml
│   └── adamw.yaml
│   └── rmsprop.yaml
│   └── SGD.yaml
├── train/
│   └── default.yaml
├── wandb/
│   └── default.yaml
└── config.yaml
```

### Exercise 1.1
In the file `DLA-Lab1/src/exercise_1_1.py` you can view the code that performs a preliminary analysis on the GTSRB dataset data.
```bash
python DLA-Lab1/src/exercise_1_1.py experiments=experiment_1_1 experiments.plot_cls=true experiments.plot_distribution=true

```
Where:
- `experiments.plot_cls` generates a visualization of each class element of the dataset and saves it in `DLA-Lab1/Lab1_outputs/exercise_1`
- `experiments.plot_distribution` generates information about the class distribution in the dataset through histograms and saves it in the same folder mentioned above

### Exercise 1.2
In the file `DLA-Lab1/src/exercise_1_2.py` it is possible to view the code that builds a **baseline model** that sends the computed features to an **SVM**
```bash
python DLA-Lab1/src/exercise_1_2.py
```
This will output to the terminal a **report on various metrics** for each class as well as an overall report based on the classification performed by the model on the **test split**.

It is possible to **use different backbones** among those available in `DLA-Lab1/configs/model` by running the command with an **override**
```bash
python DLA-Lab1/src/exercise_1_2.py model=resnet50
```

### Exercise 1.3
In the file `DLA-Lab1/src/exercise_1_3.py` it is possible to view the code that **trains and evaluates the fine-tuned model** on the classification task for the classes of the input dataset.
```bash
python DLA-Lab1/src/exercise_1_3.py model=resnet50 model.head.type=Linear model.freeze_lrs=0 wandb.run_name=resnet50_Linear
```
This will log to **wandb** to allow observation of accuracy and loss metrics during training, validation, and finally testing on the trained model.
Among the configurations that can be modified from the command line:
- `model=resnet50` defines the chosen backbone
- `model.head.type=Linear` defines the type of head used, you can choose between `[Linear, MLP]`
- `model.freeze_lrs=0` defines whether the backbone will have frozen layers or not during fine-tuning **WARNING: use only with resnet-type backbones**
- `wand.run_name=resnet50_Linear` defines wandb run name

**Examples of alternative runs**:
```bash
python DLA-Lab1/src/exercise_1_3.py model=resnet101 model.head.type=MLP model.freeze_lrs=0 wandb.run_name=resnet101_MLP
```
```bash
python DLA-Lab1/src/exercise_1_3.py model=resnet18 model.head.type=Linear model.freeze_lrs=1 wandb.run_name=resnet18_Linear
```

### Exercise 2
**Code abstraction** was achieved through **Hydra**; besides the models, **it is possible to instantiate different optimizers or losses** among those available in `DLA-Lab1/configs/opt` and `DLA-Lab1/configs/loss`, for example:
```bash
python DLA-Lab1/src/exercise_1_3.py model=resnet18 model.head.type=Linear model.freeze_lrs=0 opt=adamw loss=multi_margin wandb.run_name=resnet18_Linear_adamw
```
Experiment **logging** is done on **wandb**.

### Exercise 3.2
In the file `DLA-Lab1/src/exercise_3_2.py` you can view the code that performs the **image retrieval** task and then implements and runs the **Nearest Mean Classifier (NMC)**
```bash
python DLA-Lab1/src/exercise_3_2.py model=inception model.head.type=Linear experiments=experiment_3_2 experiments.K=50 experiments.cls=1
```
where:
- `experiments.cls=1` is the reference class on which to display the highest similarities
- `experiments.K=50` is the number of topK similarities displayed on the terminal

<br>

<br>

# Lab. 2: The Transformative Transformer

In this laboratory I used OmegaConf to manage configurations with a simpler methodology than the previous laboratory.

## Exercise 1
In the first exercise, in `DLA-Lab2/src/exercise_1.py`, a **preliminary data study** is performed on the `cornell-movie-review-data/rotten_tomatoes` dataset.
```bash
python DLA-Lab2/src/exercise_1.py --read_data=True
```
Then the data features are extracted with the `distilbert/distilbert-base-uncased` model to be passed to a classifier such as `SVC`, whose best configuration is chosen based on evaluation metrics.
```bash
python DLA-Lab2/src/exercise_1.py --build_classifier=True
```

## Exercise 2
In the second exercise, in `DLA-Lab2/src/exercise_2.py`, the model is fine-tuned on the sentiment analysis task
```bash
python DLA-Lab2/src/exercise_2.py --checkpoint=distilbert/distilbert-base-uncased --dataset=cornell-movie-review-data/rotten_tomatoes --epochs=10
```
Logging is done on wandb, showing the evaluation metrics defined by the `compute_metrics` function in `DLA-Lab2/src/utils.py` and passed into the `trainer`

## Exercise 3
In the third exercise, in `DLA-Lab2/src/app.py` **++WARNING: `DLA-Lab2/src/exercise_3.py` is the implementation without the gradio interface++**. A model is implemented for the **Text-to-Image retrieval** task. The chosen model is `openai/clip-vit-base-patch32`, working on the `jxie/flickr8k` dataset.
```bash
python DLA-Lab2/src/app.py --checkpoint=openai/clip-vit-base-patch32 --dataset=jxie/flickr8k 
```
<br>

<br>

# Lab. 3: Getting Up to Speed with Deep Reinforcement Learning

## Exercise 1
In the first exercise, in `DLA-Lab3/src/exercise_1.py`, the `reinforce` algorithm is run to **train a policy in the** `CartPole-v1` **environment**.
```bash
python DLA-Lab3/src/exercise_1.py --train=true --eval=false --gamma=0.99 --temp=1
```
where:
- `temp=1` is the temperature parameter for action selection
- `gamma` is the discount factor value

The trained policy model will be saved in `DLA-Lab3/checkpoints`, and from there you can load it by entering the checkpoint name on **line 26** (if you haven't changed the run name, this step is not necessary), and then **run an episode with the trained model**:
```bash
python DLA-Lab3/src/exercise_1.py --train=false --eval=true 
```

For training, you can observe the **logs on wandb**, where the evaluation metrics will also be logged.

## Exercise 2
In the second exercise, in `DLA-Lab3/src/exercise_2.py`, the `reinforce_baseline` algorithm is run to **train a policy in the** `CartPole-v1` **environment** using a **baseline**.
```bash
python DLA-Lab3/src/exercise_2.py --train=true --eval=false
```
And for evaluation:
```bash
python DLA-Lab3/src/exercise_2.py --train=false --eval=true
```
## Exercise 3
In the third exercise, in `DLA-Lab3/src/exercise_3.py`, the `deepQ_learning` algorithm is run to **train an action value function in the** `CarRacing-v3` **environment**.
```bash
python DLA-Lab3/src/exercise_3.py --train=true --eval=false --maxIt=500 --target_update=1000 --eps_decay_step=50000 --capacity=50000 --gamma=0.995
```
Where:
- `maxIt=500` is the number of training episodes
- `target_update=2000` is the number of steps needed to update the target
- `eps_decay_step=50000` is the number of steps needed to minimize the epsilon value
- `capacity=50000` is the size of the replay memory

And for evaluation:
```bash
python DLA-Lab3/src/exercise_3.py --train=false --eval=true
```
