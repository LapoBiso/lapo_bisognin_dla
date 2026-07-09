# Deep Learning Applications laboratories
Author: Lapo Bisognin

## Setup the environment
Ho utilizzato `uv` per gestire l'environment del progetto. Per installare
tutte le dipendenze necessarie, da terminale:
```bash
uv sync
source .venv/bin/activate
```
Ai fini dei path inseriti all'interno del codice è necessario lanciare comandi dalla repository che contiene tutte le cartelle relative ai laboratori così che i path di tipo `DLA-Lab2/.../...` non forniscano problemi.

### ++ IMPORTANTE: per analisi implementative e sperimentali complete leggere i file pdf relativi a ciascun laboratorio che si trovano nella cartella omonima ```(es. \DLA-Lab1\DLA-Lab1.pdf)```++


# Lab. 1: From Pixels to Semantics


Questo è l'unico laboratorio che utilizza hydra ai fini della configurazione. **Perchè hydra?** Solo per fini didattici ho pensato di testare più metodologie per gestire le configurazioni. La struttura dei file config default è dunque:

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

Che fa riferimento alla struttura dei file:

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

### Esercizio 1.1
Nel file `DLA-Lab1/src/exercise_1_1.py` è possibile visionare il codice che si occupa di eseguire un'analisi preliminare sui dati del dataset GTSRB.
```bash
python DLA-Lab1/src/exercise_1_1.py experiments=experiment_1_1 experiments.plot_cls=true experiments.plot_distribution=true

```
Con:
- `experiments.plot_cls` genera una visualizzazione di ciascun elemento di classe del dataset e la salva in `DLA-Lab1/Lab1_outputs/exercise_1`
- `experiments.plot_distribution`  genera informazioni sulla distribuzione delle classi nel dataset attraverso degli istogrammi e la salva nella stessa cartella indicata sopra

### Esercizio 1.2
Nel file `DLA-Lab1/src/exercise_1_2.py` è possibile visionare il codice che si occupa di costruire un **modello di baseline** che invia le feature computate a una **SVM**
```bash
python DLA-Lab1/src/exercise_1_2.py
```
Questo fornirà in output su terminale un **report su varie metriche** per ciascuna classe e anche complessivo in base alla classificazione eseguita dal modello sullo **split di testing**.

È possibile **utilizzare backbone diversi** fra quelli disponibili in `DLA-Lab1/configs/model` eseguendo il comando con **override**
```bash
python DLA-Lab1/src/exercise_1_2.py model=resnet50
```

### Esercizio 1.3
Nel file `DLA-Lab1/src/exercise_1_3.py` è possibile visionare il codice che si occupa di **addestrare e valutare il modello su cui si fa fine-tuning** nel task di classificazione delle classi del dataset in input.
```bash
python DLA-Lab1/src/exercise_1_3.py model=resnet50 model.head.type=Linear model.freeze_lrs=0
```
Questo farà log su **wandb** per permettere di osservare le misure di accuratezza e loss in fase di training, validation e infine di testing sul modello addestrato.
Fra le configurazioni modificabili in riga di codice:
- `model=resnet50` definisce il backbone scelto
- `model.head.type=Linear` definisce il tipo di testa individuato, è possibile scegliere fra `[Linear, MLP]`
- `model.freeze_lrs=0` definisce se il backbone avrà dei layer congelati oppure no in fase di fine-tuning **ATTENZIONE: da usare solo su backbone di tipo resnet**

**Esempi di esecuzioni alternative**:
```bash
python DLA-Lab1/src/exercise_1_3.py model=inception model.head.type=MLP model.freeze_lrs=0
```
```bash
python DLA-Lab1/src/exercise_1_3.py model=resnet18 model.head.type=Linear model.freeze_lrs=1
```

### Esercizio 2
**L'astrazione del codice** è stata eseguita tramite **Hydra**, oltre ai modelli **è possibile istanziare ottimizzatori o loss differenti fra quelli disponibili** in `DLA-Lab1/configs/opt` e `DLA-Lab1/configs/loss`, ad esempio:
```bash
python DLA-Lab1/src/exercise_1_3.py model=resnet18 model.head.type=Linear model.freeze_lrs=0 opt=adamw loss=multi_margin
```
Il **log** degli esperimenti è eseguito su **wandb**.

### Esercizio 3.2
Nel file `DLA-Lab1/src/exercise_3_2.py` è possibile visionare il codice che si occupa di eseguire il task di **image retrieval** e poi di implementare e eseguire il **Nearest Mean Classifier (NMC)**
```bash
python DLA-Lab1/src/exercise_3_2.py model=resnet18 model.head.type=Linear experiments=experiment_3_2 experiments.K=50 experiments.cls=1
```
con:
- `experiments.cls=1` la classe di riferimento su cui visualizzare le più alte similarità
- `experiments.K=50` il numero delle topK similarità visualizzate su terminale

<br>

<br>

# Lab. 2: The Transformative Transformer

In questo laboratorio ho utilizzato OmegaConf al fine di gestire le configurazioni in una metodologia più semplificata rispetto al laboratorio precedente.

## Esercizio 1
Nel primo esercizio in `DLA-Lab2/src/exercise_1.py` si esegue uno **studio preliminare dei dati** del dataset `cornell-movie-review-data/rotten_tomatoes`.
```bash
python DLA-Lab2/src/exercise_1.py --read_data=True
```
E in seguito si estraggono le feature dei dati con il modello `distilbert/distilbert-base-uncased`per passarle a un classificatore come `SVC` di cui si sceglie la migliore configurazione tramite le metriche di evaluation.
```bash
python DLA-Lab2/src/exercise_1.py --build_classifier=True
```

## Esercizio 2
Nel secondo esercizio in `DLA-Lab2/src/exercise_2.py` si esegue il fine-tuning del modello sul task di sentiment analysis 
```bash
python DLA-Lab2/src/exercise_2.py --checkpoint=distilbert/distilbert-base-uncased --dataset=cornell-movie-review-data/rotten_tomatoes --epochs=10
```
Viene fatto log su wandb che mostra le metriche di evaluation definite dalla funzione `compute_metrics` in `DLA-Lab2/src/utils.py` e inserite all'interno del `trainer`

## Esercizio 3
Nel terzo esercizio in `DLA-Lab2/src/app.py` **++ATTENZIONE `DLA-Lab2/src/exercise_3.py` è l'implementazione senza interfaccia gradio++**. Si va a implementare un modello atto al task di **Text-to-Image retrieval**. Il modello scelto è `openai/clip-vit-base-patch32` che lavora sul dataset `jxie/flickr8k`. 
```bash
python DLA-Lab2/src/app.py --checkpoint=openai/clip-vit-base-patch32 --dataset=jxie/flickr8k 
```
<br>

<br>

# Lab. 3: Getting Up to Speed with Deep Reinforcement Learning

## Esercizio 1
Nel primo esercizio in `DLA-Lab3/src/exercise_1.py` viene eseguito l'algoritmo `reinforce` per **addestrare una policy nell'environment** `CartPole-v1`. 
```bash
python DLA-Lab3/src/exercise_1.py --train=true --eval=false --gamma=0.99 --temp=1
```
con:
- `temp=1` il parametro di temperatura nella scelta dell'azione
- `gamma` il valore del fattore di discount

Il modello della policy addestrato sarà salvato in `DLA-Lab3/checkpoints` e da qui è possibile caricarlo inserendo il nome del checkpoint alla **riga 26** (se non si è cambiato il nome della run non è necessario svolgere questo passaggio) e dunque **eseguire un episodio con il modello addestrato**:
```bash
python DLA-Lab3/src/exercise_1.py --train=false --eval=true 
```

Per il train è possibile osservare i **log su wandb** su cui saranno loggate anche le metriche di evaluation.

## Esercizio 2
Nel secondo esercizio in `DLA-Lab3/src/exercise_2.py` viene eseguito l'algoritmo `reinforce_baseline` per **addestrare una policy nell'environment** `CartPole-v1` con l'utilizzo della **baseline**.
```bash
python DLA-Lab3/src/exercise_2.py --train=true --eval=false
```
E per l'evaluation:
```bash
python DLA-Lab3/src/exercise_2.py --train=false --eval=true
```
## Esercizio 3
Nel terzo esercizio in `DLA-Lab3/src/exercise_3.py` viene eseguito l'algoritmo `deepQ_learning` per **addestrare una action value function nell'environment** `CarRacing-v3`.
```bash
python DLA-Lab3/src/exercise_3.py --train=true --eval=false --maxIt=500 --target_update=1000 --eps_decay_step=50000 --capacity=50000 --gamma=0.995
```
Con:
- `maxIt=500` il numero di episodi di training
- `target_update=2000` il numero di step necessari per l'aggiornamento del target
- `eps_decay_step=50000` il numero di step necessari alla minimizzazione del valore di epsilon
- `capacity=50000` la dimensionalità della replay memory

E per l'evaluation:
```bash
python DLA-Lab3/src/exercise_3.py --train=false --eval=true
```