# Affect Conditioned Generation

## Evaluate affect

```
python3 get_affect.py
--prompt <str>          The text prompt if we want a prompt's affect
--img_path <str>        Path to the image if we want an image's affect
--format <str>          Affect display format ('score', 'uniform' or 'latex')
```
<br>

## Train MLP

We do not provide the image
```
python3 train_mlp.py
--num_epochs <int>      number of training epochs
--scaling <str>         scaling for input and output data. Can be 'uniform', 'whiten', 'normalize' or 'none'
--lr <float>            learning rate
--layer_dims <str>      layer dimensions. Separate with |
--use_dropout <bool>    Use dropout for training?
--use_sigmoid <bool>    Use sigmoid at the end of last layer?
```
<br>

## Affect-conditioned VQGAN+CLIP generation

### Install

```
source setup_vqganclip.sh
```
download model.yaml and model.ckpt for VQGAN. For example, from
https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/?p=%2Fconfigs&mode=list
and paste in vqgan_model folder
```
cd vqgan_model
curl -L -o vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' 
curl -L -o vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1'
cd ..
```
<br>

### Run
```
python3 gen_vqganclip.py
--prompt <str>          Prompt of which to get affect score
--V <float>             Valence score (bad<good) in [0,1], default=None
--A <float>             Arousal score (calm<exciting) in [0,1], default=None,
--D <float>             Dominance score (from http://crr.ugent.be/archives/1003controlled<in control) in [0,1], default=None,
--max_iter <int>        Number of algorithm iterations
```
<br>

## Affect-conditioned CLIPDraw generation

### Install
```
source setup_clipdraw_aff.sh
```

### Run
```
python3 gen_clipdraw.py
--prompt <str>          Prompt of which to get affect score
--V <float>             Valence score (bad<good) in [0,1], default=None
--A <float>             Arousal score (calm<exciting) in [0,1], default=None,
--D <float>             Dominance score (controlled<in control) in [0,1], default=None,
--max_iter <int>        Number of algorithm iterations
--num_paths <int>       Number of individual traces to draw
--save_path <str>       Subfolder for saving resutls
```

<br>

## Affect-conditioned Stable Diffusion

### Install
```
source setup_stdiff_aff.sh
```
Download a stable diffusion pretrained checkpoint from https://huggingface.co/CompVis/stable-diffusion
and paste into stable_diffusion/models/ldm/stable-diffusion-v1/ as model.ckpt


### Train BERT

First, download the affect scores from http://crr.ugent.be/archives/1003
```
curl -L -o data/Ratings_Warriner_et_al.csv http://crr.ugent.be/papers/Ratings_Warriner_et_al.csv
```

Then train the models (this can take a while)
```
python3 train_with_bert.py
```

### Run
```
python3 gen_diffusion.py
--prompt <str>          Prompt of which to get affect score
--V <float>             Valence score (bad<good) in [0,1], default=None
--A <float>             Arousal score (calm<exciting) in [0,1], default=None,
--D <float>             Dominance score (controlled<in control) in [0,1], default=None,
--reg <float>           Regularization parameter
--max_iter <int>        Z search iterations
--save_path <str>       Subfolder for saving resutls
```