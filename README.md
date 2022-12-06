# Affect Conditioned Generation


### VQGAN model download
curl -L -o vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384

<br>

## Usage

### Evaluate affect

```
python3 get_affect.py
--prompt <str>          The text prompt if we want a prompt's affect
--img_path <str>        Path to the image if we want an image's affect
--format <str>          Affect display format ('score', 'uniform' or 'latex')
```
<br>

### Train MLP

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

### Generate image with specific affect scores using VQGAN+CLIP

```
python3 gen_vqganclip.py
--prompt <str>          Prompt of which to get affect score
--E <float>             Evaluation score (bad<good) in [0,1], default=None
--P <float>             Potency score (weak<strong) in [0,1], default=None,
--A <float>             Activity (calm<exciting) in [0,1], default=None,
--max_iter <int>        Number of algorithm iterations
```
<br>

### Generate image with specific affect scores using CLIPDraw
```
python3 gen_clipdraw.py
--prompt <str>          Prompt of which to get affect score
--E <float>             Evaluation score (bad<good) in [0,1], default=None
--P <float>             Potency score (weak<strong) in [0,1], default=None,
--A <float>             Activity (calm<exciting) in [0,1], default=None,
--max_iter <int>        Number of algorithm iterations
--num_paths <int>       Number of individual traces to draw
--save_path <str>       Subfolder for saving resutls
```