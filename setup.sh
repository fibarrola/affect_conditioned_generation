conda create -n aff_gen -y
conda activate aff_gen
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y

pip install git+https://github.com/openai/CLIP.git

mkdir data  

# export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/"