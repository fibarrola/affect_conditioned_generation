conda create -n aff_gen -y
conda activate aff_gen
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y

pip install git+https://github.com/openai/CLIP.git

git clone https://github.com/CompVis/taming-transformers
touch taming-transformers/taming/__init__.py
touch taming-transformers/taming/modules/__init__.py
touch taming-transformers/taming/modules/vqvae/__init__.py
cd taming-transformers/
python3 setup.py install
cd ..

git clone https://github.com/CompVis/stable-diffusion.git
mv stable-diffusion stable_diffusion
touch stable_diffusion/ldm/__init__.py
touch stable_diffusion/ldm/modules/__init__.py
touch stable_diffusion/ldm/models/__init__.py
cp aux/txt2img_aff_many.py stable_diffusion/scripts/
mkdir stable_diffusion/models/ldm/stable-diffusion-v1/
python3 fix_st_diff.py
cd stable_diffusion
python3 setup.py install
cd ..