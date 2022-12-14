pip install git+https://github.com/openai/CLIP.git
git clone https://github.com/CompVis/taming-transformers
pip install -r requirements.txt

git clone https://github.com/CompVis/stable-diffusion.git
mv stable-diffusion stable_diffusion
touch stable_diffusion/ldm/__init__.py
touch stable_diffusion/ldm/modules/__init__.py
python3 fix_st_diff.py
python3 stable_diffusion/setup.py install