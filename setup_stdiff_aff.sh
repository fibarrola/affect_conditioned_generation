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