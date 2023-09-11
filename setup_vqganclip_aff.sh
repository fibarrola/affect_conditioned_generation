git clone https://github.com/CompVis/taming-transformers
touch taming-transformers/taming/__init__.py
touch taming-transformers/taming/modules/__init__.py
touch taming-transformers/taming/modules/vqvae/__init__.py
cd taming-transformers/
python3 setup.py install
cd ..

mkdir vqgan_model

pip install imageio
pip install stegano
pip install imgtag