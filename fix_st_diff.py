#open file in read mode
FILEPATH = "stable_diffusion/scripts/txt2img.py"

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

for line in [
    "configs/latent-diffusion/txt2img-1p4B-eval.yaml",
    "models/ldm/text2img-large/model.ckpt",
    "outputs/txt2img-samples-laion400m",
    "configs/stable-diffusion/v1-inference.yaml",
    "models/ldm/stable-diffusion-v1/model.ckpt",
    "outputs/txt2img-samples"
    ]:
    replace(FILEPATH, line, "stable_diffusion/"+line)