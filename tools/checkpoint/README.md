# Convert checkpoints


## From megatron to HF

1. Activate a conda environment with torch
```bash
conda activate <env_name>
```
2. Export PYTHONPATH to the root of the repo
```bash
export PYTHONPATH=/path/to/Megatron-LM
```
3. Run the conversion script
```bash
python convert_llama_back_to_hf.py --reference-hf-model REFERENCE_MODEL --save-path SAVE_DIR PATH_TO_MEGATRON_CHECKPOINT
```
`REFERENCE_MODEL` is the name of the model in the HF model hub that you want to use as a reference for the conversion.  
This can take a huggingface model format like `org/model` or a direct path to a transformers model

`SAVE_DIR` is the directory where the converted model will be saved  

`PATH_TO_MEGATRON_CHECKPOINT` is the path to the megatron checkpoint you want to convert,  
this should have a directory like iterxxxx

4. Validate the conversion

```bash
# Assuming same path as above 
python validate_conversion.py SAVE_DIR REFERENCE_MODEL
```

## From HF to megatron

1. Start enroot container
```bash
enroot create -n nvidia path_to_megatron.sqsh
enroot start --rw -m SHARED_DIRECTORY nvidia
```
SHARED_DIRECTORY is the shared mount, usually /fsx/ or /shared/

2. Run convert script
```
bash 
