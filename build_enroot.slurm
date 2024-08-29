#!/bin/bash

#SBATCH --job-name=build_enroot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

set -euo pipefail

SHARED_FS_DIR=/shared/aisingapore
SQSH_PARENT_DIR=${SHARED_FS_DIR}/sqsh
# you comment this out to let enroot rename it
# SQSH_DIR=${SQSH_PARENT_DIR}/container_name.sqsh


# note for nvidia images, change to nvcr.io#nvidia/pytorch:xx.xx-py3

REMOTE_IMAGE="nvcr.io#nvidia/pytorch:24.08-py3"

[[ -d ${SQSH_PARENT_DIR} ]] || mkdir -p

# if sqsh_dir is empty, cd into parent dir and pull the image

if [[ -z "$(ls -A ${SQSH_DIR})" ]]; then
    cd ${SQSH_PARENT_DIR}
    enroot import docker://${REMOTE_IMAGE}
fi

# help user create a folder in shared to bind mount

enroot create 