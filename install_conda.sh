#!/bin/bash

# Miniconda 설치 스크립트
MINICONDA_INSTALLER_SCRIPT="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_PREFIX="${HOME}/miniconda3"

# Miniconda 다운로드
wget https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER_SCRIPT} -O /tmp/${MINICONDA_INSTALLER_SCRIPT}

# Miniconda 설치
bash /tmp/${MINICONDA_INSTALLER_SCRIPT} -b -p ${MINICONDA_PREFIX}

# Miniconda 초기화 및 PATH 설정
eval "$(${MINICONDA_PREFIX}/bin/conda shell.bash hook)"
conda init
conda config --set auto_activate_base false

# 임시 파일 삭제
rm /tmp/${MINICONDA_INSTALLER_SCRIPT}

echo "Miniconda 설치가 완료되었습니다. 터미널을 재시작하거나 'source ~/.bashrc'를 실행하여 환경을 적용하세요."

conda create --name level_1_1 python=3.8 

CONDA_BASE=$(conda info --base)
. $CONDA_BASE/etc/profile.d/conda.sh
conda activate level_1


read -p "What is your domain? 1. CV, 2. NLP, 3.Recsys / : " domain_name 
if [ ${domain_name} -eq 1 ];then #cv
    wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000263/data/data.tar.gz
    wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000263/data/code.tar.gz
fi

if [ ${domain_name} -eq 2 ];then #nlp
    wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000264/data/data.tar.gz
    wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000264/data/code.tar.gz
fi  
    
if [ ${domain_name} -eq 3 ];then #recsys
    wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000265/data/data.tar.gz
    wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000265/data/code.tar.gz    
fi

tar -zxvf data.tar.gz
tar -zxvf code.tar.gz

if [ ${domain_name} -eq 3 ];then #recsys
    mv data /code  
fi


rm -r ._*

cd code

pip install -r requirement.txt