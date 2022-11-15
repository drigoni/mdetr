#!/bin/bash
# inputs
MODE=$1

if [[ $MODE == "install" ]]; then
  # conda create -n mdetr pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
  # conda create -n pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 torchtext -c pytorch
  # those work at some extent
  # conda create -n mdetr pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.6 torchtext -c pytorch -c conda-forge
  conda create -n mdetr pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.1 torchtext=0.11.0 -c pytorch -c conda-forge
  
  conda activate mdetr
  python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html


  pip install numpy cython scipy xmltodict tqdm transformers==4.5.1
  # pip install setuptools==59.5.0  # https://stackoverflow.com/questions/70520120/attributeerror-module-setuptools-distutils-has-no-attribute-version
  pip install onnx onnxruntime prettytable submitit wandb timm
  pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI&egg=pycocotools # uninstall pycocotools 2.0.5 for pycocotools-2.0 ???
  pip install git+https://github.com/cocodataset/panopticapi.git#egg=panopticapi


elif [[ $MODE == "uninstall" ]]; then
  conda deactivate
  conda env remove -n mdetr
else
  echo "To be implemented."
fi
