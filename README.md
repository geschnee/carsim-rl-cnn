
# install instructions on fresh windows machine

- clone repo
- python -m pip install -r python/requirements.txt
- add Path variables
    - Python310
    - Python310/Scripts
        - this is for tensorboard
- Install Unity version 2022.3.10f1
- start Unity and import the project
- open Scenes/PythonControlledScene.unity
- use NuGet in Unity to install 
    - Magick.NET-Q16-AnyCPU
    - Magick.NET.Core
- add Magick.Native-Q16-x64.dll to Assets/Packages
- add to Assets/Plugins
    - AustinHarrisJsonRpc.dll
    - PeacefulPie.dll
- add NetManager.cs Script from Assets/Packages/PeacefulPie to RpcCommunicator Object in the scene

## AustinHarrisJsonRpc.dll

- download https://www.nuget.org/api/v2/package/AustinHarris.JsonRpc/1.2.3
- rename to ".zip"
- unzip file
- copy lib/netstandard2.1/AustinHarris.JsonRpc.dll into Assets/Plugins
- in Unity Editor select the file in the Plugins directory
    - in the inspector select 'validate references' and apply

## PeacefulPie.dll

Download at https://github.com/hughperkins/peaceful-pie/releases/tag/v2.1.0

# running the code

- model training and evaluation are done using the python script `sb3_ppo.py`
- model training and evaluation configurations are stored in the `cfg` directory

## training

- start unity scene
- start python script
    - `python sb3_ppo.py --config-name <config-filename>`
    - `python sb3_ppo.py --config-name cfg/hardDistanceMixedLight.yaml`


## evaluation

- copy model to `models` directory
- start unity scene
- start python script
    - `python sb3_ppo.py --config-name <config-filename>`
    - `python sb3_ppo.py --config-name cfg/hardDistanceMixedLight_eval.yaml`

## episode replays only

see replay_on_jetbot.md for instructions

# trained model

trained models availible on huggingface:
https://huggingface.co/geschnee/carsim-rl-cnn

# Master's Thesis document

https://github.com/geschnee/masterarbeit

# Acknowledgements

## This repo and thesis builds upon previous work at the Scads.AI
https://github.com/jonaskonig/BachelorNN
https://github.com/Maddi97/master_thesis