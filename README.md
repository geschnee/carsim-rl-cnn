
# install instructions on fresh windows machine

- clone repo
- python -m pip install -r python/requirements.txt
  - I used 3.10.6, 3.10.11 and 3.11.5
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
- add to Assets/Packages
    - Magick.Native-Q16-x64.dll 
- add to Assets/Plugins
    - AustinHarrisJsonRpc.dll
    - PeacefulPie.dll
- add NetManager.cs Script from Assets/Plugins/PeacefulPie to RpcCommunicator Object in the scene


Magick is only required for the recording of videos

## Magick.Native-Q16-x64.dll

- download from internet
  - e.g. https://www.dllme.com/dll/files/magick_native-q16-x64/e924369b24e1de791993f37f0aad1e3c
- add to Assets/Packages

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

see replays_on_jetbot.md for instructions

## drive the agent yourself

- start unity scene
- python play_game_from_python.py
  - use arrow keys to change left and right acceleration values
  - use space bar to reset the acceleration values to 0

# trained models + video files + episode recordings

availible on huggingface:
https://huggingface.co/geschnee/carsim-rl-cnn

# Master's Thesis document

https://github.com/geschnee/carsim-rl-cnn/blob/main/Masterarbeit_Georg_Schneeberger_signed.pdf

# Acknowledgements

## This repo and thesis builds upon previous work at the Scads.AI
https://github.com/jonaskonig/BachelorNN
https://github.com/Maddi97/master_thesis