
# install instructions on fresh windows machine

- clone repo
- python -m pip install -r python/requirements.txt
- add Path variables
    - Python310
    - Python310/Scripts
        - this is for tensorboard
- Install Unity version 2022.3.10f1
- use NuGet in Unity to install 
    - Magick.NET-Q16-AnyCPU
    - Magick.NET.Core
- add Magick.Native-Q16-x64.dll to Assets/Packages
- add to Assets/Plugins
    - AustinHarrisJsonRpc.dll
    - PeacefulPie.dll

# running the code

## training

## evaluation

## episode replays only

see replay_on_jetbot.md for instructions

# trained model

todo upload to huggingface
https://huggingface.co/geschnee/carsim-rl-cnn




# Acknowledgements

## This repo and thesis builds upon previous work at the Scads.AI
https://github.com/jonaskonig/BachelorNN
https://github.com/Maddi97/master_thesis