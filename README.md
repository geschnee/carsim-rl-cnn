# CarController
This repo houses the project for Unity for the self-driving vehicle trained by evolutionary network training.

Parent repo: https://github.com/jonaskonig/BachelorNN


# Json RPC

https://de.wikipedia.org/wiki/JSON-RPC

# THis repo takes the code from Maddi97

https://github.com/Maddi97/master_thesis

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


# Slack

https://scads-workspace.slack.com/team

https://app.slack.com/client/TCZPCS29E/D06BRHAHT40


# dedicated server

./carsim_no_mlagents.exe -batchmode -nographics

it does not yet work, it looks like it does not properly render the scene

possibly due to some kind of configs for the server

https://docs.unity3d.com/Manual/dedicated-server-optimizations.html

seems like some fundamental problem (no rendering --> camera has no good image)

# Windows Standalone

works well (windows_build directory)