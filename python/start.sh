#/bin/bash

if [ 1 -gt $# ]
then
    echo "no param set"
fi

echo "$1"

if [[ "$1" == "tensorboard" ]] || [[ "$1" == "tb" ]];
then
    echo "starting tensorboard"
    tensorboard --logdir .\\outputs\\
fi

if [[ "$1" == "ppo_log" ]];
then
    echo "starting ppo_sb3.py with full logging"
    python ppo_sb3.py hydra.verbose=true
    # The default log level vor hydra is INFO (Debug is not printed)
    # The log file can be found in the respective output folder
fi

if [[ "$1" == "ppo" ]]
then
    echo "starting ppo_sb3.py with info logging"
    python ppo_sb3.py
fi