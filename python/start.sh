#/bin/bash

if [ 1 -gt $# ]
then
    echo "no param set"
fi

echo "$1"

if [[ "$1" == "tensorboard" ]] || [[ "$1" == "tb" ]];
then
    echo "starting tensorboard"
    tensorboard --logdir .\\tmp\\
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

if [[ "$1" == "dqn_log" ]];
then
    echo "starting dqn_sb3_as_intended.py with full logging"
    python dqn_sb3_as_intended.py hydra.verbose=true
    # The default log level vor hydra is INFO (Debug is not printed)
    # The log file can be found in the respective output folder
fi

if [[ "$1" == "dqn" ]]
then
    echo "starting dqn_sb3_as_intended.py with info logging"
    python dqn_sb3_as_intended.py
fi