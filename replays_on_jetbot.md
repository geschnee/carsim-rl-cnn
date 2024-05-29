# connect to jetbot

check space on device
df -h

check size of files 
ls -lh

check size of directories in current folder
du -sh *

## there is not a lot of space on the JetBot

jetbot@nano-4gb-jp45:/$ df -h
Filesystem      Size  Used Avail Use% Mounted on
/dev/mmcblk0p1   24G   23G   35M 100% /
none            1.8G     0  1.8G   0% /dev
tmpfs           2.0G  4.0K  2.0G   1% /dev/shm
tmpfs           2.0G   27M  2.0G   2% /run
tmpfs           5.0M  4.0K  5.0M   1% /run/lock
tmpfs           2.0G     0  2.0G   0% /sys/fs/cgroup
tmpfs           397M     0  397M   0% /run/user/1000

# mount usb drive to jetBot

## check usb drive is connected

lsblk

## mount usb drive

sudo mount /dev/sda1 /mnt/usb_drive

## make mount permanent
mount will be removed in each reboot unless we do:

https://www.youtube.com/watch?v=t1MCbL95Xcg

get usb drive uuid:

lsblk -f

add new line:

sudo vi /etc/fstab

UUID=usb_uuid /mnt/usb_drive ext4 nofail,x-systemd.automount,x-systemd.idle-timeout=60,x-systemd.device-timeout=2

reload config:

sudo systemctl daemon-reload && sudo systemctl restart local-fs.target

# copy carsim repo to JetBot

scp -r carsim_no_mlagents jetbot@192.168.1.3:/mnt/usb_drive/carsim_no_mlagents

# copy replay to JetBot

scp -r episode_recordings jetbot@192.168.1.3:/mnt/usb_drive/carsim_no_mlagents/python/episode_recordings


# create virtual env for python

https://docs.python.org/3/library/venv.html

cd /mnt/usb_drive
python -m venv sb3_env

## activate virtual env

cd /mnt/usb_drive
source sb3_env/bin/activate

## install packages

python3 -m pip install -r minimal_requirements.txt


# run replay

python3 sb3_ppo_replay_only.py > replay_output.txt
