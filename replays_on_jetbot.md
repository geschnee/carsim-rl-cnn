# connect to jetbot

on jetbot exceute:
```
nmcli
```
connect to jetbot from the same network
```
ssh jetbot@ip
```

ip = ipv4 aus nmcli


## there is not enough space on the JetBot

### check space on device
```
df -h

Filesystem      Size  Used Avail Use% Mounted on
/dev/mmcblk0p1   24G   23G   35M 100% /
none            1.8G     0  1.8G   0% /dev
tmpfs           2.0G  4.0K  2.0G   1% /dev/shm
tmpfs           2.0G   27M  2.0G   2% /run
tmpfs           5.0M  4.0K  5.0M   1% /run/lock
tmpfs           2.0G     0  2.0G   0% /sys/fs/cgroup
tmpfs           397M     0  397M   0% /run/user/1000
```

### check size of files 
```
ls -lh
```

### check size of files in current folder
```
du -sh *
```

### file sizes sorted ascending order:
```
du -hs * | sort -h
```


# mount usb drive to jetBot

a usb drive is used to run the code and install libraries

## check usb drive is connected
```
lsblk
```
## create filesystem on the usb drive
```
sudo mkfs -t ext4 /dev/sda1
```
## create mount directory
```
mkdir /mnt/usb_drive
```

## mount usb drive
```
sudo mount /dev/sda1 /mnt/usb_drive
```

## make mount permanent
mount will be removed in each reboot unless we do:

https://www.youtube.com/watch?v=t1MCbL95Xcg

### get usb drive uuid

```
lsblk -f
```

### change /etc/fstab
```
sudo vi /etc/fstab
```

add new line:
```
UUID=usb_uuid /mnt/usb_drive ext4 nofail,x-systemd.automount,x-systemd.idle-timeout=60,x-systemd.device-timeout=2
```



### reload config

```
sudo systemctl daemon-reload && sudo systemctl restart local-fs.target

sudo reboot
```

# install packages and python to mounted usb drive

- a new python installation is needed because there is not enough space on the JetBot
- the new python installation is compiled from source and installed on the usb drive
- compiling from source requires openssl and libffi to be installed

## copy carsim repo to JetBot

```
scp -r carsim_no_mlagents jetbot@192.168.1.3:/mnt/usb_drive/carsim_no_mlagents
```

## copy replay to JetBot

```
scp -r episode_recordings jetbot@192.168.1.3:/mnt/usb_drive/carsim_no_mlagents/python/episode_recordings
```

## install openssl

openssl is installed to the usb drive at /mnt/usb_drive/.localssl

```
cd /mnt/usb_drive
wget https://www.openssl.org/source/openssl-1.1.1k.tar.gz
tar -xf openssl-1.1.1k.tar.gz
```

```
mkdir /mnt/usb_drive/.localssl
```

```
cd /mnt/usb_drive/openssl-1.1.1k
./config --prefix=/mnt/usb_drive/.localssl --openssldir=/mnt/usb_drive/.localssl shared zlib
make
sudo make install
```

## install _ctypes module

libffi is installed to the usb drive at /mnt/usb_drive/.localffi

download tar.gz
https://github.com/libffi/libffi/releases
https://github.com/libffi/libffi/archive/refs/tags/v3.4.6.tar.gz

extract to /mnt/usb_drive/libffi-3.4.6
```
tar -xf libffi-3.4.6.tar.gz
```

```
mkdir /mnt/usb_drive/.localffi
```

```
cd /mnt/usb_drive/libffi-3.4.6
./configure --prefix=/mnt/usb_drive/.localffi
make
sudo make install
```

## install python

```
cd /mnt/usb_drive
wget https://www.python.org/ftp/python/3.11.5/Python-3.11.5.tgz
tar xvf Python-3.11.5.tgz
cd Python-3.11.5
mkdir .localpython
./configure --with-openssl=/mnt/usb_drive/.localssl --prefix=/mnt/usb_drive/.localpython LDFLAGS="-L/mnt/usb_drive/.localffi/lib -Wl,--rpath=/mnt/usb_drive/.localffi/lib" CFLAGS="-I/mnt/usb_drive/.localffi/include" PKG_CONFIG_PATH="/mnt/usb_drive/.localffi/lib/pkgconfig"
make
sudo make install
```



python is now installed to .localpython
executable: /mnt/usb_drive/.localpython/bin/python3.11

### set new python installation as default
where is default python?

```
python -c "import sys; print(sys.executable)"
```


new symlink:

```
ln -sf /mnt/usb_drive/.localpython/bin/python3.11 /usr/bin/python
```

### create virtual env for python

https://docs.python.org/3/library/venv.html


```
cd /mnt/usb_drive
python -m venv sb3_env
```

### activate virtual env

```
cd /mnt/usb_drive
source sb3_env/bin/activate
```

### install packages to virtual env

```
python -m pip install -r carsim_no_mlagents-main/python/replay_only_requirements.txt
```

# run replay

The virtual env must be activated before running the replay!

```
python sb3_ppo_replay_only.py
```

## use specific onfig file

```
python sb3_ppo_replay_only.py --config-name cfg/...
```

# replay results

replaying episodes from /mnt/usb_drive/carsim_no_mlagents-main/python/episode_recordings/hardDistanceSuccess_recordings_desktop
replay episode results:
avg time for preprocessing and infer: 0.06583681669864026
max time for preprocessing and infer: 0.20911359786987305
preprocessing and infer times (maximum 0.20911359786987305) are below the timestep length of 0.3 seconds
this would leave 0.09088640213012694 seconds for the camera to take an image and send it to python for preprocessing and inferencing

# encountered problems

## neural network accuracy

The neural network's outputs change when executed on a different device. This is due to the different hardware and software configurations of the devices. It is NOT due to the action distribution sampling.

The replay_episode function in my_on_policy_algorithm.py uses the np.isclose to check if the neural network's outputs are correct. 


## installation

packages cannot be easily installed with apt
https://unix.stackexchange.com/questions/167231/installing-python-without-package-manager


The ./configure for python should show the following:
checking for openssl/ssl.h in /mnt/usb_drive/.localssl... yes
checking whether compiling and linking against OpenSSL works... yes
checking for --with-openssl-rpath...
checking whether OpenSSL provides required ssl module APIs... yes
checking whether OpenSSL provides required hashlib module APIs... yes
checking for --with-ssl-default-suites... python

### openssl is required for pip

### libffi is required for _ctypes module
_ctypes is required basically everywhere in python