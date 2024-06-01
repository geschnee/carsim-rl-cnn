# connect to jetbot

on jetbot exceute:
nmcli

ssh jetbot@ip

ip = ipv4 aus nmcli


check space on device
df -h

check size of files 
ls -lh

sorted ascending order:
du -hs * | sort -h

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

## create filesystem on the usb drive

sudo mkfs -t ext4 /dev/sda1

## create mount directory

mkdir /mnt/usb_drive

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


sudo reboot

# copy carsim repo to JetBot

scp -r carsim_no_mlagents jetbot@192.168.1.3:/mnt/usb_drive/carsim_no_mlagents

# copy replay to JetBot

scp -r episode_recordings jetbot@192.168.1.3:/mnt/usb_drive/carsim_no_mlagents/python/episode_recordings

# install openssl

wget https://www.openssl.org/source/openssl-1.1.1k.tar.gz
tar -xf openssl-1.1.1k.tar.gz

mkdir /mnt/usb_drive/.localssl

cd /mnt/usb_drive/openssl-1.1.1k
./config --prefix=/mnt/usb_drive/.localssl --openssldir=/mnt/usb_drive/.localssl shared zlib
make
make install


# install _ctypes

download tar.gz
https://github.com/libffi/libffi/releases

./configure --prefix=/mnt/usb_drive/.localffi
make
sudo make install

Libraries have been installed in:
   /mnt/usb_drive/.localffi/lib/../lib

If you ever happen to want to link against installed libraries
in a given directory, LIBDIR, you must either use libtool, and
specify the full pathname of the library, or use the '-LLIBDIR'
flag during linking and do at least one of the following:
   - add LIBDIR to the 'LD_LIBRARY_PATH' environment variable
     during execution
   - add LIBDIR to the 'LD_RUN_PATH' environment variable
     during linking
   - use the '-Wl,-rpath -Wl,LIBDIR' linker flag
   - have your system administrator add LIBDIR to '/etc/ld.so.conf'

https://stackoverflow.com/questions/65691539/locally-compiled-libffi-files-not-getting-picked-up-while-recompiling-python-3-p

TODO
https://github.com/yaml/pyyaml/issues/742



# install a new python (due to no space on default installation)

packages cannot be installed with apt
https://unix.stackexchange.com/questions/167231/installing-python-without-package-manager


source python_download.sh

The ./configure should show the following:
checking for openssl/ssl.h in /mnt/usb_drive/.localssl... yes
checking whether compiling and linking against OpenSSL works... yes
checking for --with-openssl-rpath...
checking whether OpenSSL provides required ssl module APIs... yes
checking whether OpenSSL provides required hashlib module APIs... yes
checking for --with-ssl-default-suites... python



python ist nun in localpython installiert
executable: /mnt/usb_drive/.localpython/bin/python3.11

## python installation als standard python setzen
wo ist python installiert?

python -c "import sys; print(sys.executable)"

neuen Link

ln -sf /mnt/usb_drive/.localpython/bin/python3.11 /usr/bin/python


## mit virtual env

https://pages.github.nceas.ucsb.edu/NCEAS/Computing/local_install_python_on_a_server.html

# create virtual env for python

https://docs.python.org/3/library/venv.html

cd /mnt/usb_drive
python -m venv sb3_env


## activate virtual env

cd /mnt/usb_drive
source sb3_env/bin/activate


## install packages

python -m pip install -r carsim_no_mlagents-main/python/replay_only_requirements.txt


# copy recordings

scp -r episode_recordings jetbot@192.168.1.2:/mnt/usb_drive/carsim_no_mlagents-main/python/episode_recordings

scp -r episode_recordings/episode_recordings_laptop_for_testing_on_linux_deterministic jetbot@192.168.1.2:/mnt/usb_drive/carsim_no_mlagents-main/python/episode_recordings/episode_recordings_laptop_for_testing_on_linux_deterministic

scp -r myPPO jetbot@192.168.1.2:/mnt/usb_drive/carsim_no_mlagents-main/python/myPPO

# run replay

python sb3_ppo_replay_only.py > replay_output.txt
