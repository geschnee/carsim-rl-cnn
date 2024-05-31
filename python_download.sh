wget https://www.python.org/ftp/python/3.11.5/Python-3.11.5.tgz
tar xvf Python-3.11.5.tgz
cd Python-3.11.5
mkdir .localpython
./configure --with-openssl=/mnt/usb_drive/.localssl --prefix=/mnt/usb_drive/.localpython
make
sudo make install