wget https://www.python.org/ftp/python/3.11.5/Python-3.11.5.tgz
tar xvf Python-3.11.5.tgz
cd Python-3.11.5
mkdir .localpython
./configure --with-openssl=/mnt/usb_drive/.localssl --prefix=/mnt/usb_drive/.localpython LDFLAGS="-L/mnt/usb_drive/.localffi/lib -Wl,--rpath=/mnt/usb_drive/.localffi/lib" CFLAGS="-I/mnt/usb_drive/.localffi/include" PKG_CONFIG_PATH="/mnt/usb_drive/.localffi/lib/pkgconfig"
make
sudo make install