_Version 1.2_

# Jetbot Einrichtungsanleitung

Diese Anleitung hat den Anspruch als allgemein gültiges Rezept zum Neuaufsetzen eines Jetbot's zu gelten. Dafür bitte ich dich, wenn du zusätzliche oder alternative Schritte durchführen musstest, diese Anleitung gegebenenfalls zu überarbeiten und gemeinsam auf dem aktuellsten Stand zu halten.

# Inhaltsverzeichnis
for [English Version](#english-version) see below

1. [Benötigte Hardware und Software](#benötigte-hardware-und-software)
2. [Software Setup](#software-setup)
   - [Download des SD Karten Images](#download-des-sd-karten-images)
   - [Flashen der SD Karte mit Etcher](#flashen-der-sd-karte-mit-etcher)
3. [Lokales Netzwerk vorbereiten](#lokales-netzwerk-vorbereiten)
4. [Erster Start des Jetson](#erster-start-des-jetson)
    - [SSH Verbindung von deinem Computer zum Jetbot herstellen](#ssh-verbindung-von-deinem-computer-zum-jetbot-herstellen)
    - [Tastaturkonfiguration zu Deutsch ändern](#tastaturkonfiguration-zu-deutsch-ändern)
    - [JetBot über WLAN verbinden](#jetbot-über-wlan-verbinden)
5. [Über den Webbrowser mit dem JetBot verbinden](#über-den-webbrowser-mit-dem-jetbot-verbinden)

<br>
<br>

<img src="https://jetbot.org/master/images/jetbot_800x630.png" alt="jetson" width="400"/>

<br>

## Benötigte Hardware und Software

- JetBot Hardware
    - Jetson Nano Dev Kit
    - [Aufbauanleitung](https://jetbot.org/master/hardware_setup.html)
- Computer
- SD Karte (min. 32GB)
- SD Kartenlesegerät
- Lokales Netzwerk
    - Router (inkl. Stromkabel)
    - LAN-Kabel

<br>
<br>

## Software Setup

Im folgenden wirst du das JetBot SD Karten Image auf deine SD Karte flashen

### Download des SD Karten Images
Downloade das aktuelle JetBot Image für dein Jetson Nano Modell

**JETSON 4G:**
 [jetbot-043\_nano-4gb-jp45.zip](https://drive.google.com/file/d/1o08RPDRZuDloP_o76tCoSngvq1CVuCDh/view?usp=sharing)

**JETSON 2G:**
[jetbot-043\_nano-2gb-jp45.zip](https://drive.google.com/file/d/1tsuSY3iZrfiKu4ww-RX-eCPcwuT2DPwJ/view?usp=sharing)

### Flashen der SD Karte mit Etcher

Nutze [Etcher](https://www.balena.io/etcher/), wähle das heruntergeladene Image aus und flashe es auf die SD Karte

<br>

## Lokales Netzwerk vorbereiten

Um mit dem JetBot via Secure Shell (SSH) reibungslos zu kommunizieren bietet es sich an ein eigenes Netzwerk aufzuspannen.
Dafür brauchst du einen Router und ein bis zwei LAN-Kabel.

Das LAN-Kabel wird nur zum ersten Start benötigt.

<br>

## Erster Start des Jetson

1.  Setze die SD Karte in den dafür vorgesehenen Slot auf der Unterseite des Jetson Nano Moduls

2. Schließe ein LAN-Kabel zwischen dem Router und dem Jetson Nano an.

3. Schließe das Stromkabel oder die Powerbank an den Micro-USB Power Slot des Jetson Moduls um den Jetson Nano zu starten
    - Es sollte nun der Power-Indikator grün leuchten und nach ca. 20 Sekunden sollte das piOLED Display die IP Adresse und weitere Infos anzeigen

### SSH Verbindung von deinem Computer zum Jetbot herstellen

1. Öffne ein neues Terminal auf deinem Computer ( `Strg + Alt + t` )

    ```
    ssh -C jetbot@<IP-ADRESSE_JETBOT>
    ```
2. Logge dich mit dem Passwort `jetbot` ein

### Tastaturkonfiguration zu Deutsch ändern

```
sudo dpkg-reconfigure keyboard-configuration
```
Wähle generic keyboard with 105 keys (Intl) > Germany > German. Und bestätige weitere Default Einstellungen.

### JetBot über WLAN verbinden

In diesem Schritt werden wir den JetBot mit dem WLAN verbinden, damit wir nicht mehr auf das LAN-Kabel angewiesen sind.

Mit diesem Befehl kannst du dir ggf. alle verfügbaren drahtlosen Netzwerke (SSIDs) als Liste ausgeben lassen:
```
nmcli device wifi list
```     

Verbinde dich mit dem gleichen Netzwerk, das auch dein Computer nutzt:
```
sudo nmcli device wifi connect <SSID> password '<PASSWORT>'
```

Starte deinen JetBot anschließend einmal neu:
```
sudo shutdown -h now
```
Kappe die Stromzufuhr, entferne das LAN Kabel, warte 10 Sekunden und schliesse dann das Micro-USB Kabel wieder an das Board. Der JetBot sollte sich nun automatisch beim Start über WLAN verbinden.

<br>

## Über den Webbrowser mit dem JetBot verbinden

Notiere dir die IP-Adresse, die du auf dem _piOLED_ Display siehst und navigiere in deinem Browser zu `http://<JETBOT_IP_ADRESSE>:8888`
<br>
<br>
Das Passwort ist `jetbot`

<br>
<br>
<br>
<br>


#### English Version

# Jetbot Setup Guide

This guide aims to serve as a universally applicable recipe for setting up a Jetbot. If you had to perform additional or alternative steps, please revise this guide accordingly to keep it up to date.

# Table of Contents

1. [Required Hardware and Software](#required-hardware-and-software)
2. [Software Setup](#software-setup-1)
   - [Downloading the SD Card Image](#downloading-the-sd-card-image)
   - [Flashing the SD Card with Etcher](#flashing-the-sd-card-with-etcher)
3. [Preparing the Local Network](#preparing-the-local-network)
4. [First Start of the Jetson](#first-start-of-the-jetson)
    - [Establishing SSH Connection from Your Computer to the Jetbot](#establishing-ssh-connection-from-your-computer-to-the-jetbot)
    - [Changing Keyboard Configuration to German](#changing-keyboard-configuration-to-german)
    - [Connecting JetBot to Wi-Fi](#connecting-jetbot-to-wi-fi)
5. [Connecting to the JetBot via Web Browser](#connecting-to-the-jetbot-via-web-browser)

<br>
<br>

<img src="https://jetbot.org/master/images/jetbot_800x630.png" alt="jetson" width="400"/>

<br>

## Required Hardware and Software

- JetBot Hardware
    - Jetson Nano Dev Kit
    - [Assembly Instructions](https://jetbot.org/master/hardware_setup.html)
- Computer
- SD Card (min. 32GB)
- SD Card Reader
- Local Network
    - Router (including power cable)
    - LAN Cable

<br>
<br>

## Software Setup

Next, you will flash the JetBot SD card image onto your SD card.

### Downloading the SD Card Image
Download the latest JetBot image for your Jetson Nano model.

**JETSON 4G:**
 [jetbot-043_nano-4gb-jp45.zip](https://drive.google.com/file/d/1o08RPDRZuDloP_o76tCoSngvq1CVuCDh/view?usp=sharing)

**JETSON 2G:**
[jetbot-043_nano-2gb-jp45.zip](https://drive.google.com/file/d/1tsuSY3iZrfiKu4ww-RX-eCPcwuT2DPwJ/view?usp=sharing)

### Flashing the SD Card with Etcher

Use [Etcher](https://www.balena.io/etcher/), select the downloaded image, and flash it onto the SD card.

<br>

## Preparing the Local Network

To smoothly communicate with the JetBot via Secure Shell (SSH), it's advisable to set up a dedicated network. You'll need a router and one or two LAN cables.

The LAN cable is only needed for the first start.

<br>

## First Start of the Jetson

1. Insert the SD card into the designated slot on the underside of the Jetson Nano module.

2. Connect a LAN cable between the router and the Jetson Nano.

3. Connect the power cable or power bank to the Micro-USB power slot of the Jetson module to start the Jetson Nano.
    - The power indicator should now turn green, and after about 20 seconds, the piOLED display should show the IP address and other information.

### Establishing SSH Connection from Your Computer to the Jetbot

1. Open a new terminal on your computer (`Ctrl + Alt + t`).

    ```
    ssh -C jetbot@<JETBOT_IP_ADDRESS>
    ```
2. Log in with the password `jetbot`.

### Changing Keyboard Configuration to German

```
sudo dpkg-reconfigure keyboard-configuration
```
Select generic keyboard with 105 keys (Intl) > Germany > German. Confirm the other default settings.

### Connecting JetBot to Wi-Fi

In this step, we will connect the JetBot to Wi-Fi, so we are no longer reliant on the LAN cable.

Use this command to display a list of all available wireless networks (SSIDs):
```
nmcli device wifi list
```     

Connect to the same network your computer is using:
```
sudo nmcli device wifi connect <SSID> password '<PASSWORD>'
```

Restart your JetBot afterwards:
```
sudo shutdown -h now
```
Disconnect the power supply, remove the LAN cable, wait 10 seconds, and then reconnect the Micro-USB cable to the board. The JetBot should now automatically connect to Wi-Fi upon starting.

<br>

## Connecting to the JetBot via Web Browser

Note the IP address displayed on the _piOLED_ and navigate in your browser to `http://<JETBOT_IP_ADDRESS>:8888`
<br>
<br>
The password is `jetbot`