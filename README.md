# MASTERS PROJECT - BOT

## Project Setup

### Components

| Component         | Info                        |
| ----------------- | --------------------------- |
| Jetson TX1        |                             |
| Ubuntu 16.04      |                             |
| ROS Kinetic       |                             |
| Hokuyo Lidar      | apt: `ros-kinetic-urg-node` |
| ZED Stereo Camera | sdk: `v1.2`                 |

### Users

To SSH into the Jetson use your own account so that all git pushes/pulls are from your own user account. By default there are 4 accounts:

```sh
ssh av@[jetsontx1]  # Andrew
ssh el@[jetsontx1]  # Elton
ssh ep@[jetsontx1]  # Eric
ssh jg@[jetsontx1]  # James
```

There is a shared folder that all users can access located at: `/home/shared` and is symlinked into your own home folder `~/shared`. Use this folder for the project work so that all users can access the data.

A visual representation of the folder structure is shown below:

```
/home
    /shared             # shared folder
        /masters-bot    # shared project folder
    /av
        /shared         # symlink points to /home/shared
    /el
        /shared         # symlink points to /home/shared
    /ep
        /shared         # symlink points to /home/shared
    /jg
        /shared         # symlink points to /home/shared
```

## Development

### Cloning the repo 

When first cloning this repository you must run the `init.sh` script to properly setup the catkin workspace. This assumes that ROS-Kinetic is already installed. 

You will only need to run this step once when first cloning this repository so ROS can set up the proper build folders and scripts for your machine.

```sh
git clone https://github.com/atkvo/masters-bot.git
cd masters-bot
chmod +x init.sh
./init.sh
```

### Editor Settings

Please use the following editor settings to make sure source code is consistent across all users.

| Setting | Value | Comment |
| ------- | ----- | ------- |
| tab style | use `spaces` | This is so that all source has the same formatting regardless of viewer settings |
| tab size | 4 spaces | Python requires 4 spaces as an indentation |

If these rules aren't followed, then please at the minimum **do NOT mix tabs and spaces**. Stick to one. Especially with python code.

### Git Repo

For consistency and tidyness, please follow some of the `git` guidelines for this repo.

#### Workflow

It's recommended to work *in your own branch* and not the `master` branch. The master branch should only contain snapshots of code that are working.

Example

```sh
# assume in master branch
git branch devfeature
git checkout devfeature
# shortcut: git checkout -b devfeature

# You are now in the 'devfeature' branch. do all your work here
# to push to the upstream, you must create a ref on the remote
# Run the following command to create the remote branch + push to it.  
git push --set-upstream origin devfeature # this command only needs to be run once

# If the branch is already created on the remote you can just push to it with:
git push

# Merging the changes into master branch.
# Assumes you're in your 'devfeature' branch:
git pull # syncs your devfeature branch with upstream
git checkout master

# You're now in the master branch
git pull             # sync local master branch with the upstream branch
git merge devfeature # merges your 'devfeature' branch into 'master'
git push --all       # updates the remote branches
```

#### Commit Messages

All commit messages should be in the form of "This commit will ___" where your message is in the blank space. The first letter must be capitalized as well and not end with a period. 

For example: 

Correct way:
```sh
git commit -m "Add a new feature"
```

**Incorrect**
```sh
git commit -m "added a new feature."
```

##### Long Messages
Try to keep the commit message short and simple. If you need a longer description, add a **second `-m`** flag to put the longer message there. 

Example:

```sh
git commit -m "Fix a bug" -m "The bug did so and so and to fix it I did this and that. The bug arose because of a specific reason"
```

### Committing Files
Don't blindly run `git add .` or `git commit --all`. Check `git status` to make sure no garbage files are added to your commit.

The `.gitignore` file will include some common filetypes not to include like compiled python files (`.pyc`), or temporary vim files `.swp`. If the `.gitignore` is missing a file extension, please add to it. 

Also never try to commit generated files/folders. These include the `build` and `devel` directories. Only commit the `src` folder and allow the `catkin*` commands to generate the proper folders/files.

## Troubleshooting Common Issues

### My ROS packages are not showing up

In the root directory of the project workspace make sure to source the `devel/setup.bash` file first. This must be done in every new shell session.

```sh
cd masters-bot
source devel/setup.bash
```

### Unable to run `roscore` due to incorrect ROS_IP
> **NOTE:** This may  not apply to the current configuration since a bridge is no longer used. 

Make sure that the environment variable `ROS_IP` is properly set to the Jetson's IP address on the bridge `br0`. View this IP address by running the command `ifconfig` and looking at the IP address for `br0`

Example
> Note: The output below is not real. Your output may be different 

```sh
ifconfig
br0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
    inet6 fe80::1097:d471:db27:21d6%en0 prefixlen 64 secured scopeid 0x4
    inet 192.168.1.132 netmask 0xffffff00 broadcast 192.168.1.255
    nd6 options=201<PERFORMNUD,DAD>
    status: active

```

The important section to note is the **`inet 192.168.1.143`**. Use this IP for the ROS_IP like so:

```sh
export ROS_IP=192.168.1.132
```

You will need to do this for **every terminal** opened, or you can add the line to `.bashrc` 

## Setting up the Jetson TX1

### ROS Kinetic (Ubuntu 16.04)
```sh
# Add ROS repository
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# Add ROS key
sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116

# Install ROS Kinetic
sudo apt-get install ros-kinetic-desktop-full

sudo apt install ros-kinetic-urg-node

# Setup ROS
sudo rosdep init
# sudo c_rehash /etc/ssl/certs  # Run this if the above command fails
rosdep update

echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt-get install python-rosinstall
```

### Network Setup

The table below shows the Jetson's IP address on each of its interfaces after following the setup instructions.

| Interface | Jetson IP     | Comments                                |
| --------- | ------------- | --------------------------------------- |
| `eth0`    | 192.168.x.xx  | Used for Internet connectivity          |
| `hokuyo0` | 192.168.13.14 | Used for Lidar                          |
| `wlan0`   | 192.168.8.1   | Used so that users can SSH into the TX1 |
| `tpspot0` | 192.168.0.100 | Used so that users can SSH into the TX1 |

The Hokuyo Lidar's IP address is set to **`192.168.13.15`** so that it doesn't conflict with the common network subnets `192.168.1.x`.



#### Setting up the Hokuyo Lidar

To set the IP of the Hokuyo Lidar:
```sh
# Install required package
sudo apt install ros-kinetic-urg-node
rosrun urg_node urg_node set_urg_ip.py 192.168.13.15 192.168.13.1 --nm 255.255.255.0 --ip 192.168.13.14
```

Add the following to `/etc/network/interfaces`
> Note: the `metric 2000` line ensures that linux prioritizes eth0 over eth1 for Internet connectivity 

```
# Setup eth1 (Hokuyo)
auto eth1
iface eth1 inet static
address 192.168.13.14
netmask 255.255.255.0
metric 2000

# Automatically activated by hotplug subsystem
# Internet network interface
auto eth0
iface eth0 inet dhcp
```

#### Setting up SSH access on the TX1
> The TX1 has a built in wifi chip (Broadcom?). The best configuration of the TX1 that's useable at SJSU is the following:


You made need to create the file if necessary. 
First add this line to `/etc/modprobe.d/bcmdhd.conf`
> This line will allow the Jetson TX1 to broadcast its AP 
```
options bcmdhd op_mode=2
```

##### Install Utilities

`hostapd` is used to create the network hotspot
`dnsmaskq` is used as a mini dns server to give the clients IP addresses

```sh
sudo apt-get update
sudo apt-get install hostapd dnsmaskq
```

##### Setup `hostapd`

Create the file `/etc/hostapd/hostapd.conf` and write the following to it:
Note that you can change the `ssid` option to whatever you'd like to name the wifi AP.
```
interface=wlan0
driver=nl80211
ssid=jeonjetsontx1
macaddr_acl=0
channel=7
ieee8021x=0
eap_server=0
```

Add to the following to `/etc/network/interface`:
```
auto wlan0
iface wlan0 inet static
hostapd /etc/hostapd/hostapd.conf
address 192.168.8.1
netmask 255.255.255.0
```

##### Setup `dnsmaskq`

There is a file called `/etc/dnsmasq.conf` that is already populated with a lot of options that are commented out. Either find the corresponding lines and uncomment/fill out the variables or just add these to the top of the file:

```
interface=lo,wlan0
no-dhcp-interface=lo
dhcp-range=192.168.8.20,192.168.8.254,255.255.255.0,12h
```

Note the `dhcp-range` parameter. All connecting devices will get an IP address between `192.168.8.20` and `192.168.8.254` in the above configuration.

##### Apply Changes

Start the `hostapd` and `dnsmasq` services and then reboot the system
```sh
sudo systemctl start dnsmasq
sudo systemctl start hostapd
sudo reboot now
```

#### /etc/network/interfaces

The final `/etc/network/interfaces` file looks like this:
```
# interfaces(5) file used by ifup(8) and ifdown(8)
# Include files from /etc/network/interfaces.d:
source-directory /etc/network/interfaces.d

# Setup hotspot from wlan0
auto wlan0
    iface wlan0 inet static
    hostapd /etc/hostapd/hostapd.conf
    address 192.168.8.1
    netmask 255.255.255.0

# Setup eth1 (Hokuyo)
auto eth1
    iface eth1 inet static
    address 192.168.13.14
    netmask 255.255.255.0
    metric 2000

# Setup for use with external WiFi AP
auto eth2
    iface eth2 inet static
    address 192.168.0.100
    netmask 255.255.255.0
    gateway 192.168.0.254
    metric 1500

# Automatically activated by hotplug subsystem
# Internet network interface
auto eth0
    iface eth0 inet manual
```


If there are problems connecting to the Internet via `eth0`, use this command:

```
dhclient eth0
```


### Optional Tools

Install `vim`
```
sudo apt-get install vim
```

Install `tmux`. This program is extremely useful in managing multiple shell sessions over SSH or in a single terminal window.
```
sudo apt-get install tmux
```

> TODO: Add sane defaults for `vim` and `tmux`

