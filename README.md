# FTN Solo
A repository for everything Solo related developed on FTN

## Instalation

### ROS
First thing to install is definitely ROS2. To do so, follow the instructions [here](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)

### Prepparing the environment
First of all let's create ROS2 workspace and clone the repositories needed

```
export INSTALL_DIR=test_ws
mkdir -p $INSTALL_DIR/src
cd $INSTALL_DIR/src
git clone https://github.com/ajsmilutin/robot_properties_solo.git
git clone https://github.com/ajsmilutin/ftn_solo.git
git clone https://github.com/ajsmilutin/ftn_solo_control.git
cd ..
```

Then let's install the python dependencies:
```
sudo apt-get install python3-virtualenv libboost-all-dev
sudo apt install ros-jazzy-rviz2 ros-jazzy-joy ros-jazzy-xacro ros-jazzy-proxsuite ros-jazzy-eigenpy
```

Unfortunately we have to install pinocchio ourselves as we need version 3

```
cd ..
git clone --recursive https://github.com/stack-of-tasks/pinocchio
cd pinocchio && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/ros/jazzy/
make -j2
make install
cd ../..

# make a symlink to python bindings
ln -s /opt/ros/jazzy/lib/python3.12/dist-packages/pinocchio  /opt/ros/jazzy/lib/python3.12/site-packages/


```

Let's create virtual environment
```
cd $INSTALL_DIR
virtualenv solo_env
source solo_env/bin/activate
pip install -r src/ftn_solo/requirements.txt 
touch solo_env/COLCON_IGNORE
```

Let's now compile (twice) and run some demo!:
```
source /opt/ros/jazzy/setup.bash
colcon build && source install/setup.bash
# HACK! We need 2 compiles of robot_propperties_solo
colcon build --packages-select robot_properties_solo && source install/setup.bash 
ros2 launch ftn_solo robot_launch.py hardware:=mujoco task:=move_base config:=up_slope_60.yaml
```
And you should be able to see something like this:
![Mujoco starting up](images/mujoco_and_solo.gif)
