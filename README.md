# FTN Solo
A repository for everything Solo related developed on FTN

## Instalation
```
sudo apt-get install python3-virtualenv

git clone git@github.com:ajsmilutin/ftn_solo.git

virtualenv solo_env && source solo_env/bin/activate
cd ftn_solo
pip install -r requirements.txt 
```

And you should be able to see something like this:
![Mujoco starting up](images/mujoco_and_solo.gif)

## Installing Solo12 interfaces

Getting kinematic / dynamic model and interfacing with robot is easies through [`robot_properties_solo`](https://github.com/open-dynamic-robot-initiative/robot_properties_solo)

You don't have to install `pinocchio` since it's already installed through `pip`. So we can install lightweight `robot_properties_solo`:
```
git clone git@github.com:ajsmilutin/robot_properties_solo.git
cd robot_properties_solo
pip3 install .
```

After finishing run the visualize script:
```
python visualize.py 
```
