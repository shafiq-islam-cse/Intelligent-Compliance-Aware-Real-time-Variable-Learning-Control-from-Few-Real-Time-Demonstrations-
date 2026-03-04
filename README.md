
# ALOHA Haptic Teleoperation for the proposed Intelligent Compliance-Aware Real-time Variable Learning Control from Few Real-Time Demonstrations for Joint-level Robust Bimanual Teleoperation

High-fidelity dual-arm teleoperation for the ALOHA robot using 3D Systems Haptic devices. Features a custom Bi-Action Chunking Policy (Transformers + GAT + Meta-Learning) and Kalman Filtering for smooth, compliant, and stable motion.

## Features

Haptic Teleoperation: Direct control via 3D Systems Geomagic/Touch devices.
Adaptive RL Policy: Uses Graph Attention Networks (GAT) and Meta-Learning to blend haptic input with learned trajectories.
Signal Processing: Kalman Filtering for optimal smoothing.
Adaptive Compliance: Modulates stiffness based on velocity.
Automated Reporting: Generates a PDF analysis of smoothness, jerk, and control effort after each session.

## Setup

This project requires the official ALOHA MuJoCo meshes, which are not included in this repo.

1. Download Assets:

```bash
git clone https://github.com/agilexrobotics/mobile_aloha_sim.git
```

2. Locate Path:
   Find `mobile_aloha_sim-master/aloha_mujoco/aloha/meshes_mujoco/aloha_v1.xml`.

3. Update Code:
   Edit the `XML_PATH` variable in the script to point to your downloaded file.

## Installation

Prerequisites: Ubuntu (20.04/22.04), Python 3.8+, 3D Systems OpenHaptics drivers, Geomagic Touch device.

Dependencies:

```bash
pip install numpy opencv-python matplotlib scipy pyOpenHaptics mujoco
```

## Usage

1. Connect and power on your haptic device.
2. Run the script:

```bash
python your_script_name.py
```

3. Controls: (If you have one haptic device to control)
   r: Switch to Right Arm
   l: Switch to Left Arm
   ESC: Exit simulation and generate report
   Haptic Buttons: Top/Bottom to control gripper open/close.

4. Results: Close the pop-up graph windows to finalize the PDF report generation.

## Credits

Robot Model: [AgileXRobotics / mobile_aloha_sim](https://github.com/agilexrobotics/mobile_aloha_sim)
Haptics: 3D Systems OpenHaptics
