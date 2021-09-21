# rl_toolbox - Actual implementation of PPO for quadrupedal locomotion

The name rl_toolbox comes from the many implementations of RL algorithms that I tested before I setteled with PPO implemented by [stable baseline 3](https://github.com/DLR-RM/stable-baselines3).

This repo contains the environment as well as the helper algorithms necessary to train a quadruped in simulation and deploy the neural networks on a real machine.

## Key features

- Gym environment to simulate a quadruped (specifically Idef'X) using the simulator [Erquy](https://github.com/o-Oscar/erquy).
- Helper functions to calculate IK and build a small library of motions for the RL to be based on. 
- Small reimplementation of PPO by [stable baseline 3](https://github.com/DLR-RM/stable-baselines3) to enforce symetrical gait and avoid very large kl-divergence. 
- Transfer algorithm to train a student that can only access measurable physical properties  against a teacher trained in RL that is given the full space of observations. Inspiration heavily drawn from this [paper](https://arxiv.org/pdf/2010.11251.pdf).

## Resulting policy

The baseline policy we can get is a simple walking policy, robust to small push on the robot.

https://user-images.githubusercontent.com/48491393/134125845-abba3d11-557e-4e37-90d2-0d54024840c0.mp4
