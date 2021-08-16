
# python3 train_teacher.py ; python3 train_student.py

from environments.dog_env import DogEnv
from config import Config

import time
import numpy as np

import gym

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common import utils
from stable_baselines3.common.utils import get_schedule_fn

import torch as th

class CustomCallback(BaseCallback):
	def __init__(self, config, verbose=0):
		super(CustomCallback, self).__init__(verbose)
		# Those variables will be accessible in the callback
		# (they are defined in the base class)
		# The RL model
		# self.model = None  # type: BaseAlgorithm
		# An alias for self.model.get_env(), the environment used for training
		# self.training_env = None  # type: Union[gym.Env, VecEnv, None]
		# Number of time the callback was called
		# self.n_calls = 0  # type: int
		# self.num_timesteps = 0  # type: int
		# local and global variables
		# self.locals = None  # type: Dict[str, Any]
		# self.globals = None  # type: Dict[str, Any]
		# The logger object, used to report things in the terminal
		# self.logger = None  # stable_baselines3.common.logger
		# # Sometimes, for event callback, it is useful
		# # to have access to the parent object
		# self.parent = None  # type: Optional[BaseCallback]
		self.config = config

	def _on_training_start(self) -> None:
		pass
	def _on_rollout_start(self) -> None:
		pass
	def _on_step(self) -> bool:
		return True
	def _on_training_end(self) -> None:
		pass
	def _on_rollout_end(self) -> None:
		self.model.save(config.models_save_path["teacher/PPO"].format(epoch=self.num_timesteps), exclude=["default_env"])
		# self.model.save(config.models_save_path["teacher/actor"].format(epoch=self.num_timesteps))

		self.model.batch_size = 128 # self.num_timesteps

		decay = 700000
		alpha = 1 if self.num_timesteps > decay else self.num_timesteps/decay
		des_log_std = (-1) * (1-alpha) + (-3) * alpha
		self.model.policy.log_std = th.nn.Parameter(self.model.policy.log_std*0 + des_log_std)

from my_ppo import MyPPO, TeacherActorCriticPolicy

from environments.dog_env import DogEnv_follow
from my_ppo import MyPPO, MotorActorCriticPolicy
from config import Config


def create_dog_env ():
	config = Config("follow_0", models_names=["follower/PPO"])
	env = DogEnv_follow(debug=False)
	motor_model = MyPPO.load(config.models_best_path["follower/PPO"], env=env, policy=MotorActorCriticPolicy)
	return Monitor(DogEnv(motor_model=motor_model))


if __name__ == "__main__":
	
	if False:
		check_env(DogEnv(debug=False))
		print("working !!")
		exit()

	config = Config("follow_0", models_names=["follower/PPO"])
	env = DogEnv_follow(debug=False)
	motor_model = MyPPO.load(config.models_best_path["follower/PPO"], env=env, policy=MotorActorCriticPolicy)


	config = Config("exp_0", models_names=["teacher/PPO", "teacher/tensorboard"])
	# env = SubprocVecEnv([lambda : Monitor(DogEnv(motor_model=motor_model)) for i in range(2)])
	env = SubprocVecEnv([create_dog_env for i in range(2)])
	default_env = DogEnv(debug=False)
	
	
	my_callback = CustomCallback(config)
	callback = CallbackList([my_callback])
	policy_kwargs = dict(
		log_std_init=-1.,
	)

	# model = MyPPO(TeacherActorCriticPolicy, env, default_env, policy_kwargs=policy_kwargs, verbose=1, batch_size=256, tensorboard_log=config.models_path["teacher/tensorboard"])
	model = MyPPO(TeacherActorCriticPolicy, env, default_env, policy_kwargs=policy_kwargs, verbose=1, batch_size=1024, tensorboard_log=config.models_path["teacher/tensorboard"])
	
	model.learn(total_timesteps=1000000, callback=callback)
	# model.learn(total_timesteps=1000, callback=callback)

	print("working !!")
	