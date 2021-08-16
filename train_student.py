from environments.dog_env import DogEnv, State
from environments.dog_env.obs_gen import RealisticObsGenerator
from torch.utils.tensorboard import SummaryWriter
from config import Config
import time
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
import gym

import torch as th
from torch import nn
from models import StudentModule

p_student = 0.5
log_std = -2
h = 5 # 30
all_p_student = []


base_n_rollout = 1
base_n_success = 1
base_suc_prob = 1

switch_legs = np.asarray([	[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
							[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
							[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
							[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], ]).astype(np.float32)

def generate_dataset (teacher, student, env, all_obs_gen):
	global p_student
	global log_std
	global h
	
	
	obs_dataset = []
	act_dataset = []
	act_student_dataset = []
	mask_dataset = []
	latent_dataset = []
	
	i = 0
	while i < 5:
	
		done = False
		increase_student_prob = False
		obs = env.reset()
		for gen in all_obs_gen:
			gen.reset()
		
		all_act = []
		all_student_act = []
		all_obs = []
		all_latent = []
		all_mask = []
		
		t = 0
		p_student = min(max(p_student, 0), 1)
		all_p_student.append(p_student)
		
		if p_student < 0.9:
			log_std = -3
		print("p_student :", p_student)
		print("log_std :", log_std)
		# while not done and t < 50: # 200
		while t < 200:
			t += 1
			
			# obs = obs.astype(np.float32)
			# --- logging the observations ---
			all_obs.append(np.stack([gen.generate() for gen in all_obs_gen]))
			stacked_obs = np.stack(all_obs, axis=1).astype(np.float32)
			
			# --- stepping the simulation ---
			# teacher_action = teacher(th.tensor(np.expand_dims(obs, axis=0)))
			teacher_action = model.predict(obs, deterministic=True)[0]
			teacher_latent = model.policy.mlp_extractor.pi_network.sim_network(th.tensor(obs["sim_obs"].astype(np.float32)))
			# print(teacher_latent.shape)
			# teacher_action = teacher({key:np.expand_dims(x, axis=0) for key, x in obs.items()})
			obs_id = int(np.random.random()*len(all_obs_gen))
			student_action = student(th.tensor(stacked_obs))[0][obs_id, -1].numpy()
			if np.random.random() < p_student or True:
				rand_action = student_action
			else:
				rand_action = teacher_action
			rand_action = rand_action + np.random.normal(size=rand_action.shape) * np.exp(log_std)
			obs, rewards, done, info = env.step(rand_action)
			# done = False
			increase_student_prob = increase_student_prob or done
				
			# --- logging the action ---
			all_act.append(np.expand_dims(teacher_action, 0))
			all_latent.append(np.expand_dims(teacher_latent, 0))
			all_student_act.append(np.expand_dims(student_action, 0))
			all_mask.append(done)
		
		if increase_student_prob:
			p_student -= 1/h * base_suc_prob
			log_std -= 1/h * base_suc_prob
			pass
		else:
			p_student += 1/h * (2-base_suc_prob)
			if p_student > 0.9:
				log_std += 1/h * (2-base_suc_prob)
		i += 1
			
		obs_dataset.append(np.stack(all_obs, axis=1)) # batch, time, obs
		act_dataset.append(np.repeat(np.expand_dims(np.concatenate(all_act, axis=0), axis=0), len(all_obs_gen), axis=0)) # batch, time, act
		act_student_dataset.append(np.repeat(np.expand_dims(np.concatenate(all_student_act, axis=0), axis=0), len(all_obs_gen), axis=0)) # batch, time, act
		latent_dataset.append(np.repeat(np.expand_dims(np.concatenate(all_latent, axis=0), axis=0), len(all_obs_gen), axis=0))
		mask_dataset.append(np.repeat(np.expand_dims(1-np.asarray(all_mask).astype(np.float32), axis=0), len(all_obs_gen), axis=0))
		
		h += 1

	obs_dataset = np.concatenate(obs_dataset, axis=0).astype(np.float32)
	act_dataset = np.concatenate(act_dataset, axis=0).astype(np.float32)
	act_student_dataset = np.concatenate(act_student_dataset, axis=0).astype(np.float32)
	latent_dataset = np.concatenate(latent_dataset, axis=0).astype(np.float32)
	for m in mask_dataset:
		print (m.shape)
	mask_dataset = np.concatenate(mask_dataset, axis=0).astype(np.float32)
	print(mask_dataset.shape)
	return th.tensor(obs_dataset), th.tensor(act_dataset), th.tensor(act_student_dataset), th.tensor(latent_dataset), th.tensor(mask_dataset)


def make_teacher_from_PPO (model):
	return lambda obs: model.predict(obs, deterministic=True)[0]

def fix_parameters (student_model):
	for param in student_model.action_net.parameters():
		param.requires_grad = False
	for param in student_model.mlp_extractor.main_network.parameters():
		param.requires_grad = False


if __name__ == "__main__":

	env = DogEnv(debug=False)
	config = Config("exp_0", models_names=["teacher/PPO", "student/model", "student/tensorboard"])
	model = PPO.load(config.models_best_path["teacher/PPO"])

	new_states = [State() for i in range(10)]
	for state in new_states:
		state.update_t = int(np.random.random()*3)

	env.sim.all_states = env.sim.all_states + new_states
	all_obs_gen = [RealisticObsGenerator(state) for state in env.sim.all_states]
	
	student = StudentModule(env.get_box_space(all_obs_gen[0].obs_dim))
	student_optimize = StudentModule(env.get_box_space(all_obs_gen[0].obs_dim))
	# print(model.policy.action_net.state_dict().keys())
	# print(model.policy.mlp_extractor.pi_network.main_network.state_dict().keys())
	# print(student_module.action_net.state_dict().keys())
	# print(student_module.mlp_extractor.main_network.state_dict().keys())

	# setting the parameters of the main_network of the student_model
	with th.no_grad():
		student.action_net.load_state_dict(model.policy.action_net.state_dict())
		student.mlp_extractor.main_network.load_state_dict(model.policy.mlp_extractor.pi_network.main_network.state_dict())
		student_optimize.action_net.load_state_dict(model.policy.action_net.state_dict())
		student_optimize.mlp_extractor.main_network.load_state_dict(model.policy.mlp_extractor.pi_network.main_network.state_dict())

	# fixing the parameters of the student_model
	fix_parameters(student)
	fix_parameters(student_optimize)
	
	print(model.policy.action_net.state_dict()["bias"][0])
	print(student.action_net.state_dict()["bias"][0])

	writer = SummaryWriter(log_dir=config.models_path["student/tensorboard"])


	with th.no_grad():
		state_dict = student.mlp_extractor.cnn_network.state_dict()
		student_optimize.mlp_extractor.cnn_network.load_state_dict(state_dict)

	# optimizer = th.optim.Adam(student.parameters(), lr=0.0001)

	alpha = 0.9
	for i in range(80):
		print("Step {}:".format(i))
		with th.no_grad():
			obs_dataset, act_dataset, act_student_dataset, latent_dataset, mask_dataset = generate_dataset(model, student, env, all_obs_gen)
		with th.no_grad():
			state_dict = student.state_dict()
			student_optimize.load_state_dict(state_dict)

		optimizer = th.optim.Adam(student_optimize.parameters(), lr=0.001)
		print("Test : ", th.mean(th.mean(th.square(student_optimize(obs_dataset)[0]-act_student_dataset), 2) * mask_dataset).detach().numpy())
		for j in range(100):
			pred, lat_pred = student_optimize(obs_dataset)
			# pred = student(obs_dataset)
			act_loss = th.mean(th.mean(th.square(act_dataset-pred), 2) * mask_dataset)
			lat_loss = th.mean(th.mean(th.square(latent_dataset-lat_pred), 2) * mask_dataset)
			optimizer.zero_grad()

			sym_obs_matrix = all_obs_gen[0].get_sym_obs_matrix()
			sym_obs = th.matmul(obs_dataset, th.from_numpy(sym_obs_matrix))
			sym_act = th.matmul(student_optimize(sym_obs)[0], th.from_numpy(switch_legs))

			# sym_loss = th.mean(th.square(sym_act - self.policy._predict(rollout_data.observations).detach()), dim=1)
			sym_loss = th.mean(th.square(sym_act - pred), dim=(1,2))
			sym_loss = th.mean(sym_loss, dim=0)

			loss = act_loss + lat_loss * 0.01 + sym_loss * 0.1
			# loss = lat_loss
			loss.backward()

			optimizer.step()
		print("Step: {}, Initial Loss: {}".format(j, loss.detach().numpy()))
		
		writer.add_scalar('train_student/act_loss', act_loss.detach().numpy(), i)
		writer.add_scalar('train_student/lat_loss', lat_loss.detach().numpy(), i)
		writer.add_scalar('train_student/sym_loss', sym_loss.detach().numpy(), i)
		writer.add_scalar('train_student/loss', loss.detach().numpy(), i)
		writer.add_scalar('train_student/p_student', p_student, i)
		writer.add_scalar('train_student/log_std', log_std, i)
		writer.add_scalar('train_student/mean_ep_len', np.mean(np.sum(mask_dataset.detach().numpy(), axis=1), axis=0), i)

		# student_optimize.fit(obs_dataset, act_dataset, batch_size=30, epochs=100, verbose=2, shuffle=False)
		
		with th.no_grad():
			state_dict_optimize = student_optimize.mlp_extractor.cnn_network.state_dict()
			state_dict = student.mlp_extractor.cnn_network.state_dict()
			for name in state_dict:
				state_dict[name] = state_dict[name] * (1-alpha) + state_dict_optimize[name] * alpha
			student.mlp_extractor.cnn_network.load_state_dict(state_dict)

		
		
		print(student.action_net.state_dict()["bias"][0])
		# --- save logs ---
		# student.save_weights(os.path.join(config.student_model_path,"student_{}".format(i)))
		th.save(student.state_dict(), config.models_save_path["student/model"].format(epoch=i))
		# plt.clf()
		# plt.plot(all_p_student)
		# plt.savefig('student_training.png')
	


