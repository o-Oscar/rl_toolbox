"""
cd C:\\Users\\oscbo\\Documents\\Travail\\PSC\\raisimLib\\raisimUnity\\win32
RaiSimUnity.exe

cd C:\\Users\\oscbo\\Documents\\Travail\\PSC\\idefX\\v3\\rl_toolbox
conda activate psc
python train_transfert.py

python environments\\dog_env_rai\\src\\dog_urdf\\create_urdf.py

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf

import config
from to_lite.full_model import get_teacher_model, get_student_model
from environments.dog_env_rai import DogEnv, State
# from models.actor import SimpleActor
from environments.wrapper.stacking import get_stacked_env
import sys
np.set_printoptions(threshold=sys.maxsize)

p_student = 0.5
log_std = -2
h = 3
all_p_student = []


base_n_rollout = 1
base_n_success = 1
base_suc_prob = 1

def generate_dataset (teacher, student, env, all_obs_gen):
	global p_student
	global log_std
	global h
	
	
	obs_dataset = []
	act_dataset = []
	mask_dataset = []
	
	i = 0
	while i < 5:
	
		done = False
		has_been_done = False
		obs = env.reset()
		for gen in all_obs_gen:
			gen.reset()
		
		all_act = []
		all_obs = []
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
			
			# --- logging the observations ---
			all_obs.append(np.stack([gen.generate() for gen in all_obs_gen]))
			stacked_obs = np.stack(all_obs, axis=1)
			
			# --- stepping the simulation ---
			teacher_action = teacher(np.expand_dims(obs, axis=0))
			obs_id = int(np.random.random()*len(all_obs_gen))
			student_action = student(stacked_obs)[obs_id, -1]
			if np.random.random() < p_student:
				rand_action = student_action
			else:
				rand_action = teacher_action
			rand_action = rand_action + np.random.normal(size=rand_action.shape) * np.exp(log_std)
			obs, rewards, done, info = env.step(rand_action)
			# done = False
			has_been_done = has_been_done or done
				
			# --- logging the action ---
			all_act.append(teacher_action)
			all_mask.append(done)
		
		if has_been_done:
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
		mask_dataset.append(np.repeat(np.expand_dims(1-np.asarray(all_mask).astype(np.float32), axis=0), len(all_obs_gen), axis=0))
		
		h += 1

	obs_dataset = np.concatenate(obs_dataset, axis=0)
	act_dataset = np.concatenate(act_dataset, axis=0)
	for m in mask_dataset:
		print (m.shape)
	mask_dataset = np.concatenate(mask_dataset, axis=0)
	print(mask_dataset.shape)
	return obs_dataset, act_dataset, mask_dataset

teacher = get_teacher_model(66, 12)
teacher.summary()
teacher.load_weights(os.path.join(config.model_path, "keras_model"))

# env = get_stacked_env(DogEnv, 3)(debug=True)
env = DogEnv(debug=True)
state_config = {"min_delay":15, "max_delay":15}
# env.sim.other_states = [State(state_config) for i in range(10)]
# env.sim.other_states = [State() for i in range(10)]
env.sim.all_states = env.sim.all_states + [State() for i in range(10)]
all_obs_gen = [env.get_other_obs_gen(state) for state in env.sim.all_states]

fake_env = DogEnv(use_realistic_generator=True)

def loss(model, obs, target, mask):
	pred = model(obs)
	custom_loss = tf.reduce_mean(tf.reduce_mean(tf.square(target-pred), axis=-1) * mask)
	return custom_loss

def grad(model, inputs, targets, mask):
	with tf.GradientTape() as tape:
		loss_value = loss(model, inputs, targets, mask)
	return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.Adam()

# student = SimpleActor(fake_env.obs_dim, fake_env.act_dim)
use_temporal_conv = False
student = get_student_model(44, 12, use_temporal_conv=use_temporal_conv) # realist obs
student_optimize = get_student_model(44, 12, use_temporal_conv=use_temporal_conv) # realist obs
student.summary()
# student = get_model(63, 12) # unrealist obs
# student = get_model(56, 12) # realist obs with joint speed
# student.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mean_squared_error", metrics=["mean_squared_error"])
student.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mean_squared_error"])
student_optimize.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mean_squared_error"])
student_optimize.set_weights(student.get_weights())

alpha = 0.9
for i in range(1000):
	print("Step {}:".format(i))
	obs_dataset, act_dataset, mask_dataset = generate_dataset(teacher, student, env, all_obs_gen)
	
	student_optimize.set_weights(student.get_weights())
	for j in range(100):
		loss_value, grads = grad(student_optimize, obs_dataset, act_dataset, mask_dataset)
		print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(), loss_value.numpy()))
		optimizer.apply_gradients(zip(grads, student_optimize.trainable_variables))
	
	
	# student_optimize.fit(obs_dataset, act_dataset, batch_size=30, epochs=100, verbose=2, shuffle=False)
	
	all_weights = []
	for w_optimize, w in zip(student_optimize.get_weights(), student.get_weights()):
		all_weights.append(w*alpha + w_optimize*(1-alpha))
	student.set_weights(all_weights)
	
	
	# --- save logs ---
	student.save_weights(os.path.join(config.student_model_path,"student_{}".format(i)))
	plt.clf()
	plt.plot(all_p_student)
	plt.savefig('student_training.png')
	


