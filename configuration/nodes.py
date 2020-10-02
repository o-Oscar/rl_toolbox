
import os

from models.actor import SimpleActor

class SimpleActorNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, input_dict, output_dict):
		# create the actor
		env = input_dict['Env'][0]
		actor = SimpleActor (env)
		output_dict['Actor'] = actor
		
		# save the actor
		if self.mpi_role == 'main':
			actor.save_path = os.path.join(save_path['models'], self.data['save_name_prop'])
			os.makedirs(actor.save_path)
			actor.save_path += "\{}"
			actor.save(actor.save_path)

from models.actor import MixtureOfExpert

class MixActorNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, input_dict, output_dict):
		# create the actor
		env = input_dict['Env'][0]
		primitives = input_dict['Primitive']
		actor = MixtureOfExpert (env, primitives)
		output_dict['Actor'] = actor

		# save the actor
		if self.mpi_role == 'main':
			actor.save_path = os.path.join(save_path['models'], self.data['save_name_prop'])
			os.makedirs(actor.save_path)
			actor.save_path += "\{}"
			actor.save(actor.save_path)
		
from environments import cartpole

class CartPoleNode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, input_dict, output_dict):
		output_dict['Env'] = cartpole.CartPoleEnv()

class TrainPPONode:
	def __init__ (self, mpi_role):
		self.mpi_role = mpi_role
	
	def run (self, save_path, input_dict, output_dict):
		output_dict['Trained actor'] = input_dict['Actor'][0]
		# to test if we keep going as a worker : data = warehouse.send({"request":["node"]}) ; self.data['name'] == data['node']"


type_dict = {
		'SimpleActorNode':SimpleActorNode,
		'MixActorNode':MixActorNode,
		'CartPoleNode':CartPoleNode,
		'TrainPPONode':TrainPPONode,
		}

def get_process (type):
	if type not in type_dict:
		raise NameError('Node type {} not known'.format(type))
	
	return type_dict[type]
