
from models.actor import SimpleActor

class SimpleActorNode:
	def __init__ (self):
		pass
	
	def run (self, input_dict, output_dict):
		env = input_dict['Env'][0]
		actor = SimpleActor (env)
		output_dict['Actor'] = actor

from environments import cartpole

class CartPoleNode:
	def __init__ (self):
		pass
	
	def run (self, input_dict, output_dict):
		output_dict['Env'] = cartpole.CartPoleEnv()

from models.actor import MixtureOfExpert

class MixActorNode:
	def __init__ (self):
		pass
	
	def run (self, input_dict, output_dict):
		env = input_dict['Env'][0]
		primitives = input_dict['Primitive']
		actor = MixtureOfExpert (env, primitives)
		output_dict['Actor'] = actor

class TrainPPONode:
	def __init__ (self):
		pass
	
	def run (self, input_dict, output_dict):
		output_dict['Trained actor'] = input_dict['Actor'][0]
		


type_dict = {
		'SimpleActorNode':SimpleActorNode,
		'MixActorNode':MixActorNode,
		'CartPoleNode':CartPoleNode,
		'TrainPPONode':TrainPPONode,
		}

def get_process (type):
	if type not in type_dict:
		raise NameError('Node type {} not known'.format(type))
	
	return type_dict[type]()
