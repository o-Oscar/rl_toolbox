
class SimpleActorNode:
	def __init__ (self):
		pass
	
	def run (self, input_dict, output_dict):
		output_dict['Actor'] = 42

class CartPoleNode:
	def __init__ (self):
		pass
	
	def run (self, input_dict, output_dict):
		output_dict['Env'] = 42

class MixActorNode:
	def __init__ (self):
		pass
	
	def run (self, input_dict, output_dict):
		output_dict['Actor'] = 42

class TrainPPONode:
	def __init__ (self):
		pass
	
	def run (self, input_dict, output_dict):
		output_dict['Trained actor'] = 42
		


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
