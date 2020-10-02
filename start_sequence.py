import xml.etree.ElementTree as ET

import configuration.nodes


class ProcessNode:
	def __init__ (self, data):
		self.input_dict = {}
		self.output_dict = {}
		self.input_links = []
		self.output_links = []
		self.data = data
		
		self.proc = configuration.nodes.get_process(data['type'])
		self.proc.data = data
		
	def run (self):
		#self.data.update(self.input_dict) # do we really want to do that ?
		
		#TODO : run the actual computation and update the self.output_dict
		"""
		print("Running {} \t With type {}".format(self.data['name'], self.data['type']))
		print("Args : " + str(self.input_dict))
		"""
		self.proc.run(self.input_dict, self.output_dict)
		"""
		for link in self.output_links:
			self.output_dict[link['from_socket']] = 42
		"""

DEBUG = True

if __name__ == "__main__":
	config_file_name = "configuration/config.xml"
	tree = ET.parse(config_file_name)
	root = tree.getroot()
	
	# reading the xml
	proc_dict = {}
	
	for child in root.iter('Process'):
		attrib = child.attrib
		proc_dict[attrib['name']] = ProcessNode(attrib)
	
	for child in root.iter('Link'):
		attrib = child.attrib
		proc_dict[attrib['from_node']].output_links.append(attrib)
		proc_dict[attrib['to_node']].input_links.append(attrib)
	
	if DEBUG:
		type_outputs = {}
		for node in proc_dict.values():
			if not node.data['type'] in type_outputs:
				type_outputs[node.data['type']] = set()
			for link in node.output_links:
				type_outputs[node.data['type']].add(link['from_socket'])
		print("Expected node outputs : ")
		print(type_outputs)
	
	# calculating the execution order
	requirement_nb = {}
	dest_nb = {}
	target_nodes = []
	for key in proc_dict.keys():
		requirement_nb[key] = len(proc_dict[key].input_links)
		dest_nb[key] = len(proc_dict[key].output_links)
		if len(proc_dict[key].output_links) == 0:
			target_nodes.append(key)
	
	proc_stack = []
	while len(target_nodes) > 0:
		if requirement_nb[target_nodes[-1]] == 0:
			to_remove = target_nodes.pop()
			proc_stack.append(to_remove)
			for link in proc_dict[to_remove].output_links:
				requirement_nb[link['to_node']] -= 1
		else:
			for link in proc_dict[target_nodes[-1]].input_links:
				if dest_nb[link['from_node']] == 1:
					target_nodes.append(link['from_node'])
					dest_nb[link['from_node']] -= 1
				else:
					dest_nb[link['from_node']] -= 1
	
	# executing the stacked processes
	for proc in proc_stack:
		proc_dict[proc].run()
		
		for link in proc_dict[proc].output_links:
			if link['to_socket'] in proc_dict[link['to_node']].input_dict:
				proc_dict[link['to_node']].input_dict[link['to_socket']].append(proc_dict[proc].output_dict[link['from_socket']])
			else:
				proc_dict[link['to_node']].input_dict[link['to_socket']] = [proc_dict[proc].output_dict[link['from_socket']]]
			
	
	