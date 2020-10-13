import numpy as np
import time
import matplotlib.pyplot as plt

MEAN_PLOT = 0
INDIVIDUAL_PLOT = 1

plot_type = MEAN_PLOT

adr_names = ["exp_1"]

if __name__ == "__main__":
	all_names = []
	all_values = []
	for adr_name in adr_names:
		path = "adr/rank_"+adr_name
		names = []
		values = np.load(path+".npy")
		with open(path+".txt") as f:
			text = f.readlines()
			names = text[0].split("$")[:-1]
		all_values.append(values)
		all_names.append(names)
	
	if plot_type == INDIVIDUAL_PLOT:
		indiv_values = np.concatenate(all_values, axis=0)
		labels = sum([[name+str(i) for name in names] for i, names in enumerate(all_names)], [])
		for label, value in zip(labels, indiv_values):
			plt.plot(value, label=label)
		
		plt.legend()
		plt.show()
	elif plot_type == MEAN_PLOT:
		sum_dict = {}
		len_dict = {}
		for i, names in enumerate(all_names):
			for j, name in enumerate(names):
				if name in sum_dict.keys():
					sum_dict[name] += all_values[i][j]
					len_dict[name] += 1
				else:
					sum_dict[name] = all_values[i][j]
					len_dict[name] = 1
					
		for name in sum_dict.keys():
			plt.plot(sum_dict[name]/len_dict[name], label=name)
		
		plt.legend()
		plt.show()
		
		