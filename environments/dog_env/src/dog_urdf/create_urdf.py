import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import numpy as np

"""
conda activate psc_sb
cd C:\\Users\\oscbo\\Documents\\Travail\\PSC\\idefX\\v3\\rl_toolbox
conda activate psc_sb
python environments\\dog_env_rai\\src\\dog_urdf\\create_urdf.py

"""

def add_colors (robot):
	all_colors = [	("black", "0.0 0.0 0.0 1.0"),
					("blue", "0.0 0.0 0.8 1.0"),
					("green", "0.0 0.8 0.0 1.0"),
					("grey", "0.2 0.2 0.2 1.0"),
					("silver", "0.913725490196 0.913725490196 0.847058823529 1.0"),
					("orange", "1.0 0.423529411765 0.0392156862745 1.0"),
					("brown", "0.870588235294 0.811764705882 0.764705882353 1.0"),
					("red", "0.8 0.0 0.0 1.0"),
					("white", "1.0 1.0 1.0 1.0")]
	for name, rgba in all_colors:
		material = ET.SubElement(robot, 'material')
		material.set("name", name)
		
		color = ET.SubElement(material, 'color')
		color.set("rgba", rgba)

def add_origin (root, pos, rot):
	origin = ET.SubElement(root, 'origin')
	origin.set("xyz", pos)
	origin.set("rpy", rot)

def add_inertia (root, mass, inertia):

	mass_xml = ET.SubElement(root, 'mass')
	mass_xml.set("value", mass)
	
	inertia_xml = ET.SubElement(root, 'inertia')
	for name, value in inertia.items(): # zip([ixx, ixy, ixz, iyy, iyz, izz], inertia):
		inertia_xml.set(name, str(value))

def add_contact_material (collision, material_name):
	
	material = ET.SubElement(collision, 'material')
	material.set("name", "")
	contact = ET.SubElement(material, 'contact')
	contact.set("name", material_name)

def add_box (root, size):
	geometry = ET.SubElement(root, 'geometry')
	box = ET.SubElement(geometry, 'box')
	box.set("size", size)

def add_box_link (robot, name, pos, rot, size, mass, inertia, color, contact_material):
	
	link = ET.SubElement(robot, 'link')
	link.set("name", name)
	
	visual = ET.SubElement(link, 'visual')
	add_origin(visual, pos, rot)
	add_box(visual, size)
	material = ET.SubElement(visual, 'material')
	material.set("name", color)
	
	collision = ET.SubElement(link, 'collision')
	add_origin(collision, pos, rot)
	add_box(collision, size)
	add_contact_material (collision, contact_material)
	
	inertial = ET.SubElement(link, 'inertial')
	add_origin(inertial, pos, rot)
	add_inertia(inertial, mass, inertia)

def add_box_link_wrapper (robot, name, pos, rot, size, mass, contact_material="default_material", color=None):
	#volume = size[0]*size[1]*size[2]
	#density = mass/volume
	
	inertia_str = {	"ixx": mass / 12 * (size[1]**2 + size[2]**2),
					"ixy":0,
					"ixz":0,
					"iyy": mass / 12 * (size[0]**2 + size[2]**2),
					"iyz":0,
					"izz": mass / 12 * (size[0]**2 + size[1]**2)}
	
	
	pos_str = " ".join((str(x) for x in pos))
	rot_str = " ".join((str(x) for x in rot))
	size_str = " ".join((str(x) for x in size))
	mass_str = str(mass)
	color_str = "grey" if color is None else color
	
	add_box_link (robot, name, pos_str, rot_str, size_str, mass_str, inertia_str, color_str, contact_material)

def add_cylinder (section, l, r):
	geometry = ET.SubElement(section, 'geometry')
	cylinder = ET.SubElement(geometry, 'cylinder')
	cylinder.set("length", str(l))
	cylinder.set("radius", str(r))
	
def add_cylinder_link (robot, name, pos, rot, l, r, mass, inertia, color, contact_material):
	
	link = ET.SubElement(robot, 'link')
	link.set("name", name)
	
	visual = ET.SubElement(link, 'visual')
	add_origin(visual, pos, rot)
	add_cylinder(visual, l, r)
	material = ET.SubElement(visual, 'material')
	material.set("name", color)
	
	collision = ET.SubElement(link, 'collision')
	add_origin(collision, pos, rot)
	add_cylinder(collision, l, r)
	add_contact_material (collision, contact_material)
	
	inertial = ET.SubElement(link, 'inertial')
	add_origin(inertial, pos, rot)
	add_inertia(inertial, mass, inertia)

def add_cylinder_link_wrapper (robot, name, pos, rot, l, r, mass, contact_material="default_material", color=None):
	#volume = size[0]*size[1]*size[2]
	#density = mass/volume
	
	inertia_str = {	"ixx":1/12*mass*(3*r*r+l*l),
					"ixy":0,
					"ixz":0,
					"iyy":1/12*mass*(3*r*r+l*l),
					"iyz":0,
					"izz":1/2*mass*r*r}
	
	l_str = str(l)
	r_str = str(r)
	pos_str = " ".join((str(x) for x in pos))
	rot_str = " ".join((str(x) for x in rot))
	mass_str = str(mass)
	color_str = "grey" if color is None else color
	
	add_cylinder_link (robot, name, pos_str, rot_str, l_str, r_str, mass_str, inertia_str, color_str, contact_material)

	
"""
def add_foot (section):
	geometry = ET.SubElement(section, 'geometry')
	mesh = ET.SubElement(geometry, 'mesh')
	mesh.set("filename", "untitled.obj")
	mesh.set("scale", ".1 .1 .1")

def add_dummy_link (robot, name, contact_material="default_material"):
	
	link = ET.SubElement(robot, 'link')
	link.set("name", name)
	
	l = 0.02
	r = 0.02
	m = 0.01
	
	inertial = ET.SubElement(link, 'inertial')
	add_origin(inertial, "0 0 0", "0 0 0")
	add_inertia(inertial, "0", {	"ixx":1/12*m*(3*r*r+l*l),
								"ixy":0,
								"ixz":0,
								"iyy":1/12*m*(3*r*r+l*l),
								"iyz":0,
								"izz":1/2*m*r*r})
	
	visual = ET.SubElement(link, 'visual')
	add_origin(visual, "0 0 0", "0 0 0")
	add_cylinder(visual, l, r)
	material = ET.SubElement(visual, 'material')
	material.set("name", "black")
	
	collision = ET.SubElement(link, 'collision')
	add_origin(collision, "0 0 0", "0 0 0")
	add_cylinder(collision, l, r)
	add_contact_material (collision, contact_material)#"robot_material")
"""
	
	
	
	
	
	
def add_joint (robot, name, parent_name, child_name, type, pos, rot, axis, max_torque):
	
	joint = ET.SubElement(robot, 'joint')
	joint.set("name", name)
	joint.set("type", type)
	
	
	origin = ET.SubElement(joint, 'origin')
	origin.set("rpy",rot)
	origin.set("xyz", pos)
	
	parent = ET.SubElement(joint, 'parent')
	parent.set("link", parent_name)

	child = ET.SubElement(joint, 'child')
	child.set("link", child_name)

	axis_xml = ET.SubElement(joint, 'axis')
	axis_xml.set("xyz", axis)

	dynamics = ET.SubElement(joint, 'dynamics')
	dynamics.set("damping", "0")
	dynamics.set("friction", "0")
	
	limit = ET.SubElement(joint, 'limit')
	limit.set("effort", max_torque)

def add_joint_wrapper (robot, parent_name, child_name, pos, rot, axis, max_troque, type="revolute"):
	name = child_name + "_joint"
	# type = "revolute"
	pos_str = " ".join((str(x) for x in pos))
	rot_str = " ".join((str(x) for x in rot))
	axis_str = " ".join((str(x) for x in axis))
	max_torque_strs = str(max_troque)
	add_joint (robot, name, parent_name, child_name, type, pos_str, rot_str, axis_str, max_torque_strs)
"""
def create_standard_dog ():
	robot = ET.Element('robot')
	robot.set("name", "IdefX")
	add_colors(robot)

	add_link_wrapper(robot, "trunk", (0, 0, 0), (0, 0, 0), (0.5, 0.2, 0.2), 13)
	for strid, facx, facy, color in [("FL", 1, 1, "blue"), ("FR", 1, -1, "green"), ("BL", -1, 1, "orange"), ("BR", -1, -1, "red")]:
		
		leg_material = "{}_leg_material".format(strid)
		
		clavicle_name = strid+"_clavicle"
		add_joint_wrapper (robot, "trunk", clavicle_name, (facx*0.25, facy*0.15, 0), (0, 0, 0), (1, 0, 0))
		add_link_wrapper(robot, clavicle_name, (0, 0, 0), (0, 0, 0), (0.1, 0.04, 0.1), 0.6, contact_material=leg_material)
		
		arm_name = strid+"_arm"
		add_joint_wrapper (robot, clavicle_name, arm_name, (0, facy*0.05, 0), (0, 0, 0), (0, 1, 0))
		add_link_wrapper(robot, arm_name, (0, 0, -0.115), (0, 0, 0), (0.04, 0.02, 0.23), 0.6, contact_material=leg_material)
		
		forearm_name = strid+"_forearm"
		add_joint_wrapper (robot, arm_name, forearm_name, (0, facy*0.03, -0.2), (0, 0, 0), (0, 1, 0))
		add_link_wrapper(robot, forearm_name, (0, 0, -0.1), (0, 0, 0), (0.04, 0.02, 0.25), 0.6, contact_material=leg_material)

		foot_name = strid+"_foot"
		add_joint_wrapper (robot, forearm_name, foot_name, (0, facy*0.01, -0.25), (0, 0, 0), (0, 1, 0), type="fixed")
		add_dummy_link(robot, foot_name, contact_material=leg_material)

def create_IdefX_v1 ():
	robot = ET.Element('robot')
	robot.set("name", "IdefX")
	add_colors(robot)

	# add_link_wrapper(robot, "trunk", (0.03, 0, 0), (0, 0, 0), (0.34, 0.26, 0.2), 8.8)
	add_link_wrapper(robot, "trunk", (0, 0, 0), (0, 0, 0), (0.34, 0.26, 0.2), 8.8)
	for strid, facx, facy, color in [("FL", 1, 1, "blue"), ("FR", 1, -1, "green"), ("BL", -1, 1, "orange"), ("BR", -1, -1, "red")]:
		
		leg_material = "{}_leg_material".format(strid)
		
		max_troque = 11
		
		clavicle_name = strid+"_clavicle"
		add_joint_wrapper (robot, "trunk", clavicle_name, (facx*0.16, facy*0.08, -0.07), (0, 0, 0), (1, 0, 0), max_troque)
		add_link_wrapper(robot, clavicle_name, (0, facy*0.03, 0), (0, 0, 0), (0.1, 0.04, 0.1), 1.2, contact_material=leg_material)
		
		arm_name = strid+"_arm"
		add_joint_wrapper (robot, clavicle_name, arm_name, (0, facy*0.07, 0), (0, 0, 0), (0, 1, 0), max_troque)
		l1 = 0.196
		knee_margin = 0.04
		add_link_wrapper(robot, arm_name, (0, 0, -l1/2 - knee_margin), (0, 0, 0), (0.04, 0.04, l1+knee_margin), 0.1, contact_material=leg_material)
		
		forearm_name = strid+"_forearm"
		add_joint_wrapper (robot, arm_name, forearm_name, (0, facy*0.01, -l1), (0, 0, 0), (0, 1, 0), max_troque*24/30)
		l2 = 0.200
		lfoot = 0 # 0.02
		add_link_wrapper(robot, forearm_name, (0, 0, -(l2+lfoot)/2), (0, 0, 0), (0.04, 0.02, (l2+lfoot)), 0.1, contact_material=leg_material)

		foot_name = strid+"_foot"
		add_joint_wrapper (robot, forearm_name, foot_name, (0, 0, -l2), (1.5707, 0, 0), (0, 1, 0), 1000, type="fixed")
		add_dummy_link(robot, foot_name, contact_material=leg_material)
	return robot
"""
def create_IdefX_v2 ():
	robot = ET.Element('robot')
	robot.set("name", "IdefX")
	add_colors(robot)
	
	motor_m = 0.7
	m = 16.7 - 12*motor_m - 4*0.2
	
	root_name = "trunk"
	add_box_link_wrapper(robot, root_name, (0, 0, 0), (0, 0, 0), (0.245, 0.17, 0.093), m*2/3)
	add_joint_wrapper (robot, root_name, root_name+"_appendix0", (0, 0, 0), (0, 0, 0), (0, 0, 0), 1000, type="fixed")
	add_box_link_wrapper(robot, root_name+"_appendix0", (0, 0, 0), (0, 0, 0), (0.578, 0.135, 0.093), m*1/3)
	
	for strid, facx, facy, color in [("FL", 1, 1, "blue"), ("FR", 1, -1, "green"), ("BL", -1, 1, "orange"), ("BR", -1, -1, "red")]:
		
		leg_material = "{}_leg_material".format(strid)
		
		max_troque = 11
		
		first_motor_name = strid+"_first_motor"
		add_joint_wrapper (robot, root_name, first_motor_name, (facx*(0.34-0.035)/2, facy*0.075, 0.), (0, 0, 0), (0, 0, 0), 1000, type="fixed")
		add_cylinder_link_wrapper(robot, first_motor_name, (0, 0, 0), (0, np.pi/2, 0), 0.035, 0.05, motor_m, contact_material=leg_material)
		
		
		clavicle_name = strid+"_clavicle"
		add_joint_wrapper (robot, root_name, clavicle_name, (facx*0.45/2, facy*0.075, -0.0), (0, 0, 0), (1, 0, 0), max_troque)
		add_cylinder_link_wrapper(robot, clavicle_name, (0, facy*(0.05-0.035/2), 0), (np.pi/2, 0, 0), 0.035, 0.05, motor_m, contact_material=leg_material)
		
		
		arm_name = strid+"_arm"
		add_joint_wrapper (robot, clavicle_name, arm_name, (0, facy*(0.05+0.04/2), 0), (0, 0, 0), (0, 1, 0), max_troque)
		l1 = 0.196
		knee_margin = 0.04
		add_box_link_wrapper(robot, arm_name, (0, 0, -l1/2 - knee_margin), (0, 0, 0), (0.04, 0.04, l1+knee_margin), 0.1, contact_material=leg_material)
		
		
		third_motor_name = strid+"_third_motor"
		add_joint_wrapper (robot, arm_name, third_motor_name, (0., facy*(0.035+0.04)/2, 0.), (0, 0, 0), (0, 0, 0), 1000, type="fixed")
		add_cylinder_link_wrapper(robot, third_motor_name, (0, 0, 0), (np.pi/2, 0, 0), 0.035, 0.05, motor_m, contact_material=leg_material)
		
		
		forearm_name = strid+"_forearm"
		add_joint_wrapper (robot, arm_name, forearm_name, (0, facy*0.01, -l1), (0, 0, 0), (0, 1, 0), max_troque*24/30)
		l2 = 0.200
		lfoot = 0 # 0.02
		add_box_link_wrapper(robot, forearm_name, (0, 0, -(l2+lfoot)/2), (0, 0, 0), (0.04, 0.02, (l2+lfoot)), 0.1, contact_material=leg_material)


		foot_name = strid+"_foot"
		add_joint_wrapper (robot, forearm_name, foot_name, (0, 0, -l2), (0, 0, 0), (0, 1, 0), 1000, type="fixed")
		add_cylinder_link_wrapper(robot, foot_name, (0, 0, 0), (np.pi/2, 0, 0), 0.02, 0.02, 0.01, contact_material=leg_material)
		
	return robot
	
# robot = create_IdefX_v1()
robot = create_IdefX_v2()

with open(str(Path(__file__).parent) + "/main_test.urdf", "w") as f:
	xmlstr = minidom.parseString(ET.tostring(robot)).toprettyxml(indent="\t")
	f.write(xmlstr)