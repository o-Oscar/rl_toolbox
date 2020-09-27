import numpy as np
import matplotlib as plt
import odrive
from odrive.enums import *
import time

def find_odrive ():
	print("looking for an unpaired odrive...")
	try:
		odrv = odrive.find_any(timeout=15) # takes 8s to find.
		print("Odrive was found")
		return odrv
	except TimeoutError:
		print("No device found")

def set_odrive_config (odrv, reboot=False):
	my_drive.axis1.controller.config.pos_gain = 60
	my_drive.axis1.controller.config.vel_gain = 1e-5
	my_drive.axis1.controller.config.vel_integrator_gain = 0 #4e-5

	my_drive.axis1.trap_traj.config.accel_limit = 50000*100 # 5000
	my_drive.axis1.trap_traj.config.decel_limit = my_drive.axis1.trap_traj.config.accel_limit

	my_drive.axis1.controller.config.vel_limit = 1000000.0
	my_drive.axis1.trap_traj.config.vel_limit = 200000.0 # 20000

	my_drive.axis1.encoder.config.enable_phase_interpolation = True

	my_drive.axis1.controller.config.input_mode = 5
	
	odrv.axis0.encoder.config.mode = ENCODER_MODE_INCREMENTAL
	odrv.axis0.encoder.config.cpr = 2400
	
	odrv.axis1.encoder.config.mode = ENCODER_MODE_INCREMENTAL
	odrv.axis1.encoder.config.cpr = 2400
	
	if reboot:
		my_drive.save_configuration()
		my_drive.reboot()
		time.sleep(2)

def start_odrive (odrv):
	odrv.axis1.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE


if __name__ == "__main__":
	odrv = find_odrive()
	if odrv is not None:
		start_odrive (odrv)