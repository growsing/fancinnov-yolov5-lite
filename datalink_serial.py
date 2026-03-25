# !/usr/bin/env python
# coding=utf-8


# 通过串口（UART）与飞控建立 MAVLink 通信，实现无人机状态获取（位置、姿态、电量等）及指令下发（解锁、起飞、降落、移动等）。
# datalink 类：封装串口操作、MAVLink 消息解析与发送。
# 包含 set_pose、set_arm 等高层控制函数，将本地坐标转换为全局坐标后发送 SET_POSITION_TARGET_LOCAL_NED 指令。
# 使用 mavcrc.py 提供的 CRC 校验。
# 依赖：需要 mavlink.py 和 mavcrc.py

import time
import sys,os
import threading
import serial
import math
import struct, array, time, json
from mavlink import *  # 导入MAVLink模块

class datalink :
	def __init__(self):
		self.com = serial.Serial('/dev/ttyAMA0', 460800, timeout=1.0)   #'/dev/ttyAMA0'为树莓派的串口描述符，460800为串口波特率
		self.f=None
		self.mav_drone=None
		self.message_drone=None
		self.msg_id=None
		self.x=0 # m
		self.y=0
		self.z=0
		self.vx=0 # m/s
		self.vy=0
		self.vz=0
		self.afx=0 # m/ss
		self.afy=0
		self.afz=0
		self.yaw=0 # rad
		self.yaw_rate=0 # rad/s
		self.pos_x=0 # m
		self.pos_y=0
		self.pos_z=0
		self.att_roll=0 # rad
		self.att_pitch=0
		self.att_yaw=0
		self.relative_alt=0 # m
		self.batt_voltage=0 # V
		self.batt_current=0 # A
		self.init_xy=False
		
	def set_arm(self):
		self.mav_drone.set_mode_send(0, MAV_MODE_AUTO_ARMED, 0)
	
	def set_disarm(self):
		self.mav_drone.set_mode_send(0, MAV_MODE_AUTO_DISARMED, 0)
	
	def set_takeoff(self):
		self.mav_drone.command_long_send(0, 0, MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 0)
		
	def set_land(self):
		self.mav_drone.command_long_send(0, 0, MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)

	# dx:m  dy:m  d_alt:m  d_yaw:rad
	def set_pose(self, dx, dy, d_alt, d_yaw):
		if (self.pos_x != 0) and (self.pos_y != 0) and (self.att_yaw != 0): 
			global_dx = dx * math.cos(self.att_yaw) - dy * math.sin(self.att_yaw)
			global_dy = dx * math.sin(self.att_yaw) + dy * math.cos(self.att_yaw)
			self.x = self.pos_x + global_dx
			self.y = self.pos_y + global_dy
			self.z = 0
			self.yaw = self.att_yaw + d_yaw
			self.mav_drone.set_position_target_local_ned_send(0, 0, 0, 0, 0, self.x, self.y, self.z, 0, 0, 0, 0, 0, 0, self.yaw, 0)
	
	# dx:m  dy:m  yaw:rad
	def set_xy_pose(self, dx, dy, yaw):
		if (self.pos_x != 0) and (self.pos_y != 0) and (self.att_yaw != 0): 
			global_dx = dx * math.cos(self.att_yaw) - dy * math.sin(self.att_yaw)
			global_dy = dx * math.sin(self.att_yaw) + dy * math.cos(self.att_yaw)
			self.x = self.pos_x + global_dx
			self.y = self.pos_y + global_dy
			self.z = 0
			self.yaw = yaw
			self.mav_drone.set_position_target_local_ned_send(0, 0, 0, 0, 0, self.x, self.y, self.z, 0, 0, 0, 0, 0, 0, self.yaw, 0)

	# roll:rad  pitch:rad  yaw:rad  alt:m 
	def set_att_alt(self, roll, pitch, yaw, alt):
		self.afx = roll
		self.afy = pitch
		self.yaw = yaw
		self.z = alt
		self.mav_drone.set_position_target_local_ned_send(0, 0, 0, MAV_FRAME_BODY_NED, 0, 0, 0, self.z, 0, 0, 0, self.afx, self.afy, 0, self.yaw, 0)

	class f_drone(object):
		def __init__(self, file):
			self.file=file
			self.buf =[]
		def write(self,data):
			self.file.write(data)

	def drone(self):
		self.f=self.f_drone(self.com)
		self.mav_drone = MAVLink(self.f)
		while True:
			try:
				if self.com.is_open:
					c = self.com.read(1)
				else:
					print("串行端口未打开")
			except:
				pass
				continue
			if c == b'':
				continue
			if ord(c) > -1 :
				try:
					self.message_drone = self.mav_drone.parse_char(c)
					if self.message_drone != None:
						self.msg_id = self.message_drone.get_msgId()
						if self.msg_id == MAVLINK_MSG_ID_HEARTBEAT:
							print('heartbeat\n')
						elif self.msg_id == MAVLINK_MSG_ID_GLOBAL_VISION_POSITION_ESTIMATE:
							if(isinstance(self.message_drone, MAVLink_global_vision_position_estimate_message)):
								self.pos_x=self.message_drone.x/100
								self.pos_y=self.message_drone.y/100
								self.pos_z=self.message_drone.z/100
								self.att_roll=self.message_drone.roll
								self.att_pitch=self.message_drone.pitch
								self.att_yaw=self.message_drone.yaw
								#print('pos:',self.pos_x,'|',self.pos_y,'|',self.pos_z,'\n')
						elif self.msg_id == MAVLINK_MSG_ID_GLOBAL_POSITION_INT:
							if(isinstance(self.message_drone, MAVLink_global_position_int_message)):
								self.relative_alt=float(self.message_drone.relative_alt)/1000
								#print('relative_alt:',self.relative_alt,'\n')
						elif self.msg_id == MAVLINK_MSG_ID_BATTERY_STATUS:
							if(isinstance(self.message_drone, MAVLink_battery_status_message)):
								self.batt_voltage=float(self.message_drone.voltages[1])/1000
								self.batt_current=float(self.message_drone.current_battery)/100
								print('batt:',self.batt_voltage,'V|',self.batt_current,'A\n')
						#else:
							#print('msg:',self.msg_id,'\n')
				except Exception as e:
					self.message_drone = None
					#print("Error parsing message:", e)

	def heartbeat(self):
		while True:
			self.mav_drone.heartbeat_send(MAV_TYPE_CAMERA, MAV_AUTOPILOT_INVALID, MAV_MODE_FLAG_MANUAL_INPUT_ENABLED, 0, MAV_STATE_STANDBY, 1)
			time.sleep(1)

def main():
	print('data link is starting...\n')
	dl=datalink()
	drone_thread=threading.Thread(target=dl.drone)
	try:
		drone_thread.start()
		while True:
			dl.mav_drone.heartbeat_send(MAV_TYPE_CAMERA, MAV_AUTOPILOT_INVALID, MAV_MODE_FLAG_MANUAL_INPUT_ENABLED, 0, MAV_STATE_STANDBY, 1)
			time.sleep(1)
	except Exception as e:
		print("Error sending message:", e)
	finally:
		drone_thread.join()
		dl.com.close()

if __name__ == '__main__':
	main()
