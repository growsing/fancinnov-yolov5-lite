#!/usr/bin/env python
# coding=utf-8

import serial
import time
import threading
import math
from pymavlink import mavutil

class datalink:
    def __init__(self, device='/dev/ttyAMA0', baud=460800):
        self.device = device
        self.baud = baud
        self.master = None
        self.connected = False
        
        # 状态变量
        self.relative_alt = 0.0
        self.batt_voltage = 0.0
        self.batt_current = 0.0
        self.pos_x = 0.0   # 局部NED坐标系X（北）
        self.pos_y = 0.0   # 局部NED坐标系Y（东）
        self.pos_z = 0.0   # 高度（负值表示向上）
        self.att_roll = 0.0
        self.att_pitch = 0.0
        self.att_yaw = 0.0
        
        # 连接
        self._connect()
    
    def _connect(self):
        # 建立MAVLink连接并等待心跳
        print(f"Connecting to {self.device} at {self.baud}...")
        self.master = mavutil.mavlink_connection(self.device, baud=self.baud)
        print("Waiting for heartbeat...")
        self.master.wait_heartbeat()
        print(f"Heartbeat received from system {self.master.target_system}")
        self.connected = True
        
        # 请求数据流（10Hz）
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            10, 1
        )
    
    def _update_state(self):
        # 在后台线程中持续读取消息并更新状态
        while True:
            if self.master:
                msg = self.master.recv_match(blocking=True, timeout=0.5)
                if msg:
                    msg_type = msg.get_type()
                    if msg_type == 'GLOBAL_POSITION_INT':
                        self.relative_alt = msg.relative_alt / 1000.0
                        # 可选：将经纬度转换为局部坐标，这里简化
                    elif msg_type == 'ATTITUDE':
                        self.att_roll = msg.roll
                        self.att_pitch = msg.pitch
                        self.att_yaw = msg.yaw
                    elif msg_type == 'BATTERY_STATUS':
                        if msg.voltages[0] != 65535:  # 有效电压
                            self.batt_voltage = msg.voltages[0] / 1000.0
                        self.batt_current = msg.current_battery / 100.0
                    elif msg_type == 'LOCAL_POSITION_NED':
                        self.pos_x = msg.x
                        self.pos_y = msg.y
                        self.pos_z = msg.z
            time.sleep(0.01)
    
    def start(self):
        # 启动后台接收线程
        self.thread = threading.Thread(target=self._update_state, daemon=True)
        self.thread.start()
    
    # ========== 控制接口 ==========
    def set_arm(self):
        print("ARM command sent")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1, 0, 0, 0, 0, 0, 0)
    
    def set_disarm(self):
        print("DISARM command sent")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            0, 0, 0, 0, 0, 0, 0)
    
    def set_takeoff(self, altitude=2.0):
        print(f"TAKEOFF to {altitude}m")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, altitude)
    
    def set_land(self):
        print("LAND command sent")
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND, 0,
            0, 0, 0, 0, 0, 0, 0)
    
    def set_pose(self, dx, dy, dz, dyaw):
        
        # 相对当前机头方向移动 dx, dy，改变高度 dz（米），偏航变化 dyaw（弧度）
        # 注意：这里简化为发送 SET_POSITION_TARGET_LOCAL_NED 消息
     
        # 获取当前姿态偏航
        yaw = self.att_yaw
        # 将机体系偏移转换到NED系
        global_dx = dx * math.cos(yaw) - dy * math.sin(yaw)
        global_dy = dx * math.sin(yaw) + dy * math.cos(yaw)
        
        # 目标位置（NED坐标系，Z轴向下，所以向上移动需要负值）
        target_x = self.pos_x + global_dx
        target_y = self.pos_y + global_dy
        target_z = self.pos_z - dz   # 因为dz为正表示向上
        
        # 目标偏航
        target_yaw = yaw + dyaw
        
        # 发送位置控制指令
        self.master.mav.set_position_target_local_ned_send(
            0, self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000,  # 位置、偏航有效，速度、加速度忽略
            target_x, target_y, target_z,
            0, 0, 0, 0, 0, 0, target_yaw, 0
        )
    
    # 为兼容原代码，保留 drone 和 heartbeat 方法（空或调用 start）
    def drone(self):
        # 兼容旧接口，实际启动接收线程
        self.start()
        # 保持线程运行
        while True:
            time.sleep(1)
    
    def heartbeat(self):
        # pymavlink 会自动发送心跳，无需额外实现
        pass

# 测试用
if __name__ == '__main__':
    dl = datalink()
    dl.start()
    print("State monitor started. Press Ctrl+C to exit.")
    try:
        while True:
            print(f"Alt: {dl.relative_alt:.2f}m | Batt: {dl.batt_voltage:.1f}V {dl.batt_current:.1f}A | "
                  f"Pos: ({dl.pos_x:.1f}, {dl.pos_y:.1f}, {dl.pos_z:.1f}) | Yaw: {dl.att_yaw:.2f}rad")
            time.sleep(2)
    except KeyboardInterrupt:
        print("Exiting")