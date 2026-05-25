import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import rdplib.robot as rb
import rdplib.plot_helpers as ph

# Python 3.10 is the minimum version required for this project
assert sys.version_info >= (3, 10)


## create robot
robot = rb.robot_factory(os.path.join('robot_config_files','ABB_GoFa.yaml'))

## calculate trajectory
q_start = np.copy(robot.q_home) # start: home-position of the robot
q_end = q_start + np.pi/4 * np.ones((robot.size)) # end: all joints move +45 degrees
q = np.copy(q_start)
dot_q = np.zeros((robot.size))
ddot_q = np.zeros((robot.size))

dt = 0.001 # time step
T_ges = 4.0 # total time of the movement
T_joint = T_ges/robot.size # approximate time for the movement of one joint
N_joint = int(T_joint/dt) # steps for the movement of one joint
numsteps = robot.size*N_joint # total number of steps
deltaT = (N_joint-1)*dt # real time for the movement of one joint

# initialize buffer for visualization
q_traj = np.zeros((robot.size, numsteps))
dot_q_traj = np.zeros((robot.size, numsteps))
ddot_q_traj = np.zeros((robot.size, numsteps))
r_TCP_traj = np.zeros((3, numsteps))
v_TCP_traj = np.zeros((3,numsteps))
a_TCP_traj = np.zeros((3,numsteps))
T = dt*np.linspace(start=0,stop=numsteps-1,num=numsteps) # time array
data = np.zeros((numsteps,12*robot.size+1)) # data for the viewer

# loop over joints, rotate one after the other
for i in range(0,robot.size):
    
    # polynomial coefficients
    a = 2/pow(deltaT,3) * (q_start[i]-q_end[i])
    b = 3/pow(deltaT,2) * (q_end[i]-q_start[i])
    # reset angular acceleration
    ddot_q = np.zeros((robot.size))

    # loop over time steps per joint
    for k in range(0,N_joint):

        # polynomial interpolation in joint space
        q[i] = a*pow(k*dt,3) + b*pow(k*dt,2) + q_start[i]
        dot_q[i] = 3*a*pow(k*dt,2) + 2*b*k*dt
        ddot_q[i] = 6*a*k*dt + 2*b

        # direct kinematics
        # Exercise 0
        robot.calculate_positions(q)
        # Exercise 1
        robot.calculate_velocities(dot_q)
        robot.calculate_accelerations(ddot_q)
        robot.calculate_jacobis()

        # add data to trajectory variables
        q_traj[:,i*N_joint+k] = q
        dot_q_traj[:,i*N_joint+k] = dot_q
        ddot_q_traj[:,i*N_joint+k] = ddot_q
        r_TCP_traj[:,i*N_joint+k] = robot.r_TCP__0
        v_TCP_traj[:,i*N_joint+k] = robot.v_TCP__0
        a_TCP_traj[:,i*N_joint+k] = robot.a_TCP__0

        # build array for viewer
        for l in range(robot.size):
            data[i*N_joint+k,0] = T[i*N_joint+k]
            data[i*N_joint+k,l*12+1:l*12+4] = np.transpose(robot.links[l].r_i__0)
            data[i*N_joint+k,l*12+4:l*12+13] = np.reshape(np.transpose(robot.links[l].A_i0),(1,9))

# save data for viewer
np.savetxt('trajectory.csv', data, delimiter="\t", fmt='%.4f')

# visualization with matplotlib
# Exercise 0
ph.plot_path_3d(r_TCP_traj, 'Path in the Workspace')
# Exercise 1
ph.plot_cartesian_trajectory(v_TCP_traj, T, 'v_TCP', 'Velocity of the TCP', 'm/s')
ph.plot_cartesian_trajectory(a_TCP_traj, T, 'a_TCP', 'Acceleration of the TCP', 'm/s²')
plt.show()