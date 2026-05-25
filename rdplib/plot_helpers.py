import matplotlib.pyplot as plt
import numpy as np

def plot_angles(Q: np.array,T: np.array,label:str,title:str = '',unit:str = '') -> None:
    fig_q, ax_q = plt.subplots()
    if isinstance(Q, list):
        line_styles = ['-', '--', '-.', ':']
        for j, (q, l) in enumerate(zip(Q,label)):
            for i in range(np.size(q,0)):
                ax_q.plot(T,q[i,:],label=f'{l}_{i+1}', linestyle=line_styles[j % len(line_styles)])
    else:
        for i in range(np.size(Q,0)):
            ax_q.plot(T,Q[i,:], label=f'{label}_{i+1}')
    plt.title(title)
    plt.legend(loc="upper left")
    ax_q.set_xlabel("t in s", color='b', weight='normal')
    ax_q.set_ylabel(f"{label}_i in {unit}", color='b', weight='normal')

def plot_path_3d(W, title: str = ''):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    if isinstance(W, list):
        for w in W:
            ax.plot(w[0,:], w[1,:], w[2,:])
    else:
        ax.plot(W[0,:], W[1,:], W[2,:])
    plt.title(title)
    ax.set_xlabel('x in m', color='b', weight='normal')
    ax.set_ylabel('y in m', color='b', weight='normal')
    ax.set_zlabel('z in m', color='b', weight='normal')

def plot_cartesian_trajectory(W, T: np.array, label: str, title:str = '', unit:str = '') -> None:
    fig, ax = plt.subplots()
    label_list = ['x','y','z']
    if isinstance(W, list):
        line_styles = ['-', '--', '-.', ':']
        for j, (w, l) in enumerate(zip(W, label)):
            for i in range(np.size(w,0)):
                ax.plot(T, w[i,:], label=f'{l}_{label_list[i]}', linestyle=line_styles[j % len(line_styles)])
    else:
        for i in range(np.size(W,0)):
            ax.plot(T,W[i,:], label=f'{label}_{label_list[i]}')
    plt.title(title)
    plt.legend(loc='upper left')
    ax.set_xlabel("t in s", color='b', weight='normal')
    ax.set_ylabel(f'{label}_i in {unit}', color='b', weight='normal')

def plot_trajectory(W, T: np.array, label: str, title:str = '', unit:str='') -> None:
    fig,ax = plt.subplots()
    if isinstance(W, list):
        for w, l in zip(W,label):
            ax.plot(T, w, label=l)
        plt.legend(loc='upper left')
    else:
        ax.plot(T,W, label=label)
        ax.set_ylabel(f'{label} in {unit}', color='b', weight='normal')
    plt.title(title)
    ax.set_xlabel("t in s", color='b', weight='normal')

def plot_cartesian_trajectory_with_points(W: np.array, W_stuetz:np.array, title: str = 'Path in Workspace'):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_box_aspect((1,1,1))
    ax.plot(W[0,:], W[1,:], W[2,:], label='Bahn des TCP', color='blue')
    ax.scatter(W_stuetz[:,0], W_stuetz[:,1], W_stuetz[:,2], label='Stützpunkte', color='red')
    ax.plot(W_stuetz[:,0], W_stuetz[:,1], W_stuetz[:,2], color='gray', linestyle=':')
    plt.title(title)
    plt.legend(loc="upper left")
    ax.set_xlabel('x in m', color='b', weight='normal')
    ax.set_ylabel('y in m', color='b', weight='normal')
    ax.set_zlabel('z in m', color='b', weight='normal')

def plot_rot_speed_torque(rot_speed: np.array, torque: np.array, title: str='torque-speed plot') -> None:
    fig, ax = plt.subplots()
    plt.title(title)
    for i in range(np.size(rot_speed,0)):
        #ax.plot(np.abs(rot_speed[i,:]), np.abs(torque[i,:]),label=f'Achse_{i+1}')
        ax.plot(rot_speed[i,:], torque[i,:], label=f'axis {i+1}')
    ax.set_xlabel('n in 1/min', color='b', weight='normal')
    ax.set_ylabel('M in Nm', color='b', weight='normal')
    plt.legend(loc='upper right')
