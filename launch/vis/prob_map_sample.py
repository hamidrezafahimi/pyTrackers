from utility import draw_pyramid, get_euler_angles, set_axes_equal
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
data_path = str(Path(__file__).parent.resolve()) + "/../../results/"

v_std_dev = 0.5
beta_std_dev = 10 * np.pi / 180
samples_num = 100

def predict_linear_probs(ts, xs, ys, dt):
    l = len(ts)
    assert (l == len(xs) == len(ys))
    cur_t, last_t = ts[-1], ts[-2]
    cur_x, last_x = xs[-1], xs[-2]
    cur_y, last_y = ys[-1], ys[-2]
    psi = np.arctan2(cur_y-last_y, cur_x-last_x)
    sum_v = 0
    for k in range(l):
        vx = (xs[k] - xs[k-1]) / (ts[k] - ts[k-1])
        vy = (ys[k] - ys[k-1]) / (ts[k] - ts[k-1])
        sum_v += np.sqrt(vx**2 + vy**2)
    avg_v = sum_v / l
    vs = np.random.normal(avg_v, v_std_dev, samples_num) 
    betas = np.random.normal(0, beta_std_dev, samples_num)
    prob_points = []
    plt.axes().set_aspect('equal')
    for k in range(samples_num):
        plt.cla()
        r = vs[k] * dt
        dx = r * np.cos(psi + betas[k]) 
        x = dx + cur_x
        dy = r * np.sin(psi + betas[k])
        y = dy + cur_y
        prob_points.append([x, y])
    return np.array(prob_points)


if __name__ == '__main__':

    target_poses_ned = np.loadtxt(data_path+'park_mavic_1_target_poses.txt', delimiter=',')
    camera_poses_ned = np.loadtxt(data_path+'park_mavic_1_cam_states.txt', delimiter=',')

    num_of_numbers = target_poses_ned.shape[0]

    plt.grid(True)
    
    t0 = target_poses_ned[0, 0]
    tgt_ts = target_poses_ned[:, 0] - t0
    tgt_xs = target_poses_ned[:, 1]
    tgt_ys = target_poses_ned[:, 2]

    window_len = 30
    dt = 0.2
    for i in range(1, num_of_numbers):
        plt.cla()
        if i >= window_len:
            t_window = tgt_ts[i-window_len:i]
            x_window = tgt_xs[i-window_len:i]
            y_window = tgt_ys[i-window_len:i]
            points = predict_linear_probs(t_window, x_window, y_window, dt)
            plt.plot(x_window, y_window)
            plt.scatter(points[:,0], points[:,1])
            plt.pause(0.001)
        else:
            t_window = tgt_ts[:i]
            x_window = tgt_xs[:i]
            y_window = tgt_ys[:i]
        
