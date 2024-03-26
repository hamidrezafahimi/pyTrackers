import numpy as np
from matplotlib import pyplot as plt

ts = [0., 0.03399992, 0.06900001, 0.56599998, 0.61599994, 0.648, 0.71499991, 0.76900005, 0.81500006, 0.84800005, 0.8829999, 1.59699988, 1.64899993, 1.73199987, 1.76600003, 1.79799986, 1.84899998, 1.96700001, 2.06800008, 2.1329999, 2.18199992, 2.21700001, 2.26600003, 2.33299994, 2.41700006, 2.48799992, 2.51699996, 2.56500006, 2.60100007, 2.66700006] 

xs = [-16.19982356, -16.12264637, -16.04319888, -14.99624723, -14.91805547, -14.8680126, -14.76323568, -14.67878828, -14.60685179, -14.55524519, -14.50051115, -13.37628137, -13.29223699, -13.15808942, -13.10313712, -13.05141782, -12.96898955, -12.77827356, -12.61503353, -12.50997844, -12.43078279, -12.37421437, -12.29501872, -12.186731,   -12.05096693, -11.96104219, -11.93363097, -11.88826065, -11.85423297, -11.79184893]

ys = [ 5.38440681, 5.48590333,  5.59038555,  7.08826878,  7.24371967,  7.34320853,  7.55151264,  7.71940018,  7.86241514,  7.9650128,   8.07382805, 10.34453703, 10.52432375, 10.81129057, 10.92884387, 11.03948115, 11.21581069, 11.62378799, 11.97298906, 12.19772163, 12.36713598, 12.48814646, 12.65756081, 12.88920857, 13.1796334,  13.40809669, 13.49502706, 13.63891189, 13.74682533, 13.94466651]

v_std_dev = 0.5
beta_std_dev = 10 * np.pi / 180
samples_num = 10


def predict_linear_probs(ts, xs, ys, dt):
    l = len(ts)
    assert (l == len(xs) == len(ys))
    cur_t, last_t = ts[-1], ts[-2]
    cur_x, last_x = xs[-1], xs[-2]
    cur_y, last_y = ys[-1], ys[-2]
    #psi = np.arctan2(cur_x-last_x, cur_y-last_y)
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
        r = vs[k] * dt
        #dx = r * np.cos(psi) 
        dx = r * np.cos(psi + betas[k]) 
        x = dx + cur_x
        dy = r * np.sin(psi + betas[k])
        #dy = r * np.sin(psi )
        y = dy + cur_y
        print(psi, betas[k], dx, dy)
        plt.plot(xs, ys, color='red')
        plt.plot([cur_x, x], [cur_y, y], color='blue')
        plt.scatter(x, y)
        plt.pause(0.01)
        prob_points.append([x, y])
    plt.pause(100)
    return np.array(prob_points)


predict_linear_probs(ts, xs, ys, 0.2)
