import cv2 as cv
import numpy as np

def calc_ground_range(alt, range):
    assert(alt > 0 and range > 0)
    ang = np.arccos(alt/range)
    return np.tan(ang) * alt

def make_panorama_scan(map, panorama_width, sr2br, out_size: tuple=None):
    # Define the circles
    center = (map.shape[1] // 2, map.shape[0] // 2)
    radius1 = map.shape[0] // 2
    radius2 = int(sr2br * radius1)
    half_width = int(panorama_width/2)
    diff = radius1-radius2
    # Create a black image for the panorama
    panorama1 = np.zeros((diff, half_width), dtype=np.uint8)
    panorama2 = np.zeros((diff, half_width), dtype=np.uint8)
    # Loop through each angle
    for a in range(1, half_width, 3):
        angle = (a-1)/(panorama_width/360)
        # Rotate the probability map
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(map, M, (map.shape[1], map.shape[0]))
        # Extract the portion of the rotated image that matches the circles
        mask1 = cv.circle(np.zeros_like(rotated), center, radius1, 255, -1)
        mask2 = cv.circle(np.zeros_like(rotated), center, radius2, 255, -1)
        #print("md1 ", mask1.shape)
        #print("md2 ", mask2.shape)
        #print("md3 ", rotated.shape)
        #print("md4 ", mask1.dtype)
        #print("md5 ", mask2.dtype)
        #print("md6 ", rotated.dtype)
        masked_rotated = cv.bitwise_and(rotated, rotated, mask=cv.bitwise_xor(mask1, mask2))
        # Paste the column into the panorama
        panorama1[:, a-1:a+2] = masked_rotated[0:diff, center[0]-1:center[0]+2]
        panorama2[:, a-1:a+2] = np.flip(masked_rotated[-diff:, center[0]-1:center[0]+2], 0)
    panorama = cv.hconcat([panorama1, panorama2])
    if out_size is None:
        return panorama
    else:
        return cv.resize(panorama, out_size, interpolation=cv.INTER_LINEAR)


# def predict_linear_probs(ts, xs, ys, dt, next_x, next_y, v_std_dev, beta_std_dev, samples_num):
def predict_linear_probs(ts, xs, ys, dt, v_std_dev, beta_std_dev, samples_num):
    l = len(ts)
    assert (l == len(xs) == len(ys))
    # cur_t = ts[-1]
    # cur_x = xs[-1]
    # cur_y = ys[-1]
    # psi = np.arctan2(next_y-cur_y, next_x-cur_x)
    # cur_t, last_t = ts[-1], ts[-2]
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
    for k in range(samples_num):
        r = vs[k] * dt
        dx = r * np.cos(psi + betas[k]) 
        x = dx + cur_x
        dy = r * np.sin(psi + betas[k])
        y = dy + cur_y
        prob_points.append([x, y])
    return np.array(prob_points)


def NED2IMG_single(x, y, minX, minY, mppr):
    pixel_x = int((x - minX) / mppr)
    pixel_y = int((y - minY) / mppr)
    return pixel_x, pixel_y


def NED2IMG_array(arr, minX, minY, mppr):
    # assuming each row of 'arr' as [x, y] in NED
    l = arr.shape[0]
    pix_points = np.array([])
    for k in range(l):
        px, py = NED2IMG_single(arr[k,0], arr[k,1], minX, minY, mppr)
        if k == 0:
            pix_points = np.array([[px, py]])
        else:
            pix_points = np.vstack([pix_points, np.array([px, py])])
    return pix_points  


