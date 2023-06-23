import matplotlib.pyplot as plt

def plot_precision(gts,preds,save_path):
    plt.figure(2)
    # x,y,w,h
    threshes,precisions=get_thresh_precision_pair(gts,preds)
    idx20 = [i for i, x in enumerate(threshes) if x == 20][0]
    plt.plot(threshes,precisions,label=str(precisions[idx20])[:5])
    plt.title('Precision Plots')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def plot_success(gts,preds,save_path):
    plt.figure(1)
    threshes, successes=get_thresh_success_pair(gts, preds)
    plt.plot(threshes,successes,label=str(calAUC(successes))[:5])
    plt.title('Success Plot')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def plot_kinematics(eul,inertia_dir,ax_3d,corners):

    tl, tr, bl, br = corners

    ax_3d.cla()
    ax_3d.plot([0,inertia_dir[0]],[0,inertia_dir[1]],[0,inertia_dir[2]], color='r')

    ax_3d.quiver(0, 0, 0, 1, 0, 0, length=1, linewidth=2, color='red')
    ax_3d.quiver(0, 0, 0, 0, 1, 0, length=1, linewidth=2, color='green')
    ax_3d.quiver(0, 0, 0, 0, 0, 1, length=1, linewidth=2, color='blue')

    ax_3d.text(tl[0], tl[1], tl[2]+0.1, "tl", color='black')
    ax_3d.text(tr[0], tr[1], tr[2]+0.1, "tr", color='black')
    ax_3d.text(bl[0], bl[1], bl[2]-0.1, "bl", color='black')
    ax_3d.text(br[0], br[1], br[2]-0.1, "br", color='black')

    ax_3d.plot([0,tl[0]],[0,tl[1]],[0,tl[2]], color='black')
    ax_3d.plot([0,tr[0]],[0,tr[1]],[0,tr[2]], color='black')
    ax_3d.plot([0,bl[0]],[0,bl[1]],[0,bl[2]], color='black')
    ax_3d.plot([0,br[0]],[0,br[1]],[0,br[2]], color='black')
    ax_3d.plot([tl[0],tr[0]],[tl[1],tr[1]],[tl[2],tr[2]], color='black')
    ax_3d.plot([tl[0],bl[0]],[tl[1],bl[1]],[tl[2],bl[2]], color='black')
    ax_3d.plot([tr[0],br[0]],[tr[1],br[1]],[tr[2],br[2]], color='black')
    ax_3d.plot([bl[0],br[0]],[bl[1],br[1]],[bl[2],br[2]], color='black')

    ax_3d.set_xlim([-1.5,1.5])
    ax_3d.set_ylim([-1.5,1.5])
    ax_3d.set_zlim([-1.5,1.5])

    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')
    plt.pause(0.01)
