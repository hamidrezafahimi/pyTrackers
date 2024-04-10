import cv2  
import numpy as np
import importlib
import os
from extensions.viot.viot import VIOT
from extensions.camera_kinematics import CameraKinematics
from lib.utils.vision import APCE,PSR
from .types import ExtType
import json
import sys
from pathlib import Path
root_path = str(Path(__file__).parent.resolve()) + "/../.."
sys.path.insert(0, root_path + "/trackers")
import matplotlib.pyplot as plt
from extensions.aerial_tracker import AerialTracker
import time

class PyTracker:
    def __init__(self, img_dir, tracker_title, dataset_config, ext_type, verbose):
        self.verbose = verbose
        self.extType = ext_type
        self.img_dir = img_dir
        self.trackerType = tracker_title
        self.fov = dataset_config['fov']
        self.datasetCfg = dataset_config
        # self.viot = dataset_config['ext_type'] == ExtType.viot
        self.eval0 = None
        initFuncName = self.trackerType.group.name + "_init"
        self.init = getattr(self, initFuncName)
        evalFuncName = self.trackerType.group.name + "_eval"
        self.eval = getattr(self, evalFuncName)
        if ext_type == ExtType.raw:
            trackFuncName = self.trackerType.group.name + "_raw_track"
        else:
            trackFuncName = self.trackerType.group.name + "_ext_track"
        self.track = getattr(self, trackFuncName)
        processFuncName = "process_" + self.extType.name
        self.process = getattr(self, processFuncName)
        visFuncName = self.trackerType.group.name + "_visualize"
        visualize_ext_name = "visualize_" + self.extType.name
        self.visualize_ext = getattr(self, visualize_ext_name)
        self.type_visualize = getattr(self, visFuncName)
        self.ratio_thresh = self.datasetCfg['ratio_thresh']
        self.interp_factor = self.datasetCfg['interp_factor']
        # # self.pPath = plot_pathfig = 
        # self.ax = plt.figure().add_subplot(111)

    def mxf_ext_track(self, current_frame, est_loc, valid_track):
        return self.eth_ext_track(current_frame, est_loc, valid_track)

    def eth_ext_track(self, current_frame, est_loc, valid_track):
        out = self.tracker.track(current_frame, FI=est_loc, do_learning=valid_track)
        return [int(s) for s in out['target_bbox']]

    def cf_ext_track(self, current_frame, est_loc, valid_track):
        return self.tracker.update(current_frame, vis=self.verbose, FI=est_loc, do_learning=valid_track)

    def mxf_raw_track(self, frame):
        return self.eth_raw_track(frame)

    def eth_raw_track(self, frame):
        out = self.tracker.track(frame)
        return [int(s) for s in out['target_bbox']]

    def cf_raw_track(self, frame):
        return self.tracker.update(frame, vis=self.verbose)
    
    def cf_eval(self):
        return PSR(self.tracker.score)

    def eth_eval(self):
        return APCE(self.tracker.score)

    def mxf_eval(self):
        return self.tracker.pred_score

    def eth_init(self, frame, bbox):
        param_module = importlib.import_module('pytracking.pytracking.parameter.{}.{}'
                                               .format(self.trackerType.tag, 
                                                       self.trackerType.config['type']))
        params = param_module.parameters()
        params.tracker_name = self.trackerType.tag
        params.param_name = params
        tracker_module = importlib.import_module('pytracking.pytracking.tracker.{}'.
                                                 format(self.trackerType.tag))
        tracker_class = tracker_module.get_tracker_class()
        self.tracker = tracker_class(params)
        if hasattr(self.tracker, 'initialize_features'):
            self.tracker.initialize_features()
        x, y, w, h = bbox
        init_state = [x, y, w, h]
        box = {'init_bbox': init_state, 'init_object_ids': [1, ], 'object_ids': [1, ],
               'sequence_object_ids': [1, ]}
        self.tracker.initialize(frame, box)

    def mxf_init(self, frame, bbox):
        param_module = importlib.import_module(
                                    'MixFormer.lib.test.parameter.mixformer_{}_online'
                                    .format(self.trackerType.config['type']))
        tracker_module = importlib.import_module(
                                    'MixFormer.lib.test.tracker.mixformer_{}_online'
                                    .format(self.trackerType.config['type']))
        params = param_module.parameters(self.trackerType.config['yaml_name'], self.trackerType.config['model'], 
                            self.trackerType.config['search_area_scale'])
        self.tracker = tracker_module.MixFormerOnline(params, self.trackerType.config['dataset_name'])
        
        x, y, w, h = bbox
        init_state = [x, y, w, h]
        box = {'init_bbox': init_state, 'init_object_ids': [1, ], 'object_ids': [1, ],
               'sequence_object_ids': [1, ]}
        self.tracker.initialize(frame, box)

    def cf_init(self):
        pass

    def initLog(self, data_name, init_gt):
        self.data_path = root_path + "/results/{:s}_{:s}_".format(self.trackerType.name, 
                                                                    data_name) + self.extType.name
        video_path = self.data_path + ".mp4"
        self.writer=None
        if self.verbose is True:
            self.writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, 
                                          (self.frameWidth, self.frameHeight))
        ratios_path = self.data_path + "_params.txt"
        if os.path.exists(ratios_path):
            os.remove(ratios_path)
        poses_path = self.data_path + "_poses.txt"
        if os.path.exists(poses_path):
            os.remove(poses_path)
        self.ratios_file = open(ratios_path, 'a')
        self.poses_file = open(poses_path, 'a')
        self.tracker_bboxes = [np.array([int(init_gt[0]), int(init_gt[1]), int(init_gt[2]), int(init_gt[3])])]
    
    def endLog(self, data_name):
        pose_results = {}
        pose_results[data_name] = {}
        pses = []
        for pred in self.tracker_bboxes:
            pses.append(list(pred.astype(np.int)))
        pose_results[data_name]['tracker_{:s}_preds'.format(self.trackerType.tag)] = pses
        f = open(self.data_path + ".json", 'w')
        json_content = json.dumps(pose_results, default=str)
        f.write(json_content)
        f.close()
        self.ratios_file.close()
        self.poses_file.close()

    def log(self, bbox, frame, log_nums, pose=None):
        np.savetxt(self.ratios_file, np.array(log_nums).reshape(1, -1), delimiter=", ")
        if not pose is None and not pose[1] is None:
            np.savetxt(self.poses_file, np.array(pose).reshape(1, -1), delimiter=", ")
        if self.writer is not None:
            self.writer.write(frame)
        self.tracker_bboxes.append(bbox)

    def postProc(self, bbox):
        keeps_on = not (bbox[2] > self.frameWidth or bbox[3] > self.frameHeight)
        if not keeps_on:
            bbox = None
        ## evaluating tracked target
        score = self.eval()
        if self.eval0 is None: 
            self.eval0 = score
        ratio = score/self.eval0
        valid = ratio > self.ratio_thresh
        return valid, keeps_on, score, ratio

    def process_viot(self, frame_list, states, init_gt):
        kin = VIOT(self.interp_factor, self.frameWidth/2, self.frameHeight/2, w=self.frameWidth,
                       h=self.frameHeight, hfov=self.fov, vis=False, ref=states[0,1:4])
        est_loc = tuple(init_gt)
        valid = True
        keeps = True
        for idx in range(1, len(frame_list)):
            current_frame=cv2.imread(frame_list[idx])
            bbox = self.track(current_frame, est_loc, valid and keeps)
            valid, keeps, score, ratio = self.postProc(bbox)
            ## estimating next target location using kinematc model
            if valid:
                est_loc, p = kin.updateRect3D(states[idx,:], current_frame, bbox)
            else:
                est_loc, p = kin.updateRect3D(states[idx,:], current_frame, None)
            sh_frame = self.visualize(current_frame, bbox, valid)
            ## Calling the following function leads to drawing a point being updated - only - based
            ## on motion model when the target is occluded
            # print(est_loc)
            self.visualize_ext(sh_frame, est_loc)
            self.log(np.array([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]), sh_frame,
                     [idx, score, ratio, valid], [idx, *p])

    def process_raw(self, frame_list, states, b=None):
        kin = CameraKinematics(cx=self.frameWidth/2, cy=self.frameHeight/2, w=self.frameWidth,
                               h=self.frameHeight, hfov=self.fov, ref=states[0,1:4])
        for idx in range(1, len(frame_list)):
            current_frame=cv2.imread(frame_list[idx])
            bbox = self.track(current_frame)
            valid, _, score, ratio = self.postProc(bbox)
            sh_frame = self.visualize(current_frame, bbox, valid)
            self.log(np.array([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]), 
                     sh_frame, [idx, score, ratio, valid], 
                     [idx, *kin.rect_to_pose(bbox, states[idx,4:7], states[idx,1:4])[0]])

    def process_kpt(self, frame_list, states, b=None):
        kin = AerialTracker(wps=states[:,1:4], vr=50, mppr=0.15,
                            cx=self.frameWidth/2, cy=self.frameHeight/2, w=self.frameWidth,
                            h=self.frameHeight, hfov=self.fov)
        for idx in range(1, len(frame_list)):
            current_frame = cv2.imread(frame_list[idx])
            bbox = self.track(current_frame, None, None)
            valid, _, score, ratio = self.postProc(bbox)
            tgt_pos = None
            target_pose = None
            if valid:            
                tgt_pos, cam_pos = kin.rect_to_pose(bbox, states[idx,4:7], states[idx,1:4])
                target_pose = [states[idx,0], tgt_pos[0], tgt_pos[1]]
            else:
                _, cam_pos = kin.rect_to_pose(bbox, states[idx,4:7], states[idx,1:4])
            
            kin.predict(states[idx,4:7], states[idx,1:4], target_pose, states[idx,0])
            sh_frame = self.visualize(current_frame, bbox, valid)
            # self.visualize_ext(tgt_est_pos, cam_pos)

    # def process_kpt(self, frame_list, states, init_gt):
    #     kin = CameraKinematics(cx=self.frameWidth/2, cy=self.frameHeight/2, w=self.frameWidth,
    #                            h=self.frameHeight, hfov=self.fov, ref=states[0,1:4])
    #     kpt = VOTPathObserver()
    #     est_loc = tuple(init_gt)
    #     valid = True
    #     for idx in range(1, len(frame_list)-1):
    #         current_frame=cv2.imread(frame_list[idx])
    #         bbox = self.track(current_frame, est_loc, valid)
    #         valid, _, score, ratio = self.postProc(bbox)
    #         if valid:            
    #             tgt_pos, cam_pos = kin.rect_to_pose(bbox, states[idx,4:7], states[idx,1:4])
    #         else:
    #             _, cam_pos = kin.rect_to_pose(bbox, states[idx,4:7], states[idx,1:4])
    #             tgt_pos = None
    #         tgt_next_pos = kpt.doPrediction(tgt_pos, states[idx,0], states[idx+1,0])
    #         est_loc = kin.pose_to_limited_rect([*tgt_next_pos,0], cam_pos, states[idx,4:7], bbox)
    #         sh_frame = self.visualize(current_frame, bbox, valid)
    #         if not est_loc is None:
    #             self.viot_vis(sh_frame, est_loc)
    #         self.log(np.array([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]), sh_frame,
    #                  [idx, score, ratio, valid], [idx, tgt_next_pos[0], tgt_next_pos[1], 0])

    def tracking(self, data_name, frame_list, init_gt, states):
        init_frame = cv2.imread(frame_list[0])
        self.frameHeight, self.frameWidth = init_frame.shape[:2]
        self.initLog(data_name, init_gt=init_gt)
        self.init(init_frame, init_gt)
        self.process(frame_list, states, init_gt)
        self.endLog(data_name)
        print('processing {:s} with {:s} tracker done!'.format(data_name, self.trackerType.tag))

    def mxf_visualize(self, show_frame=None, current_frame=None, bbox=None):
        for zone in self.tracker._sample_coords:
            show_frame=cv2.rectangle(show_frame, (int(zone[1]), int(zone[0])),
                                        (int(zone[3]), int(zone[2])), (0, 255, 255),1)

    def eth_visualize(self, show_frame=None, current_frame=None, bbox=None):
        self.cf_visualize(show_frame=show_frame, current_frame=current_frame, bbox=bbox)
        self.mxf_visualize(show_frame=show_frame)

    def cf_visualize(self, show_frame=None, current_frame=None, bbox=None):
        x1,y1,w,h = bbox
        if len(current_frame.shape)==2:
            current_frame=cv2.cvtColor(current_frame,cv2.COLOR_GRAY2BGR)
        score = self.tracker.score
        size=self.tracker.crop_size
        score = cv2.resize(score, size)
        score -= score.min()
        score =score/ score.max()
        score = (score * 255).astype(np.uint8)
        score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
        center = (int(x1+w/2-self.tracker.trans[1]),int(y1+h/2-self.tracker.trans[0]))
        x0,y0=center
        x0=np.clip(x0,0,self.frameWidth-1)
        y0=np.clip(y0,0,self.frameHeight-1)
        center=(x0,y0)
        xmin = int(center[0]) - size[0] // 2
        xmax = int(center[0]) + size[0] // 2 + size[0] % 2
        ymin = int(center[1]) - size[1] // 2
        ymax = int(center[1]) + size[1] // 2 + size[1] % 2
        left = abs(xmin) if xmin < 0 else 0
        xmin = 0 if xmin < 0 else xmin
        right = self.frameWidth - xmax
        xmax = self.frameWidth if right < 0 else xmax
        right = size[0] + right if right < 0 else size[0]
        top = abs(ymin) if ymin < 0 else 0
        ymin = 0 if ymin < 0 else ymin
        down = self.frameHeight - ymax
        ymax = self.frameHeight if down < 0 else ymax
        down = size[1] + down if down < 0 else size[1]
        score = score[top:down, left:right]
        crop_img = current_frame[ymin:ymax, xmin:xmax]
        score_map = cv2.addWeighted(crop_img, 0.6, score, 0.4, 0)
        current_frame[ymin:ymax, xmin:xmax] = score_map

    def visualize_viot(self, show_frame, est_loc):
        p1 = (int(est_loc[0]+est_loc[2]/2-1), int(est_loc[1]+est_loc[3]/2-1))
        p2 = (int(est_loc[0]+est_loc[2]/2+1), int(est_loc[1]+est_loc[3]/2+1))
        show_frame = cv2.rectangle(show_frame, p1, p2, (255, 0, 0),2)

    def visualize(self, current_frame, bbox, valid):
        if self.verbose is False:
            return None
        x1,y1,w,h = bbox
        show_frame=cv2.rectangle(current_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0),2)
        if not valid:
            show_frame = cv2.line(show_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 2)
            show_frame = cv2.line(show_frame, (int(x1+w), int(y1)), (int(x1), int(y1 + h)), (0, 0, 255), 2)
        self.type_visualize(show_frame=show_frame, current_frame=current_frame, bbox=bbox)
        cv2.imshow('demo', show_frame)
        cv2.waitKey(1)
        return show_frame
    
    def visualize_kpt(self, frame):
        plt.cla()
        self.ax.plot()
        pass
