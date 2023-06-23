import cv2
import numpy as np
import importlib
import os
from extensions.camera_kinematics import CameraKinematics
from lib.utils.vision import APCE,PSR
from lib.tracking.types import ExtType, Trackers

import sys
from pathlib import Path
pth = str(Path(__file__).parent.resolve()) + "/../trackers"
sys.path.insert(0, pth)
current_module_pth = str(Path(__file__).parent.resolve()) + "/.."


class PyTracker:
    def __init__(self, img_dir, tracker_title, dataset_config, _gts, fl, sts, ext_type):
        self.ext_type = ext_type
        self.img_dir = img_dir
        self.trackerType = tracker_title
        self.trackFuncName = self.trackerType.group.name + "_" + self.ext_type.name + "_track"
        self.initFuncName = self.trackerType.group.name + "_init"
        self.trackerName = self.trackerType.name
        self.frame_list = fl
        self.gts = _gts
        self.states = sts
        self.fov = dataset_config['fov']
        self.ethTracker = False
        self.datasetCfg = dataset_config

        start_frame = dataset_config['start_frame']
        end_frame = dataset_config['end_frame']
        self.init_gt = self.gts[0]
        self.frame_list = self.frame_list[start_frame-1:end_frame]
        self.states = self.states[start_frame-1:end_frame]

        self.ethTracker=True
        self.viot = dataset_config['ext_type'] == ExtType.viot
        self.extName = dataset_config['ext_type'].name

    # def getETHTracker(self, name, params):
    #     param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(name, params))
    #     params = param_module.parameters()
    #     params.tracker_name = name
    #     params.param_name = params

    #     tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tracker', name))
    #     tracker_module = importlib.import_module('pytracking.tracker.{}'.format(name))
    #     tracker_class = tracker_module.get_tracker_class()

    #     tracker = tracker_class(params)

    #     if hasattr(tracker, 'initialize_features'):
    #         tracker.initialize_features()
    #     return tracker

    # def initETHTracker(self, frame, bbox):
    #     x, y, w, h = bbox
    #     init_state = [x, y, w, h]
    #     box = {'init_bbox': init_state, 'init_object_ids': [1, ], 'object_ids': [1, ],
    #                 'sequence_object_ids': [1, ]}
    #     self.tracker.initialize(frame, box)

    # def doTrack(self, current_frame, verbose, est_loc, do_learning, viot=False):
    # 	if self.ethTracker:
    #         if viot:
    #             out = self.tracker.track(current_frame, FI=est_loc, do_learning=do_learning)
    #         else:
    #     	    out = self.tracker.track(current_frame)

    #         bbox = [int(s) for s in out['target_bbox']]
    # 	else:
    # 	    if viot:
    # 	        bbox=self.tracker.update(current_frame,vis=verbose,FI=est_loc, \
    # 	                                 do_learning=do_learning) ## VIOT
    # 	    else:
    # 	    	bbox=self.tracker.update(current_frame,vis=verbose)
    # 	    	# bbox=self.tracker.update(current_frame,vis=verbose,FI=est_loc)
    # 	return bbox

    def mxf_viot_track(self, inp):
        return self.eth_viot_track(inp)

    def mxf_raw_track(self, inp):
        return self.eth_raw_track(inp)

    def eth_viot_track(self, inp):
        out = self.tracker.track(inp['current_frame'], FI=inp['est_loc'], 
                                 do_learning=inp['do_learning'])
        bbox = [int(s) for s in out['target_bbox']]
        return bbox

    def cf_viot_track(self, inp):
        bbox=self.tracker.update(inp['current_frame'], vis=inp['verbose'], FI=inp['est_loc'], 
                                 do_learning=inp['do_learning']) ## VIOT
        return bbox

    def eth_raw_track(self, inp):
        out = self.tracker.track(inp['current_frame'])
        bbox = [int(s) for s in out['target_bbox']]
        return bbox

    def cf_raw_track(self, inp):
        bbox=self.tracker.update(inp['current_frame'],vis=inp['verbose'])
        return bbox

    def eth_init(self, frame, bbox):
        param_module = importlib.import_module('pytracking.parameter.{}.{}'
                                               .format(self.trackerType.tag, 
                                                       self.trackerType.config))
        params = param_module.parameters()
        params.tracker_name = self.trackerType.tag
        params.param_name = params
        tracker_module = importlib.import_module('pytracking.tracker.{}'.
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
        self.ratio_thresh = self.datasetCfg['ratio_tresh']['MIXFORMER_VIT']
        self.interp_factor = self.datasetCfg['interp_factor']['MIXFORMER_VIT']
        x, y, w, h = bbox
        init_state = [x, y, w, h]
        box = {'init_bbox': init_state, 'init_object_ids': [1, ], 'object_ids': [1, ],
               'sequence_object_ids': [1, ]}
        self.tracker.initialize(frame, box)

    def cf_init(self):
        pass

    def tracking(self, data_name, verbose=True):
        save_phrase = current_module_pth + "/results/{:s}_{:s}_".format(self.trackerName, data_name) + self.extName
        video_path = save_phrase + ".mp4"
        poses = []
        ratios = []
        init_frame = cv2.imread(self.frame_list[0])
        #print(init_frame.shape)
        init_gt = np.array(self.init_gt)
        x1, y1, w, h =init_gt
        init_gt=tuple(init_gt)
        getattr(self, self.initFuncName)(init_frame, init_gt)
        writer=None
        if verbose is True and video_path is not None:
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (init_frame.shape[1], init_frame.shape[0]))
            ratios_path = os.path.splitext(video_path)[0] + ".txt"
        ## kinematic model for MAVIC Mini with horizontal field of view (hfov)
        ## equal to 66 deg.
        kin = CameraKinematics(self.interp_factor, init_frame.shape[1]/2, init_frame.shape[0]/2,\
                                w=init_frame.shape[1], h=init_frame.shape[0],\
                                hfov=self.fov, vis=False)
        psr0=-1
        psr=-1
        est_loc=init_gt
        stop=False
        last_bbox=None
        self.last_bbox=None

        for idx in range(len(self.frame_list)):
            if idx != 0:
                current_frame=cv2.imread(self.frame_list[idx])
                height,width=current_frame.shape[:2]

                if stop:
                    bbox=last_bbox
                else:
                    args = {}
                    args['current_frame'] = current_frame
                    args['verbose'] = verbose
                    args['est_loc'] = est_loc
                    args['do_learning'] = psr/psr0>self.ratio_thresh and not stop
                    args['viot'] = self.viot
                    bbox = getattr(self, self.trackFuncName)(args)
                    # bbox=self.doTrack(current_frame, verbose, est_loc, 
                    #                   psr/psr0>self.ratio_thresh and not stop, viot=self.viot)

                stop=bbox[2] > width or bbox[3] > height

                ## evaluating tracked target
                if self.trackerType == Trackers.MIXFORMER:
                    psr = self.tracker.pred_score
                elif self.ethTracker:
                    apce = APCE(self.tracker.score)
                    psr = apce
                else:
                    psr = PSR(self.tracker.score)
                # F_max = np.max(self.tracker.score)
                if psr0 is -1: psr0=psr
                ratios.append(psr/psr0)
                ## estimating target location using kinematc model
                if psr/psr0 > self.ratio_thresh:
                    last_bbox=bbox
                    self.last_bbox = last_bbox
                    est_loc = kin.updateRect3D(self.states[idx,:], self.states[0,1:4], current_frame, bbox)
                else:
                    est_loc = kin.updateRect3D(self.states[idx,:], self.states[0,1:4], current_frame, None)

                # print("psr ratio: ",psr/psr0, " learning: ", psr/psr0 > self.ratio_thresh, " est: ", est_loc)
                x1,y1,w,h=bbox
                if verbose is True:
                    self.visualize(current_frame, x1, y1, w, h, psr, psr0, writer, width, height)

            poses.append(np.array([int(x1), int(y1), int(w), int(h)]))
        np.savetxt(ratios_path, np.array(ratios), delimiter=',')
        return np.array(poses), save_phrase

    def visualize(self, current_frame, x1, y1, w, h, psr, psr0, writer, width, height):
        show_frame=cv2.rectangle(current_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255, 0, 0),2)
        if (psr/psr0) <= self.ratio_thresh:
            show_frame = cv2.line(show_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 0, 255), 2)
            show_frame = cv2.line(show_frame, (int(x1+w), int(y1)), (int(x1), int(y1 + h)), (0, 0, 255), 2)
        if self.trackerType == Trackers.MIXFORMER:
            for zone in self.tracker._sample_coords:
                show_frame=cv2.rectangle(show_frame, (int(zone[1]), int(zone[0])), 
                                            (int(zone[3]), int(zone[2])), (0, 255, 255),1)
        else:
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
            x0=np.clip(x0,0,width-1)
            y0=np.clip(y0,0,height-1)
            center=(x0,y0)
            xmin = int(center[0]) - size[0] // 2
            xmax = int(center[0]) + size[0] // 2 + size[0] % 2
            ymin = int(center[1]) - size[1] // 2
            ymax = int(center[1]) + size[1] // 2 + size[1] % 2
            left = abs(xmin) if xmin < 0 else 0
            xmin = 0 if xmin < 0 else xmin
            right = width - xmax
            xmax = width if right < 0 else xmax
            right = size[0] + right if right < 0 else size[0]
            top = abs(ymin) if ymin < 0 else 0
            ymin = 0 if ymin < 0 else ymin
            down = height - ymax
            ymax = height if down < 0 else ymax
            down = size[1] + down if down < 0 else size[1]
            score = score[top:down, left:right]
            crop_img = current_frame[ymin:ymax, xmin:xmax]
            score_map = cv2.addWeighted(crop_img, 0.6, score, 0.4, 0)
            current_frame[ymin:ymax, xmin:xmax] = score_map
        
            if self.trackerType == Trackers.DIMP50 or \
                self.trackerType == Trackers.KYS or \
                self.trackerType == Trackers.TOMP or \
                self.trackerType == Trackers.PRDIMP50:
                for zone in self.tracker._sample_coords:
                    show_frame=cv2.rectangle(show_frame, (int(zone[1]), int(zone[0])), 
                                                (int(zone[3]), int(zone[2])), (0, 255, 255),1)

            if self.viot:
                p1 = (int(est_loc[0]+est_loc[2]/2-1), int(est_loc[1]+est_loc[3]/2-1))
                p2 = (int(est_loc[0]+est_loc[2]/2+1), int(est_loc[1]+est_loc[3]/2+1))
                show_frame = cv2.rectangle(show_frame, p1, p2, (255, 0, 0),2)

            # cv2.putText(show_frame, 'APCE:' + str(apce)[:5], (0, 250), cv2.FONT_HERSHEY_COMPLEX, 2,
            #             (0, 0, 255), 5)
            # cv2.putText(show_frame, 'PSR:' + str(psr)[:5], (0, 300), cv2.FONT_HERSHEY_COMPLEX, 2,
            #             (255, 0, 0), 5)
            # cv2.putText(show_frame, 'Fmax:' + str(F_max)[:5], (0, 350), cv2.FONT_HERSHEY_COMPLEX, 2,
            #             (255, 0, 0), 5)

            if not IN_COLAB:
                cv2.imshow('demo', show_frame)

        if writer is not None:
            writer.write(show_frame)
        cv2.waitKey(1)