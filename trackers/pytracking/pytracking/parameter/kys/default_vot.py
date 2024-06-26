import sys
from pathlib import Path
root_path = str(Path(__file__).parent.resolve()) + "/../../../.."
sys.path.append(root_path)
from pytracking.pytracking.utils import TrackerParams
from pytracking.pytracking.features.net_wrappers import NetWithBackbone


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    params.image_sample_size = 14*16
    params.search_area_scale = 4

    # Learning parameters
    params.sample_memory_size = 250
    params.learning_rate = 0.0075
    params.init_samples_minimum_weight = 0.0
    params.train_skipping = 10

    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 25
    params.net_opt_update_iter = 3
    params.net_opt_hn_iter = 3

    params.output_sigma_factor = 1/4

    # Init augmentation parameters
    params.use_augmentation = True
    params.augmentation = {'fliplr': True,
                           'rotate': [5, -5, 10, -10, 20, -20, 30, -30, 45, -45, -60, 60],
                           'blur': [(2, 0.2), (0.2, 2), (3, 1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6, -0.6)],
                           'dropout': (7, 0.2)}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1 / 3

    # localization parameters
    params.window_output = True
    params.use_clipped_window = True
    params.effective_search_area = 4.0
    params.apply_window_to_dimp_score = True

    params.target_not_found_threshold_fused = 0.05
    params.dimp_threshold = 0.05

    params.reset_state_during_occlusion = True

    params.prev_feat_remove_subpixel_shift = True
    params.move_feat_to_center = True

    params.perform_hn_mining_dimp = True
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale_safe = 2.2
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True

    # IoUnet parameters
    params.use_iou_net = True
    params.iounet_augmentation = False
    params.iounet_use_log_scale = True
    params.iounet_k = 3
    params.num_init_random_boxes = 9
    params.box_jitter_pos = 0.1
    params.box_jitter_sz = 0.5
    params.maximal_aspect_ratio = 6
    params.box_refinement_iter = 5
    params.box_refinement_step_length = 1
    params.box_refinement_step_decay = 1

    params.remove_offset_in_fused_score = True
    params.score_downsample_factor = 1

    params.net = NetWithBackbone(net_path='kys.pth',
                                 use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'
    return params
