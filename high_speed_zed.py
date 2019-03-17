import argparse
import configparser
import os
import queue
import threading
import time
import logging
import faulthandler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import chainer
import cv2
import numpy as np
from PIL import Image
import pyzed.sl as sl

from predict import get_feature, get_humans_by_feature, get_humans3d, draw_humans, create_model, load_config
from utils import parse_size

from OpenGL.GLUT import *
import viewer.viewer3D as tv


QUEUE_SIZE = 5

"""
Bonus script
If you have good USB camera which gets image as well as 60 FPS,
this script will be helpful for realtime inference
"""


class Capture(threading.Thread):

    def __init__(self, cap, insize):
        super(Capture, self).__init__()
        self.cap = cap
        self.insize = insize
        self.stop_event = threading.Event()
        self.queue = queue.Queue(QUEUE_SIZE)
        self.name = 'Capture'

    def run(self):
        left = sl.Mat()
        depth = sl.Mat()
        runtime = sl.RuntimeParameters()
        while not self.stop_event.is_set():
            try:
                self.cap.grab(runtime)
                self.cap.retrieve_image(left, sl.VIEW.VIEW_LEFT, width=self.insize[0], height=self.insize[1])
                self.cap.retrieve_measure(depth, sl.MEASURE.MEASURE_XYZ, width=self.insize[0], height=self.insize[1])
                image = cv2.cvtColor(left.get_data(), cv2.COLOR_BGRA2RGB)
                self.queue.put((image, depth.get_data()), timeout=1)
            except queue.Full:
                pass

    def get(self):
        return self.queue.get(timeout=1)

    def stop(self):
        logger.info('{} will stop'.format(self.name))
        self.stop_event.set()


class Predictor(threading.Thread):

    def __init__(self, model, cap):
        super(Predictor, self).__init__()
        self.cap = cap
        self.model = model
        self.stop_event = threading.Event()
        self.queue = queue.Queue(QUEUE_SIZE)
        self.name = 'Predictor'

    def run(self):
        while not self.stop_event.is_set():
            try:
                image, depth = self.cap.get()
                with chainer.using_config('autotune', True), \
                        chainer.using_config('use_ideep', 'auto'):
                    feature_map = get_feature(self.model, image.transpose(2, 0, 1).astype(np.float32))
                self.queue.put((image, feature_map, depth), timeout=1)
            except queue.Full:
                pass
            except queue.Empty:
                pass

    def get(self):
        return self.queue.get(timeout=1)

    def stop(self):
        logger.info('{} will stop'.format(self.name))
        self.stop_event.set()


def high_speed(args, viewer):
    config = load_config(args)
    dataset_type = config.get('dataset', 'type')
    detection_thresh = config.getfloat('predict', 'detection_thresh')
    min_num_keypoints = config.getint('predict', 'min_num_keypoints')
    model = create_model(args, config)
    svo_file_path = config.get('zed', 'svo_file_path')
    
    init_cap_params = sl.InitParameters()
    if svo_file_path:
        print("Loading SVO file " + svo_file_path)
        init_cap_params.svo_input_filename = svo_file_path
        init_cap_params.svo_real_time_mode = True
    init_cap_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720
    init_cap_params.depth_mode=sl.DEPTH_MODE.DEPTH_MODE_ULTRA
    init_cap_params.coordinate_units=sl.UNIT.UNIT_METER
    init_cap_params.coordinate_system=sl.COORDINATE_SYSTEM.COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP
    
    cap = sl.Camera()
    if not cap.is_opened():
        print("Opening ZED Camera...")
    status = cap.open(init_cap_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    capture = Capture(cap, model.insize)
    predictor = Predictor(model=model, cap=capture)

    capture.start()
    predictor.start()

    fps_time = 0

    main_event = threading.Event()

    try:
        while not main_event.is_set() and cap.is_opened():
            try:
                image, feature_map, depth = predictor.get()
                humans = get_humans_by_feature(
                    model,
                    feature_map,
                    detection_thresh,
                    min_num_keypoints
                )
                humans_3d = get_humans3d(humans,depth)
            except queue.Empty:
                continue
            except Exception as e:
                print(e)
                break
            pilImg = Image.fromarray(image)
            pilImg = draw_humans(
                model.keypoint_names,
                model.edges,
                pilImg,
                humans,
                None,
                visbbox=config.getboolean('predict', 'visbbox'),
            )
            img_with_humans = cv2.cvtColor(np.asarray(pilImg), cv2.COLOR_RGB2BGR)
            img_with_humans = cv2.resize(img_with_humans, (700,400))#(3 * model.insize[0], 3 * model.insize[1]))
            msg = 'GPU ON' if chainer.backends.cuda.available else 'GPU OFF'
            msg += ' ' + config.get('model_param', 'model_name')
            fps_display='FPS: %f' % (1.0 / (time.time() - fps_time))
            str_to_dsplay = msg + " " + fps_display
            cv2.putText(img_with_humans, fps_display, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Pose Proposal Network' + msg, img_with_humans)

            viewer.update_text(str_to_dsplay)
            viewer.update_humans(humans_3d)

            fps_time = time.time()
            # press Esc to exit
            if cv2.waitKey(1) == 27:
                exit
                main_event.set()
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        main_event.set()

    capture.stop()
    predictor.stop()

    capture.join()
    predictor.join()
    cap.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path/to/model', type=str)
    return parser.parse_args()


def start_detection(args, viewer):
    detection_callback = threading.Thread(target=high_speed, args=(args,viewer,))
    detection_callback.start()


def main():
    faulthandler.enable()    
    args = parse_arguments()
    #high_speed(args)
    viewer = tv.PyViewer3D()
    viewer.init()
    
    start_detection(args, viewer)

    viewer.exit()
    glutMainLoop()

if __name__ == '__main__':
    main()
