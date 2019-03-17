import argparse
import configparser
import os
import queue
import threading
import time
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import chainer
import cv2
import numpy as np
from PIL import Image
import pyzed.sl as sl

from predict import get_feature, get_humans_by_feature, draw_humans, create_model, load_config
from utils import parse_size

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
        mat = sl.Mat()
        runtime = sl.RuntimeParameters()
        while not self.stop_event.is_set():
            try:
                #ret_val, image = self.cap.read()
                self.cap.grab(runtime)
                self.cap.retrieve_image(mat, sl.VIEW.VIEW_LEFT)
                image = cv2.cvtColor(mat.get_data(), cv2.COLOR_BGRA2RGB)
                image = cv2.resize(image, self.insize)
                self.queue.put(image, timeout=1)
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
                image = self.cap.get()
                with chainer.using_config('autotune', True), \
                        chainer.using_config('use_ideep', 'auto'):
                    feature_map = get_feature(self.model, image.transpose(2, 0, 1).astype(np.float32))
                self.queue.put((image, feature_map), timeout=1)
            except queue.Full:
                pass
            except queue.Empty:
                pass

    def get(self):
        return self.queue.get(timeout=1)

    def stop(self):
        logger.info('{} will stop'.format(self.name))
        self.stop_event.set()


def high_speed(args):
    config = load_config(args)
    dataset_type = config.get('dataset', 'type')
    detection_thresh = config.getfloat('predict', 'detection_thresh')
    min_num_keypoints = config.getint('predict', 'min_num_keypoints')
    model = create_model(args, config)
    svo_file_path = config.get('zed', 'svo_file_path')

    if os.path.exists('mask.png'):
        mask = Image.open('mask.png')
        mask = mask.resize((200, 200))
    else:
        mask = None

    init_cap_params = sl.InitParameters()
    if svo_file_path:
        print("Loading SVO file " + svo_file_path)
        init_cap_params.svo_input_filename = svo_file_path
    init_cap_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720
    cap = sl.Camera()
    if not cap.is_opened():
        print("Opening ZED Camera...")
    status = cap.open(init_cap_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    #logger.info('camera will capture {} FPS'.format(cap.get(cv2.CAP_PROP_FPS)))

    capture = Capture(cap, model.insize)
    predictor = Predictor(model=model, cap=capture)

    capture.start()
    predictor.start()

    fps_time = 0
    degree = 0

    main_event = threading.Event()

    try:
        while not main_event.is_set() and cap.is_opened():
            degree += 5
            degree = degree % 360
            try:
                image, feature_map = predictor.get()
                humans = get_humans_by_feature(
                    model,
                    feature_map,
                    detection_thresh,
                    min_num_keypoints
                )
            except queue.Empty:
                continue
            except Exception:
                break
            pilImg = Image.fromarray(image)
            pilImg = draw_humans(
                model.keypoint_names,
                model.edges,
                pilImg,
                humans,
                mask=mask.rotate(degree) if mask else None,
                visbbox=config.getboolean('predict', 'visbbox'),
            )
            img_with_humans = cv2.cvtColor(np.asarray(pilImg), cv2.COLOR_RGB2BGR)
            msg = 'GPU ON' if chainer.backends.cuda.available else 'GPU OFF'
            msg += ' ' + config.get('model_param', 'model_name')
            cv2.putText(img_with_humans, 'FPS: %f' % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            img_with_humans = cv2.resize(img_with_humans, (3 * model.insize[0], 3 * model.insize[1]))
            cv2.imshow('Pose Proposal Network' + msg, img_with_humans)
            fps_time = time.time()
            # press Esc to exit
            if cv2.waitKey(1) == 27:
                main_event.set()
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        main_event.set()

    capture.stop()
    predictor.stop()

    capture.join()
    predictor.join()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path/to/model', type=str)
    return parser.parse_args()


def main():
    args = parse_arguments()
    high_speed(args)

if __name__ == '__main__':
    main()
