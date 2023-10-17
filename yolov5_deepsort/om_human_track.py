# -*- coding: utf-8 -*-
import time
import sys
import cv2
import numpy as np
from video_capture.CamLoader_v2 import CamLoader, CamLoader_Q
from datetime import datetime
from multiprocessing import Process
import argparse
# from Detection.Utils import ResizePadding
import random

sys.path.append("Models/")
from Models.deep_sort import DeepSort

import acl
sys.path.append("../acllite/")
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource

def ResizePadding(height, width):
    desized_size = (height, width)

    def resizePadding(image, **kwargs):
        old_size = image.shape[:2]
        max_size_idx = old_size.index(max(old_size))
        ratio = float(desized_size[max_size_idx]) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        if new_size > desized_size:
            min_size_idx = old_size.index(min(old_size))
            ratio = float(desized_size[min_size_idx]) / min(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])

        image = cv2.resize(image, (new_size[1], new_size[0]))
        delta_w = desized_size[1] - new_size[1]
        delta_h = desized_size[0] - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
        return image
    return resizePadding

class human_track():
    def __init__(self):
        # 设置目标尺寸和输出比例
        self.target_size = (960, 1080)
        self.output_ratio = 1.0
        self.resize_fn = ResizePadding(self.target_size[0], self.target_size[1])
        # 加载YOLO模型
        self.detection_model = AclLiteModel('yolov5s_nms_bs1.310b1.7.0.rc1.alpha003.om')
        # 加载DeepSort追踪器
        self.sort = DeepSort('ckpt1_bs1-100.310b1.7.0.rc1.alpha003.om', max_iou_distance=0.7, max_age=50, nn_budget=50)
        # 初始化追踪路径字典，颜色字典和最后更新时间字典
        self.track_dict = {}
        self.color_dict = {}
        self.track_last_update = {}
        # 初始化帧计数器
        self.frame_count = 0
        # 设置最大追踪路径长度
        self.max_track_length = 20

    def __call__(self, cam_source, save_out=""):
        # ==== video output setup========
        if save_out != "":
            output_size = (int(self.output_ratio * self.target_size[1]), int(self.output_ratio * self.target_size[0]))
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_out, codec, 20, output_size)

        # ===============================
        retry_t = 0
        fps_time = 0
        # cam = CamLoader(cam_source, preprocess=self.preproc).start()  # 输入为视频流时使用此函数
        cam = CamLoader_Q(cam_source, queue_size=-1, preprocess=self.preproc).start()
        while cam.grabbed():
            # Get frame
            # =====================================================================================
            frame = cam.getitem()

            # When retry over 3 times, send the exception signal and stop the program
            # =====================================================================================
            if frame is None:
                if retry_t <= 3:
                    print(self.output_time_now(), "--retrying connection--", retry_t)
                    retry_t += 1
                    cam.stop()
                    try:
                        cam = CamLoader(cam_source, preprocess=self.preproc).start()
                    except:
                        pass
                    continue
                else:
                    print(self.output_time_now(), "--Camera connection stopped--")
                    break

            # get yolo detection
            # =====================================================================================
            # 图像预处理
            img0=frame.copy()
            img_size=(640,640)
            img, ratio, pad = self.letterbox(img0)  # padding resize
            imgh, imgw = img0.shape[:2]
            img_info = np.array([img_size[0], img_size[1], imgh, imgw], dtype=np.float16)
            img_info=np.stack(img_info, axis=0)

            img=img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR tp RGB
            img=img.astype(dtype=np.float16)
            img/=255.0
            img=np.ascontiguousarray(img)
            img=np.stack(img, axis=0)
            # om推理
            outputs = self.detection_model.execute([img, img_info])
            # 推理后处理
            box_out = outputs[0]
            box_out_num = outputs[1]
            idx = 0
            num_det = int(box_out_num[idx][0])
            boxout = box_out[idx][:num_det * 6].reshape(6, -1).transpose().astype(np.float32)  # 6xN -> Nx6
            # 过滤出只含human的结果
            result = [i for i in boxout if i[-1]==0]
            result = np.array(result)

            # Tracker
            # =====================================================================================
            rst = result
            if rst.shape[0] != 0:
                rst2 = np.stack(((rst[:, 0] + rst[:, 2]) / 2, (rst[:, 1] + rst[:, 3]) / 2,
                                 (rst[:, 2] - rst[:, 0]), (rst[:, 3] - rst[:, 1])), axis=1)
                conf, human_cls = rst[:, 4], rst[:, 5]
                track_box, track = self.sort.update(rst2, conf, human_cls, frame)
            else:
                track_box = []

            # Result plot
            # =====================================================================================
            for track in track_box:
                # Calculate the center point of the box
                x1, y1, x2, y2, _, track_id = track
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                # Update tracks
                self.update_track(track_id, center)

                # Check if there are any tracks that have not been updated for a long time and remove them
                self.remove_old_tracks()

            self.frame_count += 1  # Increment the frame counter
            self.draw_tracks_and_boxes(frame, track_box)
            frame = cv2.putText(frame, 'FPS: %f' % (1.0 / (time.time() - fps_time)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)
            fps_time = time.time()
            # cv2.imshow("1", frame)
            # cv2.waitKey(15)

            if save_out != "":
                writer.write(frame)
            
        if save_out != "":
            writer.release()
        # cv2.destroyAllWindows()
        cam.stop()

    # 返回当前时间
    def output_time_now(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")

    # 对视频帧进行预处理
    def preproc(self, image):
        """preprocess function for CameraLoader.
        """
        image = self.resize_fn(image)
        return image

    # 更新追踪路径
    def update_track(self, track_id, center):
        # If the track_id is not in the dictionary, create a new track
        if track_id not in self.track_dict:
            self.create_new_track(track_id)
        self.track_dict[track_id].append(center)
        self.track_last_update[track_id] = self.frame_count
        # If the track's length exceed 20, remove the oldest point
        if len(self.track_dict[track_id]) > 20:
            self.track_dict[track_id].pop(0)

    # 创建新追踪路径
    def create_new_track(self, track_id):
        self.track_dict[track_id] = []
        # Also assign a color to this new track
        self.color_dict[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # 移除过时追踪路径
    def remove_old_tracks(self):
        outdated_tracks = [track_id for track_id, last_update in self.track_last_update.items()
                           if self.frame_count - last_update > 30]
        for track_id in outdated_tracks:
            del self.track_dict[track_id]
            del self.color_dict[track_id]
            del self.track_last_update[track_id]

    # 在视频帧上画出追踪路径和人体边界框
    def draw_tracks_and_boxes(self, frame, track_box):
        for track_id, track in self.track_dict.items():
            color = self.color_dict[track_id]  # Use the color assigned to this track
            # Draw the track
            for point in track:
                cv2.circle(frame, point, 4, color, -1)  # Draw a small circle at each point

        for track in track_box:
            x1, y1, x2, y2, _, track_id = track
            color = self.color_dict[track_id]  # Use the color assigned to this track
            # Draw the bounding box for each detection
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # yolov5预处理
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)


def detect_fun(source_str, save_out):
    acl_resource = AclLiteResource()
    acl_resource.init()
    human_track_modle = human_track()
    human_tracker = human_track_modle(cam_source=source_str, save_out=save_out)


if __name__ == '__main__':
    # 解析命令行参数
    par = argparse.ArgumentParser(description='Human Tracking Program.')
    par.add_argument("--source", default="video.mp4", help="stream source")
    par.add_argument("--save_out", default="./result.mp4", help="stream source")

    args = par.parse_args()
    source_str = args.source
    save_out = args.save_out
    print('Start Detection Model Program')
    print("Catch stream from:", source_str)

    # 创建一个新进程来运行detect_fun函数
    p = Process(target=detect_fun, args=(source_str, save_out,))
    p.start()
    p.join()
