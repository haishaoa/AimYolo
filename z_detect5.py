# -*- coding = utf-8 -*-
# @Time : 2022/10/28 17:56
# @Author : cxk
# @File : z_detect5.py
# @Software : PyCharm


import sys
import ctypes
import signal

import argparse

from mss import mss

from z_captureScreen import capScreen

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.utils import *

if sys.platform == 'win32':
    import win32con
    import win32api
    from z_ctypes import SendInput, mouse_input


def screen_position(center, bounding_box, image_shape):
    """Map capture pixels to desktop coordinates, including Retina scaling."""
    height, width = image_shape[:2]
    return (
        round(bounding_box['left'] + center[0] * bounding_box['width'] / width),
        round(bounding_box['top'] + center[1] * bounding_box['height'] / height),
    )


def pre_process(img0, img_sz, half, device):
    """
    img0: from capScreen(), format: HWC, BGR
    """
    # padding resize
    img = letterbox(img0, new_shape=img_sz)[0]
    # convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img)

    # preprocess
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0-255 to 0.0-1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


@torch.no_grad()
def inference_img(img, model, augment, conf_thres, iou_thres, classes, agnostic):
    """
    推理，模型参数，...
    """
    # inference
    pred = model(img, augment=augment)[0]
    # apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic)
    return pred


def calculate_position(xyxy):
    """
    计算中心坐标
    """
    c1, c2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
    # print('\n左上点坐标:(' + str(c1[0]) + ',' + str(c1[1]) + '), 右上点坐标:(' + str(c2[0]) + ',' + str(
    #     c1[1]) + ')')
    # print('左下点坐标:(' + str(c1[0]) + ',' + str(c2[1]) + '), 右下点坐标:(' + str(c2[0]) + ',' + str(
    #     c2[1]) + ')')
    # print("中心点的坐标为：(" + str((c2[0] - c1[0]) / 2 + c1[0]) + "," + str(
    #     (c2[1] - c1[1]) / 2 + c1[1]) + ")")
    center_x = int((c2[0] - c1[0]) / 2 + c1[0])
    center_y = int((c2[1] - c1[1]) / 2 + c1[1])
    return center_x, center_y


def view_imgs(img0):
    """
    弹窗展示结果，press q to quit
    """
    cv2.imshow('ws demo', img0)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        exit(0)


def move_mouse(mouse_pynput, aim_persons_center):
    """
    move mouse，暂定为欧氏距离
    指针移动到距离当前鼠标位置最短的坐标处
    """
    if aim_persons_center:
        # 当前鼠标位置
        if sys.platform == 'darwin':
            # Quartz uses the same desktop coordinates as MSS monitor bounds.
            # Avoid mixing AppKit points and Retina pixels in pynput's getter.
            import Quartz
            current_x, current_y = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        else:
            current_x, current_y = mouse_pynput.position
        # 距离当前位置最近的目标中心点and距离
        best_position = None
        for aim_person in aim_persons_center:
            # aim_person is a list
            dist = ((aim_person[0] - current_x) ** 2 + (aim_person[1] - current_y) ** 2) ** .5
            if not best_position:
                best_position = (aim_person, dist)
            else:
                _, old_dist = best_position
                if dist < old_dist:
                    best_position = (aim_person, dist)

        if sys.platform == 'win32':
            tx = int(best_position[0][0] / win32api.GetSystemMetrics(0) * 65535)
            ty = int(best_position[0][1] / win32api.GetSystemMetrics(1) * 65535)
            SendInput(mouse_input(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, tx, ty))
        else:
            mouse_pynput.position = tuple(best_position[0])


class AimYolo:

    def __init__(self, opt):
        self.weights = opt.weights
        self.img_size = opt.img_size
        self.conf_thres = opt.conf_thres
        self.iou_thres = opt.iou_thres
        self.view_img = opt.view_img
        self.classes = opt.classes
        self.agnostic_nms = opt.agnostic_nms
        self.augment = opt.augment
        self.no_mouse = opt.no_mouse

        # load model
        self.device = torch_utils.select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.model = self.model.to(self.device)
        self.img_size = check_img_size(self.img_size, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # name and color
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    @torch.no_grad()
    def check(self):
        """Exercise preprocessing, inference and NMS without screen/mouse access."""
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        image = pre_process(image, self.img_size, self.half, self.device)
        prediction = inference_img(image, self.model, self.augment, self.conf_thres,
                                   self.iou_thres, self.classes, self.agnostic_nms)
        assert len(prediction) == 1
        # A blank image may produce no candidates, so also exercise NMS explicitly.
        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], device=self.device, dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8], device=self.device)
        assert torchvision.ops.nms(boxes, scores, 0.5).numel() == 1
        # Exercise a nonempty detection through the same mutation used by run().
        # inference_img must return normal tensors that remain writable by callers.
        sample = torch.tensor([[[320., 192., 64., 64., 0.9, 0.9]]], device=self.device)
        sample_model = lambda batch, augment=False: (sample.clone(),)
        det = inference_img(image, sample_model, False, 0.4, 0.5, None, False)[0]
        assert det is not None and len(det) == 1
        assert not torch.is_inference(det)
        det[:, :4] = scale_coords(image.shape[2:], det[:, :4], (360, 640, 3)).round()
        assert torch.isfinite(det).all()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        print(f'Runtime check passed: torch={torch.__version__}, device={self.device}, '
              f'FP16={self.half}, input={tuple(image.shape)}')

    @torch.no_grad()
    def run(self):
        if sys.platform == 'win32':
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        with mss() as sct:
            if sys.platform == 'win32':
                self.bounding_box = {'left': 0, 'top': 0, 'width': win32api.GetSystemMetrics(0),
                                     'height': win32api.GetSystemMetrics(1)}
            else:
                # Prefer the display containing the desktop origin (primary display).
                monitors = sct.monitors[1:]
                if not monitors:
                    raise RuntimeError('No screen available for capture.')
                self.bounding_box = dict(next((monitor for monitor in monitors
                                               if monitor['left'] == 0 and monitor['top'] == 0),
                                              monitors[0]))
            try:
                self._run(sct)
            finally:
                if self.view_img:
                    cv2.destroyAllWindows()

    def _run(self, sct):

        img_sz = self.img_size

        if self.view_img:
            cv2.namedWindow('ws demo', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            width = min(960, self.bounding_box['width'])
            height = max(1, round(width * self.bounding_box['height'] / self.bounding_box['width']))
            cv2.resizeWindow('ws demo', width, height)

        # warm up
        img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None

        # mss and capture screen
        print("The mss object created.")
        # mouse control
        mouse_control = None
        if not self.no_mouse:
            from pynput import mouse
            mouse_control = mouse.Controller()
            print("The mouse controller created.")

        # 若只在循环中的if前定义空列表，在检测不到目标时，for循环没有对其进行初始化，会报错
        aim_persons_center = []
        aim_persons_center_head = []
        while True:
            img0 = capScreen(sct, self.bounding_box)  # HWC and BGR

            img = pre_process(img0=img0, img_sz=img_sz, half=self.half, device=self.device)

            t1 = torch_utils.time_synchronized()
            pred = inference_img(img=img, model=self.model, augment=self.augment, conf_thres=self.conf_thres,
                                 iou_thres=self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # process detections
            det = pred[0]

            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string

            if det is not None and len(det):
                # rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                # write results
                aim_persons_center = []
                aim_persons_center_head = []
                for *xyxy, conf, cls in det:

                    # Add bbox to image
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=3)
                    center_x, center_y = calculate_position(xyxy)
                    aim_persons_center.append([center_x, center_y])
                    if int(cls) == 2 or int(cls) == 3:
                        aim_persons_center_head.append(screen_position(
                            (center_x, center_y), self.bounding_box, img0.shape))

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # 计算中心坐标，若有多个目标，只取其中一个
            # 取距离最近的一个
            if mouse_control is not None:
                move_mouse(mouse_control, aim_persons_center_head)
            aim_persons_center = []
            aim_persons_center_head = []

            # view img
            if self.view_img:
                view_imgs(img0=img0)

        # End ------------------------------------------


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best_200.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--device', default='', help='CUDA device, e.g. 0, or cpu (default: auto)')
    parser.add_argument('--check', action='store_true', help='check inference and NMS without screen capture or mouse movement')
    parser.add_argument('--no-mouse', action='store_true', help='detect the screen without moving the mouse')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    opt = parseArgs()
    print(opt)

    aim_yolo = AimYolo(opt)
    print('The AimYolo Object Created.')

    if opt.check:
        aim_yolo.check()
    else:
        aim_yolo.run()

