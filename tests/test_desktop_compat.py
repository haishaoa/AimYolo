import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import z_detect5 as app


class DesktopCompatibilityTests(unittest.TestCase):
    def test_retina_capture_uses_desktop_points(self):
        bounds = dict(left=0, top=0, width=1440, height=900)
        self.assertEqual(app.screen_position((1440, 900), bounds, (1800, 2880, 3)),
                         (720, 450))

    def test_monitor_offset_is_included(self):
        bounds = dict(left=-1920, top=-200, width=1920, height=1080)
        self.assertEqual(app.screen_position((960, 540), bounds, (1080, 1920, 3)),
                         (-960, 340))

    def test_unscaled_primary_display(self):
        bounds = dict(left=0, top=0, width=1920, height=1080)
        self.assertEqual(app.screen_position((960, 540), bounds, (1080, 1920, 3)),
                         (960, 540))

    def test_mac_selects_nearest_using_quartz_coordinates(self):
        controller = SimpleNamespace(position=(1000, 1000))
        quartz = SimpleNamespace(CGEventCreate=Mock(return_value='event'),
                                 CGEventGetLocation=Mock(return_value=(100, 100)))
        with patch.object(app.sys, 'platform', 'darwin'), patch.dict('sys.modules', Quartz=quartz):
            app.move_mouse(controller, [(110, 100), (900, 900)])
        self.assertEqual(controller.position, (110, 100))
        quartz.CGEventGetLocation.assert_called_once_with('event')

    def test_windows_preserves_sendinput_coordinates(self):
        send_input = Mock()
        windows = dict(win32api=SimpleNamespace(GetSystemMetrics=lambda axis: (1920, 1080)[axis]),
                       win32con=SimpleNamespace(MOUSEEVENTF_MOVE=1, MOUSEEVENTF_ABSOLUTE=0x8000),
                       mouse_input=lambda *args: args, SendInput=send_input)
        controller = SimpleNamespace(position=(100, 100))
        with patch.object(app.sys, 'platform', 'win32'), patch.dict(app.__dict__, windows):
            app.move_mouse(controller, [(960, 540), (1900, 1000)])
        send_input.assert_called_once_with((0x8001, 32767, 32767))

    def test_no_detections_do_not_access_mouse(self):
        app.move_mouse(None, [])

    def test_no_mouse_can_process_a_frame_without_pynput(self):
        runner = app.AimYolo.__new__(app.AimYolo)
        runner.img_size = 640
        runner.device = app.torch.device('cpu')
        runner.half = False
        runner.view_img = False
        runner.no_mouse = True
        runner.model = Mock()
        runner.augment = runner.agnostic_nms = False
        runner.conf_thres, runner.iou_thres, runner.classes = 0.4, 0.5, None
        frame = app.np.zeros((360, 640, 3), dtype=app.np.uint8)
        with patch.dict('sys.modules', pynput=None), \
                patch.object(app, 'capScreen', side_effect=[frame, StopIteration]), \
                patch.object(app, 'inference_img', return_value=[None]) as inference, \
                patch('builtins.print'):
            runner.bounding_box = dict(left=0, top=0, width=640, height=360)
            with self.assertRaises(StopIteration):
                runner._run(Mock())
        inference.assert_called_once()

    def test_run_closes_capture_and_preview_on_failure(self):
        runner = app.AimYolo.__new__(app.AimYolo)
        runner.view_img = True
        runner._run = Mock(side_effect=RuntimeError('capture failed'))
        primary = dict(left=0, top=0, width=1440, height=900)
        secondary = dict(left=-1920, top=0, width=1920, height=1080)
        capture = Mock(monitors=[{}, secondary, primary])
        context = Mock()
        context.__enter__ = Mock(return_value=capture)
        context.__exit__ = Mock(return_value=False)
        with patch.object(app.sys, 'platform', 'darwin'), patch.object(app, 'mss', return_value=context), \
                patch.object(app.cv2, 'destroyAllWindows') as close_preview:
            with self.assertRaisesRegex(RuntimeError, 'capture failed'):
                runner.run()
        self.assertEqual(runner.bounding_box, primary)
        runner._run.assert_called_once_with(capture)
        context.__exit__.assert_called_once()
        close_preview.assert_called_once()


if __name__ == '__main__':
    unittest.main()
