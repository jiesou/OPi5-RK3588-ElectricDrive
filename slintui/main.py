import os

os.environ["SLINT_STYLE"] = "material-dark"
os.environ["SLINT_FULLSCREEN"] = "1"
os.environ["SLINT_SCALE_FACTOR"] = "1.5" # 1920 / 800
from time import sleep
import slint

from camera_service import camera_service
from evaluation import (
    bind_settings,
    bind_camera,
    bind_shots_status,
    camera_viewport,
)
from facesignin import bind_facesignin, face_signin_viewport
from deskclean import bind_deskclean, deskclean_viewport
from xiaoxin import bind_xiaoxin, xiaoxin_viewport
from udp_frame_uploader import uploader


class AppWindow(slint.loader.ui.app_window.AppWindow):
    def __init__(self):
        super().__init__()

def main():
    # uploader.start()
    camera_service.start()

    main_window = AppWindow()
    bind_settings(main_window)
    bind_camera(main_window)
    bind_shots_status(main_window)
    bind_facesignin(main_window)
    bind_deskclean(main_window)
    bind_xiaoxin(main_window)

    def activate_tab(idx: int):
        if idx == 0:
            # Evaluation
            xiaoxin_viewport.stop()
            face_signin_viewport.stop()
            deskclean_viewport.stop()
            camera_service.set_camera(0)  # 主摄像头
            camera_viewport.start()
            main_window.XiaoxinPageData.xiaoxin_running = False
            main_window.EvaluationPageData.evaluation_running = True
            main_window.FaceSigninPageData.signin_running = False
            main_window.DeskcleanPageData.deskclean_running = False
        elif idx == 1:
            # Face sign-in
            xiaoxin_viewport.stop()
            camera_viewport.stop()
            deskclean_viewport.stop()
            camera_service.set_camera(1)  # 副摄像头
            face_signin_viewport.start()
            main_window.XiaoxinPageData.xiaoxin_running = False
            main_window.EvaluationPageData.evaluation_running = False
            main_window.FaceSigninPageData.signin_running = True
            main_window.DeskcleanPageData.deskclean_running = False
        elif idx == 2:
            # Deskclean
            xiaoxin_viewport.stop()
            camera_viewport.stop()
            face_signin_viewport.stop()
            camera_service.set_camera(0)  # 主摄像头
            deskclean_viewport.start()
            main_window.XiaoxinPageData.xiaoxin_running = False
            main_window.EvaluationPageData.evaluation_running = False
            main_window.FaceSigninPageData.signin_running = False
            main_window.DeskcleanPageData.deskclean_running = True
        else:
            # Xiaoxin 智能体
            camera_viewport.stop()
            face_signin_viewport.stop()
            deskclean_viewport.stop()
            camera_service.set_camera(0)  # 主摄像头
            xiaoxin_viewport.start(main_window)
            main_window.XiaoxinPageData.xiaoxin_running = True
            main_window.EvaluationPageData.evaluation_running = False
            main_window.FaceSigninPageData.signin_running = False
            main_window.DeskcleanPageData.deskclean_running = False

    @slint.callback(global_name="AppData")
    def tab_changed(idx: int):
        activate_tab(idx)

    @slint.callback(global_name="AppData")
    def stop_app():
        main_window.hide()

    @slint.callback(global_name="AppData")
    def swap_cameras():
        camera_service.set_camera(1 - camera_service.current_idx)

    main_window.AppData.tab_changed = tab_changed
    main_window.AppData.stop_app = stop_app
    main_window.AppData.swap_cameras = swap_cameras
    activate_tab(int(main_window.AppData.current_tab))
    main_window.show()
    main_window.run()
    # 清理资源
    uploader.stop()
    xiaoxin_viewport.stop()
    camera_viewport.stop()
    face_signin_viewport.stop()
    deskclean_viewport.stop()

    camera_service.stop()

main()

# slint.run_event_loop(main())
