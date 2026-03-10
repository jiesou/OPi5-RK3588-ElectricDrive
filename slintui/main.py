import os

os.environ["SLINT_STYLE"] = "material-dark"
os.environ["SLINT_FULLSCREEN"] = "1"
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

    def activate_tab(idx: int):
        if idx == 0:
            # Evaluation
            face_signin_viewport.stop()
            deskclean_viewport.stop()
            camera_viewport.start()
            main_window.EvaluationPageData.evaluation_running = True
            main_window.FaceSigninPageData.signin_running = False
            main_window.DeskcleanPageData.deskclean_running = False
        elif idx == 1:
            # Face sign-in
            camera_viewport.stop()
            deskclean_viewport.stop()
            face_signin_viewport.start()
            main_window.EvaluationPageData.evaluation_running = False
            main_window.FaceSigninPageData.signin_running = True
            main_window.DeskcleanPageData.deskclean_running = False
        else:
            # Deskclean
            camera_viewport.stop()
            face_signin_viewport.stop()
            deskclean_viewport.start()
            main_window.EvaluationPageData.evaluation_running = False
            main_window.FaceSigninPageData.signin_running = False
            main_window.DeskcleanPageData.deskclean_running = True

    @slint.callback
    def tab_changed(idx: int):
        activate_tab(idx)

    @slint.callback
    def stop_app():
        main_window.hide()

    main_window.AppData.tab_changed = tab_changed
    main_window.AppData.stop_app = stop_app
    activate_tab(int(main_window.AppData.current_tab))
    main_window.show()
    main_window.run()
    # 清理资源
    uploader.stop()
    camera_viewport.stop()
    face_signin_viewport.stop()
    deskclean_viewport.stop()

    camera_service.stop()

main()

# slint.run_event_loop(main())
