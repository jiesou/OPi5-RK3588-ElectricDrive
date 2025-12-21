import slint

from settings_widget import bind_settings
from camera_viewport import bind_camera, camera_viewport
from shots_status_widget import bind_shots_status
from udp_frame_uploader import uploader
from api_client import api_client


def run():
    main_window = slint.loader.ui.app_window.AppWindow()
    bind_settings(main_window)
    bind_camera(main_window)
    bind_shots_status(main_window)
    
    # 启动 UDP 上传器（可选，也可以由用户在 UI 中启动）
    # uploader.start()
    
    try:
        main_window.run()
    finally:
        # 清理资源
        uploader.stop()
        camera_viewport.close()
        api_client.stop()


if __name__ == "__main__":
    run()
