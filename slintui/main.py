import aiohttp
import numpy as np
import slint

from settings_widget import bind_settings
from camera_viewport import bind_camera, camera_viewport
from shots_status_widget import bind_shots_status
from udp_frame_uploader import uploader
from settings import Settings, stored_settings

class AppWindow(slint.loader.ui.app_window.AppWindow):
    def __init__(self):
        super().__init__()

def main():
    # uploader.start()
    main_window = AppWindow()
    bind_settings(main_window)
    bind_camera(main_window)
    bind_shots_status(main_window)
    main_window.show()
    main_window.run()
    # 清理资源
    uploader.stop()
    camera_viewport.close()

main()

# slint.run_event_loop(main())
