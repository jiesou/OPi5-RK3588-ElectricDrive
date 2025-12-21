import slint

from settings_controller import bind_settings
from camera_viewport import bind_camera
from shots_status import bind_shots_status


def run():
    main_window = slint.loader.ui.app_window.AppWindow()
    bind_settings(main_window)
    bind_camera(main_window)
    bind_shots_status(main_window)
    main_window.run()


if __name__ == "__main__":
    run()
