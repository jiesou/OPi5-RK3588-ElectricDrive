from settings import stored_settings
import slint


def bind_settings(window) -> None:
    """Bind stored settings to the Slint window.

    - Initializes the window properties from stored_settings
    - Registers callback handlers that persist changes
    """

    window.server_ip = stored_settings.get_server_ip()

    @slint.callback
    def save_settings(ip: str) -> None:
        stored_settings.set_server_ip(ip)
        window.server_ip = stored_settings.get_server_ip()

    window.save_settings = save_settings
