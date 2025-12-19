import slint
import sys
import os


class MainWindow(slint.loader.ui.app_window.AppWindow):
    def init():
        pass


main_window = MainWindow()
main_window.show()
main_window.run()
