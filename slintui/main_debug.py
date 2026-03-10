import slint
import sys
import os

os.environ["SLINT_STYLE"] = "material-dark"

class MainWindow(slint.loader.ui.app_window.AppWindow):
    pass

main_window = MainWindow()
main_window.show()
main_window.run()
