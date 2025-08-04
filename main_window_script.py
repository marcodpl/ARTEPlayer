from PyQt5.QtWidgets import QMainWindow, QStackedWidget, QApplication
from home_window_script import HomeWindow as HomeGUI
import sys
"""
Main window class for ARTEPLAYER.
Thought as the "handler" for all the other windows- home, maps, etc.
Loads them all with a StackedWidget.
Connects the buttons with the relative window to be opened (connect logic).
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stacked = QStackedWidget()  # stacked widget
        self.setCentralWidget(self.stacked)
        self.home = HomeGUI()

        self.stacked.addWidget(self.home)
        self.stacked.setCurrentWidget(self.home)


app = QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
sys.exit(app.exec_())

