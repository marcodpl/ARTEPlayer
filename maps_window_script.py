import gps
import sys
import threading
import os
from PyQt5.QtCore import QUrl, QObject, pyqtSlot
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QPushButton, QVBoxLayout
from gui_engine.guis.maps import Ui_MapWindow

class GPSBridge(QObject):
    def __init__(self, view):
        super().__init__()
        self.view = view

    @pyqtSlot(float, float)
    def send_position(self, lat, lon):
        js = f"updatePosition({lat}, {lon});"
        self.view.page().runJavaScript(js)


class MapWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MapWindow()
        self.ui.setupUi(self)
        self.bridge = GPSBridge(self.ui.mapView)
        threading.Thread(target=self.gps_loop, daemon=True).start()

    def gps_loop(self):
        try:
            session = gps.gps(mode=gps.WATCH_ENABLE)
            for report in session:
                if report['class'] == 'TPV':
                    if hasattr(report, 'lat') and hasattr(report, 'lon'):
                        self.bridge.send_position(report.lat, report.lon)
        except Exception as e:
            self.bridge.send_position(45.406433, 11.876761)
            print(f"{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MapWindow()
    win.show()
    sys.exit(app.exec_())