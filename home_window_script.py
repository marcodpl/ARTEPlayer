"""
Author: @bigmarco\n
Home window class for ARTEPLAYER.
Loads the UI and handles all the logic.
Is loaded by MainWindow as part of stacked widget logic.
TAB BUTTONS should not be touched. They are assigned and implemented by MainWindow for window change.
"""
from io import BytesIO
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QWidget
from gui_engine.guis.home import Ui_MediaCenterWindow
import sys
from bluetooth_engine.functions import *
from bluetooth_engine.media_monitor import MediaMonitor
from bluetooth_engine.helpers.timing_marco import *


class HomeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MediaCenterWindow()
        self.ui.setupUi(self)  # Loads the UI assigned, from the guis folder. Initializes it.
        self.ui.playPauseButton.clicked.connect(self.on_play_pause_clicked)  # Assignment of buttons.
        self.status = "playing"  # status is used to control MediaMonitor. Defaults on loading to playing.
        self.media_monitor = MediaMonitor(on_track_update=self.on_track_update,
                                          on_status_change=self.on_status_change,
                                          on_progress_update=self.on_progress_update,
                                          on_device_connected=self.on_device_connected,
                                          on_device_disconnected=self.on_device_disconnected,
                                          mac="DC5285E30AB6")  # only because of debug purposes.
        self.f = Functions()

        self.ui.nextButton.clicked.connect(self.on_next_clicked)
        self.ui.prevButton.clicked.connect(self.on_prev_clicked)

    def on_track_update(self, data: dict[str, str]):
        """
        Called when media_monitor receives a track update.
        :param data: data dictionary with track information.
        """
        """
        Sets the track information on the UI.
        """
        self.ui.trackTitle.setText(data["Title"])
        self.ui.artistName.setText(data["Artist"])
        self.ui.albumName.setText(data["Album"])
        self.ui.totalTime.setText(str(Pointed(data["Duration"])))
        self.ui.progressBar.setRange(0, int(data["Duration"]))
        url = fetch_album_art_url(data["Title"], data["Artist"], data["Album"])
        response = requests.get(url)
        response.raise_for_status()
        image_data = BytesIO(response.content)
        pixmap = QPixmap()
        if not pixmap.loadFromData(image_data.getvalue()):
            raise IOError("Failed to load image data into QPixmap.")
        self.ui.label_2.setPixmap(pixmap)

    def on_status_change(self, status):
        print(f"stat {status}")
        self.status = str(status)

    def on_progress_update(self, progress, total):
        print(f"{progress}//{total}")
        self.ui.elapsedTime.setText(str(Pointed(progress)))
        self.ui.totalTime.setText(str(Pointed(total)))
        self.ui.progressBar.setValue(int(progress))

    def on_play_pause_clicked(self):
        print("playck")
        if "playing" in self.status:
            self.f.send_media_command("pause")
        elif "paused" in self.status:
            self.f.send_media_command("play")

    def on_prev_clicked(self):
        self.f.send_media_command("previous")

    def on_next_clicked(self):
        self.f.send_media_command("next")

    def on_device_connected(self, mac_path):
        print("connected: {}".format(mac_path))
        name = self.f.get_device_name_from_bluetoothctl_info(mac_path)
        self.ui.deviceLabel.setText(name)

    def on_device_disconnected(self):
        self.ui.deviceLabel.setText("Disconnected")


