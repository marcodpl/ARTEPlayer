from pycaw.pycaw import IAudioEndpointVolume, AudioUtilities
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
import keyboard
from win32con import VK_MEDIA_PLAY_PAUSE


class MediaControl:
    def __init__(self):
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(self.interface, POINTER(IAudioEndpointVolume))

    def increment_volume(self, incr: int):
        """
        Raises the volume level on the master channel.
        :param incr: level in percentage (i.e. "70" for 70%)
        :raises AssertionError: if invalid overall volume level.
        """
        actual = self.volume.GetMasterVolumeLevelScalar()
        assert 0 <= actual <= 100
        assert 0 <= actual + (incr / 100) <= 100
        self.volume.SetMasterVolumeLevel(actual + (incr/100), None)

    def decrement_volume(self, decr: int):
        """
        Lowers the volume level on the master channel.
        :param decr: level in percentage (i.e. "70" for 70%)
        :raises AssertionError: if invalid overall volume level.
        """
        actual = self.volume.GetMasterVolumeLevelScalar()
        assert 0 <= actual <= 100
        assert 0 <= actual - (decr / 100) <= 100
        self.volume.SetMasterVolumeLevel(actual - (decr / 100), None)

    @staticmethod
    def toggle_play_pause(self):
        keyboard.send(VK_MEDIA_PLAY_PAUSE,
                      True,
                      True
                      )