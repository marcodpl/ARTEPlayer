import dbus
from dbus.mainloop.glib import DBusGMainLoop
from gi.repository import GLib
import threading
import time

"""
MEDIA MONITOR CLASS FOR ARTEPLAYER
Keeps track of AVRCP data streamed from the phone.
NEEDS TO BE KEPT ALIVE IN A THREAD (while True or similar), after initialization.
"""


def _normalize_mac(mac):
    """
    Takes in a mac with colons or underscores and returns a clean mac.
    :param mac: target mac, with colons or underscores
    :return: clean mac, with colons and no underscores
    """
    inter = mac.replace(":","")
    return inter.replace("_","")


def _bluez_format(mac):
    """
    Cleans mac, then returns it in the format used by bluez / dbus (underscores).
    :param mac: target mac
    :return: formatted mac, with underscores
    """
    normed_mac = _normalize_mac(mac)
    if normed_mac:
        return normed_mac[:2] + "_" + normed_mac[2:4] + "_" + normed_mac[4:6] + "_" + normed_mac[6:8] + "_" + normed_mac[8:10] + "_" + normed_mac[10:12]
    return None


class MediaMonitor:
    def __init__(self, on_track_update: callable, on_status_change: callable, on_progress_update: callable, on_device_connected: callable, on_device_disconnected: callable, mac: str = None):
        """
        Initializes the media monitor. Provides three callbacks: on_track_update, on_status_change, and on_progress_update.
        """
        DBusGMainLoop(set_as_default=True)  # dbus loop, used to add signal receivers
        self.bus = dbus.SystemBus()  # bus reference
        self.loop = GLib.MainLoop()
        self.on_track_update = on_track_update  # callable method passed in constructor. Called when track is updated (new track)
        self.on_status_change = on_status_change  # callable method passed in constructor. Called when pausing / playing song.
        self.on_progress_update = on_progress_update  # callable method passed in constructor. Called when progress is changed (both for natural time progression, or manual seeking)
        self.on_device_connected = on_device_connected  # callable method passed in constructor. Called when a device is connected
        self.on_device_disconnected = on_device_disconnected
        self.noNeedForMac = True if mac is None else False  # if no mac is passed, dbus will connect to the first player it finds.
        self._bluez_mac = _bluez_format(mac) if mac is not None else False

        self.current_track: dict[str, str] = {}  # holds [title, artist, album, duration] of current song.
        self.status = "stopped"  # holds status: stopped, playing, paused
        self.position = 0  # holds position of song
        self.duration = 0  # holds total duration of song

        self._stop = False

        self._start_monitor()

    def _start_monitor(self):
        """
        Starts listening to dbus signals.
        """
        manager = dbus.Interface(
            self.bus.get_object("org.bluez", "/"),
            "org.freedesktop.DBus.ObjectManager"
        )  # gets references of managed object by dbus, aka, bluetooth connections.
        managed_objects = manager.GetManagedObjects()

        for path, interfaces in managed_objects.items():
            if "org.bluez.MediaPlayer1" in interfaces:  # filter out else, only target mediaPlayer items
                if self.noNeedForMac or self._bluez_mac in path:  # search out the target mac provided, or target the first mediaPlayer found if None is passed
                    self.player_path = path
                    self.bus.add_signal_receiver(
                        handler_function=self._on_properties_changed,
                        signal_name="PropertiesChanged",
                        dbus_interface="org.freedesktop.DBus.Properties",
                        path=path,
                        path_keyword="path"
                    )  # sets a signal receiver. on_properties_changed handles all changes in the props of the MediaPlayer (i.e. new song, song is paused, etc.)
                    print(f"📡 Listening to MediaPlayer1 at {path}")
                    self._monitor_device_connections()  # should monitor device connections / disconnections
                    break

        threading.Thread(target=self.loop.run, daemon=True).start()  # start dbus thread, to allow for dbus signal receivers to stay alive async
        threading.Thread(target=self._progress_loop, daemon=True).start()  # starts progress_loop, for async song progress updates

    def __call__(self, *args, **kwargs):  # useless.
        return self._stop

    def _on_properties_changed(self, interface_name, changed_properties, invalidated, path=None):  # called when props are updated. Connected to PropertiesChanged signal recv.
        if interface_name != "org.bluez.MediaPlayer1":  # filter out all changes not relevant to MediaPlayer
            return

        changed = dict(changed_properties)

        if "Track" in changed:  # is track changed?
            track = changed["Track"]  # reference to changed track
            title = str(track.get("Title", ""))
            artist = str(track.get("Artist", ""))
            album = str(track.get("Album", ""))
            duration = int(track.get("Duration", 0))

            """
            Load new song, if different from current song, into self.current_track.
            """
            if (self.current_track.get("Title") != title or
                self.current_track.get("Artist") != artist):
                self.current_track = {
                    "Title": title,
                    "Artist": artist,
                    "Album": album,
                    "Duration": str(duration),
                }
                self.duration = duration
                self.position = 0
                self.on_track_update(self.current_track)  # send new song to on_track_update (callable)

        if "Status" in changed:
            self.status = str(changed["Status"])
            self.on_status_change(self.status)  # sends new status to on_status_change (callable)

        if "Position" in changed:
            self.position = int(changed["Position"])

    def _progress_loop(self):
        while not self._stop:
            if self.status == "playing":
                self.position += 1_000
                self.on_progress_update(self.position, self.duration)
            time.sleep(1)

    def _monitor_device_connections(self):
        self.bus.add_signal_receiver(
            handler_function=self._on_device_connection_change,
            signal_name="PropertiesChanged",
            dbus_interface="org.freedesktop.DBus.Properties",
            path_keyword="path",
            interface_keyword="iface",
            arg0="org.bluez.Device1"
        )

    def _on_device_connection_change(self, iface, changed, invalidated, path=None):
        print(iface, changed, invalidated, path)
        if iface != "org.bluez.Device1":
            return

        if "Connected" in changed:
            mac_path = path.split("/")[-1].replace("_", ":")
            if changed["Connected"]:
                print(f"🔌 Device connected: {mac_path}")
                if self.on_device_connected:
                    self.on_device_connected(mac_path)
            else:
                print(f"❌ Device disconnected: {mac_path}")
                if self.on_device_disconnected:
                    self.on_device_disconnected(mac_path)

    def stop(self):
        self._stop = True
        self.loop.quit()
