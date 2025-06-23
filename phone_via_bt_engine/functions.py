import stat
import operator


class MoreThanOneDefaultPhoneError(Exception):
    def __init__(self, message):
        super().__init__(message)


import os
import subprocess

class BtPhone:
    def __init__(self, name, mac, date, is_default: bool = False):
        self.name = name
        self.mac = mac  # Format: XX:XX:XX:XX:XX:XX
        self.date = date
        self.is_default = is_default


class Functions:
    def __init__(self):
        self.phones: list[BtPhone] = []
        self.file_path = "data/phone_coords.txt"

        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                for line in f:
                    name, mac, date, is_default = line.strip().split(";")
                    self.phones.append(BtPhone(name, mac, date, is_default.lower() == "true"))

    def save_phones(self):
        with open(self.file_path, "w") as f:
            for phone in self.phones:
                f.write(f"{phone.name};{phone.mac};{phone.date};{phone.is_default}\n")

    def add_phone(self, name, mac, date, set_default=False):
        self.phones.append(BtPhone(name, mac, date, set_default))
        if set_default:
            for p in self.phones:
                p.is_default = (p.mac == mac)
        self.save_phones()

    def delete_phone(self, mac):
        self.phones = [p for p in self.phones if p.mac != mac]
        self.save_phones()

    @staticmethod
    def pair_and_trust_phone(mac):
        """
        Pairs and trusts a Bluetooth phone using bluetoothctl.
        """
        print(f"📲 Pairing with {mac}...")
        commands = f"""
            power on
            agent on
            default-agent
            pair {mac}
            trust {mac}
            quit
        """
        subprocess.run(['bluetoothctl'], input=commands.encode(), stdout=subprocess.PIPE)
        print("✅ Paired and trusted.")

    def connect_to_phone(self, mac: str):
        """
        Connects to a Bluetooth phone and enables audio routing to AUX.
        """
        print(f"🔗 Connecting to {mac}...")
        os.system("pulseaudio --start")
        os.system("pactl load-module module-bluetooth-discover")
        os.system(f"bluetoothctl connect {mac}")
        self.set_audio_output_to_aux()
        print("🔊 Connected and audio routed to AUX.")

    @staticmethod
    def set_audio_output_to_aux():
        """
        Forces audio to play through the Raspberry Pi 3.5mm jack.
        """
        os.system("amixer cset numid=3 1")
        print("🔈 Output set to 3.5mm AUX")

    @staticmethod
    def send_media_command(mac: str, command: str):
        """
        Sends a media control command to the connected phone (play, pause, next, previous).
        """
        valid_cmds = {"play", "pause", "next", "previous"}
        if command.lower() not in valid_cmds:
            print(f"⚠️ Invalid command: {command}")
            return

        mac_dbus = mac.replace(":", "_")
        os.system(
            f"dbus-send --system --dest=org.bluez --print-reply "
            f"/org/bluez/hci0/dev_{mac_dbus}/player0 "
            f"org.bluez.MediaPlayer1.{command}"
        )
        print(f"🎵 Sent '{command}' command to {mac}")





