"""
Bluetooth functionality implementation for ARTEPlayer.
Phones are represented by BtPhone objects. They hold name, MAC address, and date added, as well as a flag indicating whether
it is the default phone.
Default phone: represents the phone that is first focused when the app starts. Also, it is used as default phone
when focused phone is disconnected or removed.
 - it is stored inside data/default_mac.txt
Focused phone: represents the phone that is currently focused. All bluetooth operations which require a target phone
are performed on this phone. It is not stored and is session - dependent. It is set to the default phone when the app
starts.
 - it is set to the default phone when focused phone is disconnected or removed.
Phones: represents a list of phones that are known to the app.
 - they are loaded from the system when the app starts.
 - they are updated when a new phone is paired or connected.
 - they are updated when a phone is disconnected or removed.
 - they are updated when a phone is renamed.
 - they are updated when a phone is removed from the system.
"""

import subprocess
import time
import re
import os
from datetime import datetime
import requests

BLUETOOTH_SET_REMOTE_VOLUME_IS_COMPLETED = False


def fetch_album_art_url(title, artist, album=""):
    """
    Fetches the album art URL from iTunes. Requires an internet connection.
    :param title: title of the song
    :param artist: artist of the song
    :param album: (optional) album of the song for better results
    :return: pixmap URL of the album art, or None if no album art was found
    """
    query = f"{title} {artist} {album}"
    url = f"https://itunes.apple.com/search?term={requests.utils.quote(query)}&media=music&limit=1"
    response = requests.get(url)
    data = response.json()

    if data["resultCount"] > 0:
        return data["results"][0]["artworkUrl100"].replace("100x100bb", "500x500bb")  # higher res
    return None


# DEBUG
def checked(func: callable):
    """
    Debug wrapper for functions. Prints errors to console.
    :param func: target callable
    :return: whatever func returns, or None if an error occurred
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[ERROR]: {e}")
            return None
    return wrapper


class BtPhone(object):
    def __init__(self, name, mac, date_added=None, is_default=False):
        """
        A class representing a Bluetooth phone.
        :param name: phone name
        :param mac: phone physical address, in whatever format it is stored on the phone, e.g. "00:11:22:33:44:55".
        :param date_added: when phone was first connected to the system, in datetime format.
        :param is_default: if the phone is the default phone for the system.
        """
        self.name = name
        self.mac = mac.replace(':', '').replace("_", "").upper()
        self.date_added = date_added if date_added else datetime.now()
        self.is_default = is_default
        self.__doc__ = self.__repr__.__doc__

    def __repr__(self):
        return (f"BtPhone(name='{self.name}', mac='{self.mac}', "
                f"added='{self.date_added.strftime('%Y-%m-%d %H:%M:%S')}', default={self.is_default})")

    def to_dict(self):
        return {
            'name'      : self.name,
            'mac'       : self.mac,
            'date_added': self.date_added.isoformat(),
            'is_default': self.is_default
        }


class Functions(object):
    def __init__(self):
        print("[INIT]: Initializing functions...")
        with open(os.path.dirname(os.path.realpath(__file__))+"/data/default_mac.txt", "r+") as f:
            default_phone_mac = f.read().strip()

        self.default_phone_mac = self._normalize_mac(default_phone_mac)
        self.phones = []
        print(
            f"[INIT]: Default Target MAC: {self.default_phone_mac if self.default_phone_mac else 'Not specified'}")

        self._load_known_phones_from_system()
        print("[INIT]: Known phones loaded from system.")
        self._check_default_phone_registration()
        print("[INIT]: Default phone integrity checked.")
        self.focused_phone = next((phont for phont in self.phones if phont.is_default), None)
        print(f"[INIT]: Focused phone: {self.focused_phone.name if self.focused_phone else 'None'}")
        print("[INIT][END]: Functions initialized.")

    @staticmethod
    def _normalize_mac(mac: str):
        if mac:
            mac = mac.replace('_','')
            return mac.replace(':', '').upper()
        return None

    def _btctlformat(self, mac):
        """
        Returns a mac address in the format used by bluetoothctl.
        :param mac: mac to be converted
        :return: bluetoothctl formatted mac address
        """
        normed_mac = self._normalize_mac(mac)
        if normed_mac:
            return normed_mac[:2] + ":" + normed_mac[2:4] + ":" + normed_mac[4:6] + ":" + normed_mac[6:8] + ":" + normed_mac[8:10] + ":" + normed_mac[10:12]
        return None

    def _bluezformat(self, mac):
        normed_mac = self._normalize_mac(mac)
        if normed_mac:
            return normed_mac[:2] + "_" + normed_mac[2:4] + "_" + normed_mac[4:6] + "_" + normed_mac[6:8] + "_" + normed_mac[8:10] + "_" + normed_mac[10:12]
        return None

    @staticmethod
    def _run_command(command):
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"[RUNCMD]: Command failed: {e.cmd}") from e
        except FileNotFoundError:
            raise FileNotFoundError(f"[RUNCMD]: Command not found: {command.split()[0]}")

    @staticmethod
    def set_audio_output_to_aux():
        """
        Forces audio to play through the Raspberry Pi 3.5mm jack.
        """
        os.system(" amixer cset numid=3 1")
        print("[AUDIO]: Output set to 3.5mm AUX")

    @checked
    def get_bluetooth_sink_input_info(self, mac=None):
        """
        Obtains information about the active Bluetooth A2DP sink input.
        :return: JSON object with information about the active sink input, or None if no active sink input is found
        """
        current_target_mac = mac if mac else self.focused_phone.mac if self.focused_phone else None
        try:
            output = self._run_command("pactl list sink-inputs")
        except RuntimeError:
            return None

        sink_inputs = re.split(r'Sink Input #(\d+)', output)

        for i in range(1, len(sink_inputs), 2):
            input_id = int(sink_inputs[i])
            block = sink_inputs[i + 1]

            if "api.bluez5.profile = \"a2dp-source\"" in block:
                mac_match = re.search(r'api\.bluez5\.address = "([0-9A-Fa-f:]+)"', block)
                if mac_match:
                    stream_mac = self._normalize_mac(mac_match.group(1))
                    print(f"debug {stream_mac}, {current_target_mac}")
                    if current_target_mac and stream_mac != current_target_mac:
                        continue

                    media_name_match = re.search(r'media\.name = "([^"]+)"', block)
                    media_name = media_name_match.group(1) if media_name_match else "Unknown"

                    sink_id_match = re.search(r'Sink: (\d+)', block)
                    current_sink_id = int(sink_id_match.group(1)) if sink_id_match else -1

                    return {
                        "id"             : input_id,
                        "mac_address"    : mac_match.group(1),
                        "media_name"     : media_name,
                        "current_sink_id": current_sink_id,
                        "raw_info"       : block
                    }
        return f"No active Bluetooth stream found for {current_target_mac}"

    @checked
    def get_analog_sink_id(self):
        try:
            output = self._run_command("pactl list sinks short")
        except RuntimeError:
            return None

        for line in output.splitlines():
            if "alsa_output.platform-fe00b840.mailbox.stereo-fallback" in line or \
                    "analog-stereo" in line:
                return int(line.split()[0])
        return None

    @checked
    def make_default(self, mac):
        self.default_phone_mac = self._normalize_mac(mac)
        self._check_default_phone_registration(save_changes=True)

    @checked
    def set_bluetooth_sink_volume(self, percentage, input_id=None):
        if not (0 <= percentage <= 100):
            print("Volume percentage must be between 0 and 100.")
            return False

        target_input_id = input_id
        if target_input_id is None:
            info = self.get_bluetooth_sink_input_info()
            if info:
                target_input_id = info['id']
            else:
                print("No active Bluetooth stream found to set volume for.")
                return False

        command = f"pactl set-sink-input-volume {target_input_id} {percentage}%"
        try:
            self._run_command(command)
            print(f"Bluetooth stream {target_input_id} volume set to {percentage}%.")
            return True
        except RuntimeError:
            return False

    @checked
    def set_master_volume(self, percentage):
        if not (0 <= percentage <= 100):
            print("Volume percentage must be between 0 and 100.")
            return False
        command = f"amixer sset Master {percentage}%"
        try:
            self._run_command(command)
            print(f"Master volume set to {percentage}%.")
            return True
        except RuntimeError:
            return False

    def disconnect_bluetooth_device(self):
        current_mac = self.focused_phone.mac if self.focused_phone else None
        current_mac = self._btctlformat(current_mac)
        if not current_mac:
            print("Error: No target MAC address specified for disconnect_bluetooth_device.")
            return False

        print(f"Attempting to disconnect {current_mac}...")
        command = f"bluetoothctl disconnect {current_mac}"
        try:
            result = self._run_command(command)
            if "Successful disconnected" in result or "Failed to disconnect" not in result:
                print(f"Successfully sent disconnect command for {current_mac}.")
                self.focused_phone = next((phont for phont in self.phones if phont.is_default), None)
                return True
            else:
                print(f"Failed to disconnect {current_mac}. Result: {result}")
                return False
        except RuntimeError:
            print(f"Error while trying to disconnect {current_mac}.")
            return False

    def connect_bluetooth_device(self):
        current_mac = self.focused_phone.mac if self.focused_phone else None
        current_mac = self._btctlformat(current_mac)
        if not current_mac:
            print("Error: No target MAC address specified for connect_bluetooth_device.")
            return False

        print(f"Attempting to connect {current_mac}...")
        command = f"bluetoothctl connect {current_mac}"
        try:
            result = self._run_command(command)
            if "Connection successful" in result or "Failed to connect" not in result:
                print(f"Successfully sent connect command for {current_mac}.")
                return True
            else:
                print(f"Failed to connect {current_mac}. Result: {result}")
                return False
        except RuntimeError:
            print(f"Error while trying to connect {current_mac}.")
            return False

    @checked
    def is_bluetooth_connected(self):
        current_mac = self.focused_phone.mac if self.focused_phone else None
        current_mac = self._btctlformat(current_mac)
        print(f"Checking if {current_mac} is connected...")
        if not current_mac:
            print("Error: No target MAC address specified for is_bluetooth_connected.")
            return False

        try:
            info_output = self._run_command(f"bluetoothctl info {current_mac}")
            # Check for "Connected: yes" or "State: active" in bluetoothctl info
            if "Connected: yes" not in info_output and "State: active" not in info_output:
                return False
        except RuntimeError:
            return False

        sink_info = self.get_bluetooth_sink_input_info()
        return sink_info is not None

    @checked
    def get_device_name_from_bluetoothctl_info(self, mac_address):
        try:
            btmac = self._btctlformat(mac_address)
            output = self._run_command(f"bluetoothctl info {btmac}")
            name_match = re.search(r'Name: (.+)', output)
            if name_match:
                return name_match.group(1).strip()
        except RuntimeError:
            pass
        return None

    @checked
    def _load_known_phones_from_system(self):
        """
        Populates the known_phones list by querying bluetoothctl for paired devices.
        """
        print("[Bluetooth] Loading known phones from system...")
        self.phones = []
        try:
            # Get list of paired devices
            paired_devices_output = self._run_command("bluetoothctl devices Paired")

            for line in paired_devices_output.splitlines():
                match = re.match(r'Device ([0-9A-Fa-f:]+) (.+)', line)
                if match:
                    mac = match.group(1)
                    name = match.group(2).strip()
                    print(f"[Bluetooth] Found paired device: n:{name} ({mac})")
                    phone1 = BtPhone(name, mac, datetime.now(), False)
                    self.phones.append(phone1)
                elif "No devices paired" in line:
                    print("[Bluetooth] No devices currently paired on the system.")

        except RuntimeError as e:
            print(f"[Bluetooth] Could not load known phones from system: {e}")
        except FileNotFoundError:
            print("[Bluetooth] bluetoothctl command not found. Cannot load known phones.")
        print(f"[Bluetooth] Known phones loaded. Total: {len(self.phones)}")

    @checked
    def _dynamic_update_known_phones_list(self, name, mac):
        """
        Adds a new phone to known_phones or updates an existing one.
        """
        normalized_mac = self._normalize_mac(mac)

        # Check if phone already exists in our list
        found_phone = next((phone for phone in self.phones if phone.mac == normalized_mac), None)

        if found_phone:
            found_phone.name = name  # Update name in case it changed on the device
            found_phone.date_added = datetime.now()  # Update last connected time
            print(f"[Bluetooth] Updated known phone entry: {found_phone}")
        else:
            # Add new phone
            new_phone = BtPhone(name, mac, datetime.now(), False)
            self.phones.append(new_phone)
            print(f"[Bluetooth] Added new phone to known list: {new_phone}")

    @checked
    def _save_default(self):
        with open("data/default_mac.txt", "w") as f:
            f.write(self.default_phone_mac)

    def _check_default_phone_registration(self, save_changes=True):
        print("[CHKDEF]: Checking default phone integrity...")
        if len(self.phones) == 0:
            print("[CHKDEF]: self.phones is empty. Cannot check default phone integrity.")
            return False
        def_phone = next((phone2 for phone2 in self.phones if self._normalize_mac(phone2.mac) == self.default_phone_mac), None)
        if not def_phone:
            print("[CHKDEF]: Default phone not found in self.phones. Updating default phone (first in list).")
            self.default_phone_mac = self.phones[0].mac
            def_phone = self.phones[0]
            if save_changes:
                self._save_default()
        else:
            print("[CHKDEF]: Default phone found in self.phones.")
        for phone3 in self.phones:
            phone3.is_default = phone3.mac == def_phone.mac
        return True

    def make_discoverable_and_listen(self, timeout=180):
        try:
            import pexpect
            if not os.path.exists("/usr/bin/bluetoothctl"):
                raise FileNotFoundError("bluetoothctl not found. Ensure BlueZ is installed.")
        except ImportError:
            print("\nERROR: 'pexpect' library not found. Please install it using: pip install pexpect\n")
            return None
        except FileNotFoundError as e:
            print(f"\nERROR: {e}\n")
            return None

        print("\n--- Pi is now discoverable and pairable. ---")
        print(f"Waiting for a device to connect or pair (timeout: {timeout} seconds).")
        print("Please initiate pairing/connection from your Bluetooth device.")

        child = None
        connected_mac_result = None
        newly_paired_mac = None  # Track newly paired device to trust it
        start_time = time.time()

        try:
            child = pexpect.spawn('bluetoothctl', encoding='utf-8', timeout=5)
            child.sendline('power on')
            child.sendline('discoverable on')
            child.sendline('pairable on')
            child.sendline('agent on')
            child.sendline('default-agent')

            child.expect('bluetooth')
            print("bluetoothctl process started. Waiting for patterns...")

            while time.time() - start_time < timeout:
                try:
                    i = child.expect([
                        r'\[agent\] Confirm passkey (\d+) \(yes/no\):',
                        r'\[agent\] Authorize service ([0-9a-fA-F\-]+) \(yes/no\):',
                        r'\[NEW\] Device ([0-9A-Fa-f:]+) (.+)',
                        # Look for 'Paired: yes'
                        r'(?:\[CHG\] )?Device ([0-9A-Fa-f:]+) Paired: (?:yes|Yes|YES)',
                        # Look for 'Transport ... State: active' for connection
                        r'\[CHG\] Transport /org/bluez/hci0/dev_([0-9A-Fa-f_]+)/fd\d+ State: active',
                        r'Failed to pair:',
                        r'Failed to connect:',
                        pexpect.EOF
                    ], timeout=5)

                except pexpect.exceptions.TIMEOUT:
                    # Proactive check for connection status, especially after pairing
                    if newly_paired_mac:
                        print(
                            f"[Bluetooth] (DEBUG) No pattern match in 5s. Checking status for newly paired device {newly_paired_mac}...")
                        try:
                            info_output = self._run_command(f"bluetoothctl info {self._btctlformat(newly_paired_mac)}")
                            print(f"[Bluetooth] (DEBUG) `bluetoothctl info` output for {newly_paired_mac}")
                            if "Connected: yes" in info_output or "State: active" in info_output:
                                print(
                                    f"[Bluetooth] (DEBUG) `bluetoothctl info` confirms {newly_paired_mac} is now Connected/Active. Proceeding.")
                                current_connected_mac = newly_paired_mac

                                print(f"[Bluetooth] Verifying A2DP sink input for {current_connected_mac}...")
                                for _ in range(5):
                                    time.sleep(1)
                                    sink_info = self.get_bluetooth_sink_input_info(current_connected_mac)
                                    if sink_info:
                                        print(
                                            f"[Bluetooth] A2DP sink input active for {current_connected_mac}. Ready to stream.")
                                        phone_name = self.get_device_name_from_bluetoothctl_info(
                                            current_connected_mac) or \
                                                     sink_info.get('media_name',
                                                                   f"Unknown Device ({current_connected_mac})")
                                        self._dynamic_update_known_phones_list(phone_name, current_connected_mac)
                                        if len(self.phones) <= 1:
                                            self.make_default(current_connected_mac)
                                            print(f"[Bluetooth] Default phone set to {current_connected_mac}.")
                                        connected_mac_result = current_connected_mac
                                        break

                                if connected_mac_result:
                                    break
                                else:
                                    print(
                                        f"[Bluetooth] (DEBUG) Connected/Active, but A2DP sink input did not activate. Continuing loop.")
                            else:
                                print(
                                    f"[Bluetooth] (DEBUG) `bluetoothctl info` for {newly_paired_mac} still shows not connected. Waiting...")
                        except Exception as e:
                            print(f"[Bluetooth] (DEBUG) Error checking info for {newly_paired_mac}: {e}")
                    time.sleep(0.5)
                    continue

                except pexpect.exceptions.EOF:
                    print("[Bluetooth] bluetoothctl exited unexpectedly (EOF).")
                    break

                if i == 0:  # Confirm passkey
                    passkey = child.match.group(1)
                    print(f"[Bluetooth] Passkey requested: {passkey}. Responding 'yes'.")
                    child.sendline('yes')
                    time.sleep(0.1)
                elif i == 1:  # Authorize service
                    service_uuid = child.match.group(1)
                    print(f"[Bluetooth] Service authorization requested for {service_uuid}. Responding 'yes'.")
                    child.sendline('yes')
                    time.sleep(0.1)
                elif i == 2:  # New device found
                    new_mac = child.match.group(1)
                    new_name = child.match.group(2)
                    print(f"[Bluetooth] Found device: {new_name} ({new_mac})")
                    # Add to known list immediately, but not as default unless it connects
                    self._dynamic_update_known_phones_list(new_name, new_mac)
                elif i == 3:  # Device just paired (Now with flexible regex)
                    paired_mac = child.match.group(1)
                    print(f"[Bluetooth] Device {paired_mac} Paired: yes")
                    newly_paired_mac = paired_mac

                    # Ensure it's in our known list and update its name if available
                    phone_name = self.get_device_name_from_bluetoothctl_info(
                        paired_mac) or f"Unknown Device ({paired_mac})"
                    self._dynamic_update_known_phones_list(phone_name, paired_mac)  # Add or update, but don't set default yet

                    # After pairing, trust the device immediately
                    print(f"[Bluetooth] Device {paired_mac} paired. Trusting...")
                    child.sendline(f'trust {self._btctlformat(paired_mac)}')
                    print(f'[Bluetooth] Sent command: trust {self._btctlformat(paired_mac)}')
                    try:
                        child.expect(f'Changing {self._btctlformat(paired_mac)} trusted: yes', timeout=10)
                        print(f"[Bluetooth] Device {paired_mac} trusted.")
                    except pexpect.TIMEOUT:
                        print(
                            f"[Bluetooth] Timeout waiting for trust confirmation for {paired_mac}. Continuing anyway.")
                    except Exception as trust_e:
                        print(f"[Bluetooth] Error during trust operation for {paired_mac}: {trust_e}")

                elif i == 4:  # Transport State: active (This is our new connection indicator)
                    connected_mac_underscores = child.match.group(1)
                    current_connected_mac = connected_mac_underscores.replace('_', ':')
                    print(f"[Bluetooth] Transport for {current_connected_mac} is now active. Audio stream established.")

                    print(f"[Bluetooth] Verifying A2DP sink input for {current_connected_mac}...")
                    for _ in range(5):
                        time.sleep(1)
                        sink_info = self.get_bluetooth_sink_input_info(current_connected_mac)
                        if sink_info:
                            print(f"[Bluetooth] A2DP sink input active for {current_connected_mac}. Ready to stream.")
                            phone_name = self.get_device_name_from_bluetoothctl_info(current_connected_mac) or \
                                         sink_info.get('media_name', f"Unknown Device ({current_connected_mac})")

                            self._dynamic_update_known_phones_list(phone_name,
                                                                   current_connected_mac)

                            if len(self.phones) <= 1:
                                self.make_default(current_connected_mac)

                            connected_mac_result = current_connected_mac
                            break

                    if connected_mac_result:
                        break
                    else:
                        print(
                            f"[Bluetooth] Device {current_connected_mac} audio transport active, but A2DP sink input did not activate. Will continue listening.")

                elif i == 5:  # Failed to pair
                    error_msg = child.match.group(0)
                    print(f"[Bluetooth] Pairing failed: {error_msg}")
                    mac_in_error = re.search(r'Device ([0-9A-Fa-f:]+) ', error_msg)
                    if mac_in_error:
                        failed_mac = mac_in_error.group(1)
                        print(f"[Bluetooth] Attempting to remove failed device {failed_mac} to allow retry.")
                        child.sendline(f'remove {self._btctlformat(failed_mac)}')
                        try:
                            child.expect('Device has been removed', timeout=5)
                        except pexpect.TIMEOUT:
                            print(f"[Bluetooth] Timeout waiting for remove confirmation for {failed_mac}.")
                        except Exception as remove_e:
                            print(f"[Bluetooth] Error during remove operation for {failed_mac}: {remove_e}")
                        newly_paired_mac = None
                    time.sleep(1)
                elif i == 6:  # Failed to connect
                    error_msg = child.match.group(0)
                    print(f"[Bluetooth] Connection failed: {error_msg}")
                    time.sleep(1)

            if time.time() - start_time >= timeout and connected_mac_result is None:
                print(
                    f"[Bluetooth] Overall timeout ({timeout}s) reached while waiting for connection. No device connected.")

        except Exception as e:
            print(f"[Bluetooth] An unexpected error occurred in make_discoverable_and_listen: {e}")
        finally:
            if child and child.isalive():
                child.sendline('quit')
                try:
                    child.expect(pexpect.EOF, timeout=5)
                except pexpect.TIMEOUT:
                    print("[Bluetooth] Timeout waiting for bluetoothctl to quit.")
                child.close()
                print("[Bluetooth] bluetoothctl session closed.")
            elif child:
                child.close()

        return connected_mac_result

    def focus_phone(self, mac):
        mac = self._normalize_mac(mac)
        if not mac:
            print("Error: No target MAC address specified for focus_phone.")
            return False
        macs = [phone.mac for phone in self.phones]
        if mac not in macs:
            print(f"Error: Target MAC address {mac} not found in known phones.")
            return False

        self.focused_phone = next((phont for phont in self.phones if phont.mac == mac), None)
        assert self.focused_phone is not None and self.focused_phone.mac == mac
        return True

    def send_media_command(self, command):
        tmac = self._bluezformat(self.focused_phone.mac)
        print("sending {}".format(tmac))
        if not tmac:
            print("Error: No target MAC address specified for send_media_command.")
            return False
        recognized = ["play", "pause", "stop", "next", "previous", "fastforward", "rewind"]
        cmd = command.lower().capitalize() if command.lower() in recognized else None
        try:
            result = self._run_command(
                f"dbus-send --system --dest=org.bluez --print-reply "
                f"/org/bluez/hci0/dev_{tmac}/player0 "
                f"org.bluez.MediaPlayer1.{command.capitalize()} "
            )
            return result
        except RuntimeError:
            return False

    def seek_song(self, pos: str | int):
        pos = int(pos) if isinstance(pos, str) else pos
        tmac = self._bluezformat(self.focused_phone.mac)
        print("sending {}".format(tmac))
        if not tmac:
            print("Error: No target MAC address specified for seek_song.")

        try:
            result = self._run_command(
                f"dbus-send --system --dest=org.bluez --print-reply "
                f"/org/bluez/hci0/dev_{tmac}/player0 "
                f"org.bluez.MediaPlayer1.Seek int64:{pos}"
            )
            return result
        except RuntimeError:
            return False
