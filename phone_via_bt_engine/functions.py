import io
import stat
import operator
import os
import subprocess
import sys
from datetime import datetime as dt
import threading
import time
import pexpect
import re


class MoreThanOneDefaultPhoneError(Exception):
    def __init__(self, message):
        super().__init__(message)


class BtPhone:
    def __init__(self, name, mac, date, is_default: bool = False):
        self.name = name
        self.mac = mac  # Format: XX:XX:XX:XX:XX:XX
        self.date = date
        self.is_default = is_default

    def __repr__(self):
        # A friendly representation for printing the object
        return (f"BtPhone(name='{self.name}', mac='{self.mac}', "
                f"added='{self.date.strftime('%Y-%m-%d %H:%M:%S')}', default={self.is_default})")


"""
ALL OPERATIONS ARE PERFORMED ON FOCUSED PHONE.
FOCUSED PHONE is, at start, the default phone saved in file.
Operations as add_phone and similar are exempt.
FOCUSED PHONE DIFFERS FROM DEFAULT PHONE. TO CHANGE DEFAULT PHONE USE self.set_default_phone()
FOCUSED PHONE SHOULD ALWAYS BE CONNECTED PHONE IF CONNECTED PHONE IS PRESENT!
Focused phone changes:
 - at start (default phone)
 - via focus_phone_via_name/mac()
 - via unfocus_phone() [returns to default phone]
 - via delete_phone() [ deletes focused phone, returns to default phone] 
 
Where information is kept:
 - focused phone: in focused_phone attribute
 - default phone: in is_default attribute of BtPhone instances, stored as a list inside self.phones. Moreover, in data/phone_coords.txt.
"""


class Functions:
    @staticmethod
    def _run_command(command):
        """
        Executes a shell command and returns its stdout.
        Raises an exception on error for better error handling in methods.
        """
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Command failed: {e.cmd}") from e
        except FileNotFoundError:
            raise FileNotFoundError(f"Command not found: {command.split()[0]}")

    def __init__(self):
        """
        Reads all phones listed in data/phone_coords.txt.
        NOTE: DOES NOT CHECK FOR DEFAULT PHONE UNICITY. CHECK FILE INTEGRITY (could lead to errors)
        Focuses phone on default phone. Will be None if no default phone exists.
        """
        self.phones: list[BtPhone] = []
        self.file_path = "data/phone_coords.txt"
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                for line in f:
                    print(str(line))
                    name, mac, date, is_default = line.strip().split(";")

                    self.phones.append(BtPhone(name, mac, date, is_default.lower() == "true"))
        self.focused_phone: BtPhone = self.get_default_phone()

    def get_default_phone(self):
        """
        Returns default phone.
        :return: default BtPhone instance
        """
        return next((p for p in self.phones if p.is_default), None)

    def set_default_phone(self):
        """
        Sets default phone to focused phone. Will be saved immediately. Prevents multiple default phones from existing.
        """
        for p in self.phones:
            p.is_default = (p.mac == self.focused_phone.mac)
        self.save_phones()

    def focus_phone_via_name(self, name: str):
        """
        Focuses a phone from self.phones list, searching via provided name.
        :param name: phone search criteria.
        """
        self.focused_phone = next((p for p in self.phones if p.name == name), None)

    def focus_phone_via_mac(self, mac: str):
        """
        Focuses a phone from self.phones list, searching via provided mac (physical) address.
        :param mac: phone search criteria.
        """
        self.focused_phone = next((p for p in self.phones if p.mac == mac), None)

    def unfocus_phone(self):
        """
        Focuses default phone. Prevents focused phone being empty (as long as a default phone exists).
        """
        self.focused_phone = self.get_default_phone()

    def save_phones(self):
        """
        Dumps all phones in self.phones list to data/phone_coords.txt.
        Called automatically when a phone is added, removed, or paired client-side.
        Overwrite mode means it can be called multiple times without overwriting issues.
        """
        with open(self.file_path, "w") as f:
            for phone in self.phones:
                f.write(f"{phone.name};{phone.mac};{phone.date};{phone.is_default}\n")

    def _add_phone_to_phones(self, name, mac, date, set_default=False):
        """
        Manually adds a BtPhone instance to self.phones list. No confirmation of data is provided.
        Shouldn't be used alone: connections should come client-wise (from the phone). No pairing is provided. No trusting is provided.
        :param name: phone name
        :param mac: phone mac address
        :param date: date phone was added
        :param set_default: is phone default (overrides current default phone)
        """
        self.phones.append(BtPhone(name, mac, date, set_default))
        if set_default:
            for p in self.phones:
                p.is_default = (p.mac == mac)
        self.save_phones()

    def delete_phone(self):
        """
        Deletes focused phone. Reverts focused phone to default phone.
        """
        self.phones = [p for p in self.phones if p.mac != self.focused_phone.mac]
        self.focused_phone = self.get_default_phone()
        self.save_phones()
        # TODO: IF DELETED PHONE IS DEFAULT PHONE, CHOOSE ANOTHER TO BE DEFAULT.

    def pair_and_trust_phone(self):
        """
        Pairs and trusts focused Bluetooth phone using bluetoothctl.
        """
        mac = self.focused_phone.mac
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

    def connect_to_phone(self):
        """
        Connects to focused Bluetooth phone and enables audio routing to AUX.
        """
        mac = self.focused_phone.mac
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
        os.system(" amixer cset numid=3 1")
        print("🔈 Output set to 3.5mm AUX")

    def send_media_command(self, command: str):
        """
        Sends a media control command to focused phone (play, pause, next, previous).
        """
        valid_cmds = {"play", "pause", "next", "previous"}
        if command.lower() not in valid_cmds:
            print(f"⚠️ Invalid command: {command}")
            return

        mac_dbus = self.focused_phone.mac.replace(":", "_")
        os.system(
            f"dbus-send --system --dest=org.bluez --print-reply "
            f"/org/bluez/hci0/dev_{mac_dbus}/player0 "
            f"org.bluez.MediaPlayer1.{command.capitalize()} "
        )
        print(f"🎵 Sent '{command}' command to {mac_dbus}")

    @staticmethod
    def get_device_name(mac):
        """
        Queries the device name via bluetoothctl.
        """
        try:
            result = subprocess.run(["bluetoothctl", "info", mac], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                if "Name:" in line:
                    return line.split("Name:")[1].strip()
        except Exception as e:
            print(f"⚠️ Failed to get name for {mac}: {e}")
        return "Unknown"

    def make_discoverable_and_listen(self, timeout=180):
        """
        Makes the Pi discoverable and pairable, and waits for a Bluetooth device
        to connect and stream A2DP audio.
        Automatically handles pairing and trust for new devices.

        Returns the MAC address (string) of the successfully connected and
        streaming device on success, or None on timeout/failure.
        """
        try:
            import pexpect  # Imports pexpect, needed for command handling.
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
        connected_mac_result = None  # This variable will hold the MAC on success
        newly_paired_mac = None  # This variable holds the pairing MAC.

        try:
            # Spawn bluetoothctl process
            child = pexpect.spawn('bluetoothctl', encoding='utf-8', timeout=timeout)
            child.sendline('power on')
            child.sendline('discoverable on')
            child.sendline('pairable on')
            child.sendline('agent on')  # Use the default agent for prompts
            child.sendline('default-agent')

            # Flush initial output and wait for the prompt
            child.expect('bluetooth')
            print("bluetoothctl process started. Waiting for patterns...")

            # Main loop to listen for events
            while True:
                i = child.expect([
                    r'\[agent\] Confirm passkey (\d+) \(yes/no\):',
                    r'\[agent\] Authorize service ([0-9a-fA-F\-]+) \(yes/no\):',
                    r'\[NEW\] Device ([0-9A-Fa-f:]+) (.+)',
                    r'\[CHG\] Device ([0-9A-Fa-f:]+) Connected: yes',
                    r'\[CHG\] Device ([0-9A-Fa-f:]+) Paired: yes',
                    r'Failed to pair:',
                    r'Failed to connect:',
                    pexpect.TIMEOUT,
                    pexpect.EOF
                ], timeout=timeout)

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
                elif i == 3:  # Device connected!
                    current_connected_mac = child.match.group(1)
                    current_connected_name = child.match.group(2)
                    print(f"[Bluetooth] Device {current_connected_mac} Connected: yes")

                    if current_connected_mac == newly_paired_mac:
                        print(f"[Bluetooth] Newly paired device {current_connected_mac} connected. Trusting...")
                        child.sendline(f'trust {current_connected_mac}')
                        try:
                            child.expect(f'Changing {current_connected_mac} trusted: yes', timeout=10)
                            print(f"[Bluetooth] Device {current_connected_mac} trusted.")
                        except pexpect.TIMEOUT:
                            print(
                                f"[Bluetooth] Timeout waiting for trust confirmation for {current_connected_mac}. Continuing anyway.")
                        except Exception as trust_e:
                            print(f"[Bluetooth] Error during trust operation for {current_connected_mac}: {trust_e}")
                        finally:
                            print("Adding device to known devices list...")
                            self._add_phone_to_phones(current_connected_name, current_connected_mac, dt.now(), False)
                            print("Focusing phone...")
                            self.focus_phone_via_mac(current_connected_mac)
                            if len(self.phones) <= 1:
                                print("First phone connected. Setting as default.")
                                self.set_default_phone()
                    else:
                        # Check if already trusted. If not, try to trust for future auto-connections
                        try:
                            info_output = self._run_command(f"bluetoothctl info {current_connected_mac}")
                            if "Trusted: yes" not in info_output:
                                print(
                                    f"[Bluetooth] Device {current_connected_mac} connected but not trusted. Trusting now...")
                                child.sendline(f'trust {current_connected_mac}')
                                child.expect(f'Changing {current_connected_mac} trusted: yes', timeout=10)
                                print(f"[Bluetooth] Device {current_connected_mac} trusted.")
                        except Exception as e:
                            print(f"[Bluetooth] Could not check/set trust status for {current_connected_mac}: {e}")
                        finally:
                            print("Focusing phone...")
                            self.focus_phone_via_mac(current_connected_mac)

                    # Verify PipeWire sink input is active for the connected device
                    print(f"[Bluetooth] Verifying A2DP sink input for {current_connected_mac}...")
                    for _ in range(5):  # Check up to 5 times (5 seconds)
                        time.sleep(1)
                        if self.get_bluetooth_sink_input_info(current_connected_mac):
                            print(f"[Bluetooth] A2DP sink input active for {current_connected_mac}. Ready to stream.")
                            connected_mac_result = current_connected_mac  # Store result
                            break  # Break out of stream verification loop

                    if connected_mac_result:  # If stream was successfully verified
                        break  # Break out of the main while True loop (connection found)
                    else:
                        print(
                            f"[Bluetooth] Device {current_connected_mac} connected, but A2DP sink input did not activate. Will continue listening.")
                        # Do not set connected_mac_result, let loop continue

                elif i == 4:  # Device just paired
                    paired_mac = child.match.group(1)
                    print(f"[Bluetooth] Device {paired_mac} Paired: yes")
                    newly_paired_mac = paired_mac  # Mark as newly paired. Trust will happen on connection.
                elif i == 5:  # Failed to pair
                    error_msg = child.match.group(0)
                    print(f"[Bluetooth] Pairing failed: {error_msg}")
                    mac_in_error = re.search(r'Device ([0-9A-Fa-f:]+) ', error_msg)
                    if mac_in_error:
                        failed_mac = mac_in_error.group(1)
                        print(f"[Bluetooth] Attempting to remove failed device {failed_mac} to allow retry.")
                        child.sendline(f'remove {failed_mac}')
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
                elif i == 7:  # Timeout
                    print(f"[Bluetooth] Timeout ({timeout}s) reached while waiting for connection.")
                    break  # Break out of the main while True loop
                elif i == 8:  # EOF - bluetoothctl exited unexpectedly
                    print("[Bluetooth] bluetoothctl exited unexpectedly.")
                    break  # Break out of the main while True loop

        except pexpect.exceptions.TIMEOUT:
            print("[Bluetooth] Pexpect operation timed out during an expect call.")
        except pexpect.exceptions.EOF:
            print("[Bluetooth] bluetoothctl session ended unexpectedly (EOF).")
        except Exception as e:
            print(f"[Bluetooth] An unexpected error occurred: {e}")
        finally:
            if child and child.isalive():  # Ensure child process is still alive before sending quit
                child.sendline('quit')
                try:
                    child.expect(pexpect.EOF, timeout=5)  # Wait for it to close
                except pexpect.TIMEOUT:
                    print("[Bluetooth] Timeout waiting for bluetoothctl to quit.")
                child.close()  # Explicitly close the pexpect process
                print("[Bluetooth] bluetoothctl session closed.")
            elif child:
                # If child exited already (e.g., due to EOF match), just close it if needed
                child.close()

        # This return happens ONLY if the try block broke out or an exception occurred
        # and 'connected_mac_result' was not set by a successful stream detection.
        return connected_mac_result



