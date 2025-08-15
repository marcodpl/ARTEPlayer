"""
NMEA TRANSLATOR FOR ARTEPLAYER
Aim: interface between GNSS module and script.
"""

import serial
import time


FIX_QUALITY_TABLE = {
    0: "NO FIX",
    1: "GPS",
    2: "DGPS",
    3: "PPS Fixed",
    4: "RTK Fixed",
    5: "RTK Float",
    6: "Estimated (dead reckoning)",
    7: "Manual input mode",
    8: "Simulation mode"
}

FIX_TYPE_TABLE = {
    1: "No fix",
    2: "2D",
    3: "3D"
}

_gpsprocessed = False


class Satellite(object):
    def __init__(self):
        self._id = 0
        self.type: str = ""
        self.detailed: bool = False
        self.elevation = 0.0
        self.azimuth = 0.0
        self.snr = 0.0

    @property
    def prn(self):
        return self._id

    @prn.setter
    def prn(self, prn):
        self._id = prn
        self.type = _sort_sat(prn)

    def __call__(self):
        return f"prn={self.prn}, type={self.type}, elevation={self.elevation}, azimuth={self.azimuth}, snr={self.snr}"


class NMEAReport:
    def __init__(self):
        self.has_fix = False
        self.fix_quality = ""
        self.fix_type = ""
        self.constellation = ""
        self.latitude = 0.0
        self.longitude = 0.0
        self.speed_knots = 0.0
        self.speed_kph = 0.0
        self.time = ""
        self.date = ""
        self.SATIndex = 0
        self.satellites: list[Satellite] = []
        self.altitude = 0.0
        self.ho_dop = 0.0  # Horizontal Dilution of Precision
        self.po_dop = 0.0  # Position Dilution of Precision
        self.ve_dop = 0.0  # Vertical Dilution of Precision
        self.geoid_separation = 0.0

    def __call__(self):
        return f"has_fix={self.has_fix}, latitude={self.latitude}, longitude={self.longitude}, " \
               f"speed_knots={self.speed_knots}, time={self.time}, date={self.date}, satellites={self.satellites}, " \
               f"altitude={self.altitude}, ho_dop={self.ho_dop}, po_dop={self.po_dop}, ve_dop={self.ve_dop}"


def tick_ms():
    return time.monotonic() * 1000


def parse_gps_data(nmea_chunk):
    """Legacy function to parse GPS data (for compatibility)"""
    return _process_nmea_data(nmea_chunk)


def _process_nmea_data(nmea_data):
    """Process a complete NMEA data string"""
    # Initialize data class
    global _gpsprocessed
    gps_data = NMEAReport()

    # Split the chunk into individual NMEA sentences
    sentences = nmea_data.strip().split('$')

    for sentence in sentences:
        if sentence is None: continue

        if sentence[2:5] == "RMC":
            _parse_rmc(sentence, gps_data)
        elif sentence[2:5] == "VTG":
            _parse_vtg(sentence, gps_data)
        elif sentence[2:5] == "GGA":
            _parse_gga(sentence, gps_data)
        elif sentence[2:5] == "GSA":
            _parse_gsa(sentence, gps_data)
        elif sentence[2:5] == "GSV":
            _parse_gsv(sentence, gps_data)

    _gpsprocessed = False
    return gps_data


def _parse_rmc(sentence, gps_data):
    """Parse RMC sentence for time, date, location, and speed"""

    # Split the sentence into parts
    parts = sentence.split(',')

    if len(parts) < 12:
        return

    # Check if we have a fix
    if parts[2] == 'A':
        gps_data.has_fix = True
    else:
        gps_data.has_fix = False
        # Don't return here, continue to extract time and date

    # Extract time (format: HHMMSS.SS) with error handling
    if parts[1] and len(parts[1]) >= 6:
        try:
            hour = parts[1][0:2]
            minute = parts[1][2:4]
            second = parts[1][4:]
            gps_data.time = f"{hour}:{minute}:{second}"
        except (ValueError, IndexError):
            # Keep the existing time value if parsing fails
            pass

    # Extract date (format: DDMMYY) with error handling
    if parts[9] and len(parts[9]) >= 6:
        try:
            day = parts[9][0:2]
            month = parts[9][2:4]
            year = "20" + parts[9][4:6]  # Assuming we're in the 2000s
            gps_data.date = f"{day}/{month}/{year}"
        except (ValueError, IndexError):
            # Keep the existing date value if parsing fails
            pass

    # Only extract position and speed if we have a valid fix
    if gps_data.has_fix:
        # Extract latitude and longitude with sign based on direction
        if parts[3] and parts[5]:
            try:
                # Latitude
                lat_deg = float(parts[3][0:2])
                lat_min = float(parts[3][2:])
                lat_decimal = lat_deg + (lat_min / 60)

                # Apply sign based on direction (N is positive, S is negative)
                if parts[4] == 'S':
                    lat_decimal = -lat_decimal
                gps_data.latitude = lat_decimal

                # Longitude
                lon_deg = float(parts[5][0:3])
                lon_min = float(parts[5][3:])
                lon_decimal = lon_deg + (lon_min / 60)

                # Apply sign based on direction (E is positive, W is negative)
                if parts[6] == 'W':
                    lon_decimal = -lon_decimal
                gps_data.longitude = lon_decimal
            except (ValueError, IndexError):
                # If parsing fails, don't update coordinates
                pass

        # Extract speed in knots
        if parts[7]:
            try:
                gps_data.speed_knots = float(parts[7])
            except ValueError:
                gps_data.speed_knots = 0.0


def _parse_vtg(sentence, gps_data):
    """Parse VTG sentence for speed and direction"""
    parts = sentence.split(',')

    if len(parts) < 8:
        return

    nwspkn = float(parts[5])
    if abs(nwspkn - gps_data.speed_knots) > 0.01:
        gps_data.speed_knots = nwspkn

    spkph = float(parts[7])
    if spkph:
        gps_data.speed_kph = spkph
    else:
        gps_data.speed_kph = 0.0


def _parse_gga(sentence, gps_data):
    """Parse GGA sentence for satellites, altitude, and HDOP"""

    parts = sentence.split(',')

    if len(parts) < 15:
        return

    # Extract fix type (0 = no fix, 1 = GPS, 2 = DGPS, 3 = PPS Fixed, 4 = RTK Fixed, 5 = RTK Float, 6 = Estimated, 7 = Manual, 8 = Simulation)
    fix_type = int(parts[6])
    gps_data.fix_quality = FIX_QUALITY_TABLE.get(fix_type, "Unknown")

    # Extract number of satellites
    if parts[7]:
        try:
            gps_data.SATIndex = int(parts[7])
        except ValueError:
            gps_data.satellites = 0

    # Extract HDOP (Horizontal Dilution of Precision)
    if parts[8]:
        try:
            gps_data.ho_dop = float(parts[8])
        except ValueError:
            gps_data.ho_dop = 0.0

    # Extract altitude
    if parts[9] and parts[10] == 'M':
        try:
            gps_data.altitude = float(parts[9])
        except ValueError:
            gps_data.altitude = 0.0

    # Extract geoid separation
    if parts[11] and parts[12] == 'M':
        try:
            gps_data.geoid_separation = float(parts[11])
        except ValueError:
            gps_data.geoid_separation = 0.0


def _parse_gsa(sentence, gps_data):
    """Parse GSA sentence for PDOP, HDOP, and VDOP"""

    parts = sentence.split(',')

    if len(parts) < 19:
        return

    gps_data.fix_type = FIX_TYPE_TABLE.get(int(parts[2]), "Unknown")

    gps_prns = []
    for p in parts[3:15]:
        try:
            gps_prns.append(int(p))
        except ValueError:
            pass
    for i, prn in enumerate(gps_prns):
        if not prn:
            continue
        sat = next((sat for sat in gps_data.satellites if sat.prn == prn), None)
        if sat is None:
            sat = Satellite()
            sat.prn = prn
            gps_data.satellites.append(sat)

    # Extract PDOP (Position Dilution of Precision)
    if parts[15]:
        try:
            gps_data.po_dop = float(parts[15])
        except ValueError:
            gps_data.po_dop = 0.0

    # Extract HDOP (Horizontal Dilution of Precision)
    if parts[16]:
        try:
            gps_data.ho_dop = float(parts[16])
        except ValueError:
            pass  # Keep existing value if we can't parse this one

    # Extract VDOP (Vertical Dilution of Precision)
    if parts[17]:  # Remove checksum part
        try:
            gps_data.ve_dop = float(parts[17])
        except ValueError:
            gps_data.ve_dop = 0.0


def _parse_gsv(sentence, gps_data):
    global _gpsprocessed
    """Parse GSV sentence for satellites"""
    parts = sentence.split(',')
    if "*" in parts[4]:
        return  # GSV sentence is empty, returning

    if parts[0][0:2] == "GP" and _gpsprocessed:
        return

    if parts[1] == parts[2]:
        _gpsprocessed = True
    j = 0
    match len(parts):
        case 9: j = 1
        case 13: j = 2
        case 17: j = 3
        case 21: j = 4
    print(f"j={j}")
    for i in range(j):
        prn = int(parts[4+i*4])
        if prn:
            sat = next((sat for sat in gps_data.satellites if sat.prn == prn), None)
            if sat is None:
                sat1 = Satellite()
                sat1.prn = prn
                sat1.detailed = True
                sat1.elevation = float(parts[5+i*4])
                sat1.azimuth = float(parts[6+i*4])
                sat1.snr = float(parts[7+i*4])
                gps_data.satellites.append(sat1)
            elif not sat.detailed:
                sat.detailed = True
                sat.elevation = float(parts[5+i*4])
                sat.azimuth = float(parts[6+i*4])
                sat.snr = float(parts[7+i*4])


def _sort_sat(prn):
    assert isinstance(prn, int), "Satellite ID must be a number."
    if prn in range(1, 33):
        return "GPS"
    elif prn in range(33, 65):
        return "SBAS"
    elif prn in range(65, 97):
        return "GLONASS"
    elif prn in range(201, 238):
        return "BEIDOU"
    elif prn in range(301, 337):
        return "GALILEO"
    else:
        return ":unknown:"


class NMEAReader(object):
    def __init__(self, ser: serial.Serial):
        assert ser is not None and ser, "Serial port must be provided."
        self.ser = ser
        self.message_buffer = ""
        self.last_data_time = tick_ms()
        self.timeout_ms = 500  # 500ms timeout between message parts
        self.current_data = NMEAReport()
        self.has_new_data = False

    def update(self):
        """Check for new GPS data and process it - non-blocking version"""
        self.has_new_data = False
        current_time = tick_ms()

        # Check if timeout occurred with data in buffer
        if current_time - self.last_data_time > self.timeout_ms and self.message_buffer:
            self._process_buffer()
            self.has_new_data = True

        # Check if new data is available - only read what's currently available
        bytes_available = self.ser.in_waiting
        if bytes_available > 0:
            try:
                # Read ONLY the currently available data - this is the key non-blocking change
                data = self.ser.read(bytes_available).decode('utf-8')

                # If buffer is empty or we recently received data, append to buffer
                if not self.message_buffer or current_time - self.last_data_time <= self.timeout_ms:
                    self.message_buffer += data
                else:
                    # If timeout occurred, process the old buffer first
                    self._process_buffer()
                    # Then start a new buffer
                    self.message_buffer = data
                    self.has_new_data = True

                # Update last data time
                self.last_data_time = current_time
            except Exception as e:
                print(f"Error reading GPS data: {e}")

        return self.has_new_data

    def get_data(self):
        """
        Get the current GPS data.
        Automatically updates before returning the data.

        Returns:
            GPSData: The current GPS data object with the most recent reading
        """
        self.update()
        return self.current_data

    def _process_buffer(self):
        """Process the complete message in buffer"""
        if not self.message_buffer:
            return

        self.current_data = _process_nmea_data(self.message_buffer)
        self.message_buffer = ""

    @property
    def latitude(self):
        """Get the current latitude"""
        self.update()
        return self.current_data.latitude

    @property
    def longitude(self):
        """Get the current longitude"""
        self.update()
        return self.current_data.longitude

    @property
    def altitude(self):
        """Get the current altitude"""
        self.update()
        return self.current_data.altitude

    @property
    def has_fix(self):
        """Get the current fix status"""
        self.update()
        return self.current_data.has_fix

    @property
    def SATIndex(self):
        """Get the current number of satellites"""
        self.update()
        return self.current_data.SATIndex

    @property
    def speed(self):
        """Get the current speed in knots"""
        self.update()
        return self.current_data.speed_knots

    @property
    def time(self):
        """Get the current GPS time"""
        self.update()
        return self.current_data.time

    @property
    def date(self):
        """Get the current GPS date"""
        self.update()
        return self.current_data.date





