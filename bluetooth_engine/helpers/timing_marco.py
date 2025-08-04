
"""
HELPER FUNCTIONS FOR ARTEPLAYER, TIMING WISE
"""


def MinutesSeconds(millis):
    """
    Turns milliseconds into minutes and seconds.
    :param millis: milliseconds
    :return: dictionary with keys "minutes" and "seconds"
    """
    if isinstance(millis, str):
        if not millis.isdigit():
            return False
        else:
            millis = int(millis)
    return {
        "minutes" : int(millis / 60_000),
        "seconds" : int((millis / 1000) % 60)
    }


def Pointed(millis):
    """
    Turns milliseconds into minutes and seconds.
    :param millis: milliseconds
    :return: pointed float in the format mins.secs
    """
    if isinstance(millis, str):
        if not millis.isdigit():
            return False
        else:
            millis = int(millis)
    minutes = int(millis / 60_000)
    seconds = int((millis / 1000) % 60)
    return float(f"{minutes}.{seconds:02d}")


def Milliseconds(minutes: int, seconds: int):
    """
    Turns minutes and seconds into milliseconds.
    :param minutes: minutes
    :param seconds: seconds
    :return: milliseconds
    """
    return minutes * 60_000 + seconds * 1000
