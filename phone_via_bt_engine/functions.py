class MoreThanOneDefaultPhoneError(Exception):
    def __init__(self, message):
        super().__init__(message)


class Functions:
    def __init__(self):
        self.name_coords: list[str] = []
        self.mac_coords: list[str] = []
        self.date_coords: list[str] = []
        self.is_default_coords: list[bool] = []
        with open("data/phone_coords.txt", "r+") as f:
            lines = [line.strip() for line in f.readlines()]
            print(str(lines))
            for i in range(len(lines)):
                split = lines[i].split(";")
                self.name_coords.append(str(split[0]))
                self.mac_coords.append(str(split[1]))
                self.date_coords.append(str(split[2]))
                self.is_default_coords.append(bool(split[3]))
        if self.is_default_coords.count(True) > 1:
            raise MoreThanOneDefaultPhoneError("cazzo")


