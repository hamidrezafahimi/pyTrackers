from enum import Enum

class ExtType(Enum):
    raw = 0
    viot = 1


class Trackers(Enum):
    KCFHOG = 1
    LDES = 2
    STRCF = 3
    CSRDCF = 4
    PRDIMP50 = 5
    DIMP50 = 6
    KYS = 7
    TOMP = 8
    MIXFORMERVIT = 9

    @property
    def type(self):
        v = self.value
        if v == 5 or v == 6 or v == 7 or v == 8:
            return TrackerType.eth
        elif v == 1 or v == 2 or v == 3 or v == 4:
            return TrackerType.cf
        elif v == 9:
            return TrackerType.mxf


class TrackerType(Enum):
    opencv = 1
    cf = 2
    eth = 3
    mxf = 4
