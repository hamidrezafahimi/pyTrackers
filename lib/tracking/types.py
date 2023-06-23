from enum import Enum

class ExtType(Enum):
    raw = 0
    viot = 1


class Trackers(Enum):
    KCFHOG = 1
    LDES = 2
    STRCF = 3
    CSRDCF = 4
    DIMP = 6
    KYS = 7
    TOMP = 8
    MIXFORMER = 9

    @property
    def tag(self):
        return self.name.lower()

    @property
    def group(self):
        v = self.value
        if v == 5 or v == 6 or v == 7 or v == 8:
            return TrackerType.eth
        elif v == 1 or v == 2 or v == 3 or v == 4:
            return TrackerType.cf
        elif v == 9:
            return TrackerType.mxf

    @property
    def config(self):
        if hasattr(self, 'a'):
            return self.__a
        else:
            raise Exception('Request for `config` attribute of tracker {:s} while no config is set\
           by user'.format(self.name))
    
    @config.setter
    def config(self, val):
        self.__a = val

    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self, var):
        self.__a = var
    

class TrackerType(Enum):
    opencv = 1
    cf = 2
    eth = 3
    mxf = 4
