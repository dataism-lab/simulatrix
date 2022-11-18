class DocetException(Exception):
    pass


class BewException(DocetException):
    pass


class BewParamsException(BewException):
    pass


class ControlException(DocetException):
    pass


class InvalidConfigException(ControlException):
    pass


class GymCarlaException(DocetException):
    pass