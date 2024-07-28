from geometry_msgs.msg import Point, Vector3, Quaternion


def ToPoint(translation):
    return Point(x=translation[0], y=translation[1], z=translation[2])


def ToVector(translation):
    return Vector3(x=translation[0], y=translation[1], z=translation[2])


def ToQuaternion(pin_quaternion):
    return Quaternion(x=pin_quaternion.x(), y=pin_quaternion.y(), z=pin_quaternion.z(), w=pin_quaternion.w())
