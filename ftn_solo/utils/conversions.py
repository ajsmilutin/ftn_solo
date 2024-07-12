from geometry_msgs.msg import Point


def ToPoint(translation):
    return Point(x=translation[0], y=translation[1], z=translation[2])
