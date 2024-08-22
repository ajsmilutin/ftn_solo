import numpy as np
import pinocchio as pin


class Motion:
    def __init__(self, *args, **kwargs) -> None:
        self.trajectory = None
        self.Kp = kwargs.pop("Kp", 100)
        self.Kd = kwargs.pop("Kd", 50)

    def set_trajectory(self, trajectory):
        self.trajectory = trajectory

    def get_desired_acceleration(self, t, model, data):
        pdes, vdes, ades = self.trajectory.get(t)
        return (
            ades
            + self.Kp * self.get_pos_error(pdes, model, data)
            + self.Kd * self.get_vel_error(vdes, model, data)
        )


class EEFLinearMotion(Motion):
    def __init__(self, eef_index, selected=[True, True, True], *args, **kwargs) -> None:
        super(EEFLinearMotion, self).__init__(*args, **kwargs)
        self.eef_index = eef_index
        self.selected = selected
        self.dim = np.count_nonzero(selected)
        self.extended = selected + [False, False, False]

    def get_jacobian(self, model, data):
        return pin.getFrameJacobian(
            model, data, self.eef_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[self.extended, :]

    def get_pos_error(self, pdes, model, data):
        return pdes - data.oMf[self.eef_index].translation[self.selected]

    def get_vel_error(self, vdes, model, data):
        return (
            vdes
            - pin.getFrameVelocity(
                model, data, self.eef_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            ).linear[self.selected]
        )

    def get_acceleration(self, model, data):
        return pin.getFrameAcceleration(
            model, data, self.eef_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ).linear[self.selected]


class EEFAngularMotion(Motion):
    def __init__(self, eef_index, selected=[True, True, True], *args, **kwargs) -> None:
        super(EEFAngularMotion, self).__init__(*args, **kwargs)
        self.eef_index = eef_index
        self.selected = selected
        self.dim = np.count_nonzero(selected)
        self.extended = [False, False, False] + selected

    def get_jacobian(self, model, data):
        return pin.getFrameJacobian(
            model, data, self.eef_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[self.extended, :]

    def get_pos_error(self, pdes, model, data):
        ori = data.oMf[self.eef_index].rotation        
        return np.matmul(ori, pin.log(np.matmul(ori.T, pdes)))[self.selected]

    def get_vel_error(self, vdes, model, data):
        return (
            vdes
            - pin.getFrameVelocity(
                model, data, self.eef_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            ).angular[self.selected]
        )

    def get_acceleration(self, model, data):
        return pin.getFrameAcceleration(
            model, data, self.eef_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        ).angular[self.selected]
