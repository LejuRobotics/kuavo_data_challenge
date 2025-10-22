
import torch
import numpy as np
from enum import Enum
from typing import Union

from scipy.spatial.transform import Rotation as R


class RotationRepr(str, Enum):
    EULER = "euler"  # TODO(refactor): Now assuming seq=ZYX; ought to find a better way to handle this.
    QUAT = "quat"
    MATRIX = "matrix"
    ROTATION_6D = "rotation_6d"
    AXIS_ANGLE = "axis_angle"

# ====Mappings to get the order of euler angles in rpy.====
# Most of the states we get from robot are in rpy order,
# but their rotation seq is usually not rpy. Need to be reordered before
# sending them into scipy functions.
_EULER_RPY_TO_SEQ_MAPPING = {
    "ZYX": np.array([2, 1, 0]),
    "XYZ": np.array([0, 1, 2]),
}

_EULER_SEQ_TO_RPY_MAPPING = {
    "ZYX": np.array([2, 1, 0]),
    "XYZ": np.array([0, 1, 2]),
}


# ====Helper functions for rotation_6d representation.====
# ====Use matrix as intermediate representation.====
def mtx_to_rot_6d(mtx):
    """Convert rotation matrix to rotation_6d.

    Args:
        mtx (np.ndarray): (n_action_steps, 4, 4) or (4, 4)
    Returns:
        np.ndarray: (n_action_steps, 6) or (6)
    """
    # TODO: Check if this is correct.
    return np.concatenate([mtx[..., 0, :], mtx[..., 1, :]], axis=-1)


def rot_6d_to_mtx(rot_6d):
    """Convert rotation_6d to rotation matrix.

    Args:
        rot_6d (np.ndarray): (n_action_steps, 6) or (6)
    Returns:
        np.ndarray: (n_action_steps, 4, 4) or (4, 4)
    """
    # TODO: Check if this is correct.
    rot_6d = rot_6d[..., np.newaxis, :]
    row_1 = rot_6d[..., :3] / np.linalg.norm(rot_6d[..., :3], axis=-1, keepdims=True)
    row_3 = np.cross(row_1, rot_6d[..., 3:6], axis=-1)
    row_3 /= np.linalg.norm(row_3, axis=-1, keepdims=True)
    row_2 = np.cross(row_3, row_1, axis=-1)
    mtx = np.concatenate([row_1, row_2, row_3], axis=-2)
    return mtx


# ====Helper functions for rotation arrays that are in [r,p,y] order but seq is not rpy.====
def rotation_from_rep(
    rotations: R,
    from_rep: Union[str, RotationRepr],
    seq=None,
    degrees=None,
    euler_in_rpy=True,
):
    """mainly handles euler angles: preserve the order to be RPY.

    Args:
        rotations (np.ndarray):
            (n_action_steps, *rotation_shape) or (*rotation_shape)
            - euler: (n_action_steps, 6) or (6); (x, y, z, r, p, y) if euler_in_rpy is True.
            - quat: (n_action_steps, 7) or (7); (x, y, z, qx, qy, qz, qw)
            - matrix: (n_action_steps, 4, 4) or (4, 4)
        from_rep (str): "euler", "quat", "matrix"
        seq (str, optional): convention for euler angles.
            None for quat and matrix. Defaults to None.
        degrees (bool, optional): _description_. Defaults to None.
        euler_in_rpy (bool, optional): _description_. Defaults to True.

    Returns:
        R: Rotation object
    """
    match from_rep:
        case RotationRepr.EULER:
            if euler_in_rpy:
                return R.from_euler(
                    seq, rotations[..., _EULER_RPY_TO_SEQ_MAPPING[seq]], degrees=degrees
                )
            else:
                return R.from_euler(seq, rotations, degrees=degrees)
        case RotationRepr.QUAT:
            return R.from_quat(rotations)
        case RotationRepr.MATRIX:
            return R.from_matrix(rotations)
        case RotationRepr.ROTATION_6D:
            return R.from_matrix(rot_6d_to_mtx(rotations))
        case RotationRepr.AXIS_ANGLE:
            return R.from_rotvec(rotations)
        case _:
            raise ValueError(
                f"Invalid from_rep: {from_rep}, \
                must be one of ['euler', 'quat', 'matrix']"
            )


def rotation_to_rep(
    rotations: R,
    to_rep: Union[str, RotationRepr],
    seq: str = None,
    degrees: bool = None,
    euler_in_rpy: bool = True,
):
    match to_rep:
        case RotationRepr.EULER:
            if euler_in_rpy:
                return rotations.as_euler(seq, degrees=degrees)[
                    ..., _EULER_SEQ_TO_RPY_MAPPING[seq]
                ]
            else:
                return rotations.as_euler(seq, degrees=degrees)
        case RotationRepr.QUAT:
            return rotations.as_quat()
        case RotationRepr.MATRIX:
            return rotations.as_matrix()
        case RotationRepr.ROTATION_6D:
            return mtx_to_rot_6d(rotations.as_matrix())
        case RotationRepr.AXIS_ANGLE:
            return rotations.as_rotvec()
        case _:
            raise ValueError(f"Invalid to_rep: {to_rep}")


class Rotation(R):
    @classmethod
    def identity(cls):
        return cls.from_quat([0.0, 0.0, 0.0, 1.0])



class Transform(object):
    """Rigid spatial transform between coordinate systems in 3D space.
    Attributes:
        rotation (R)
        translation (np.ndarray)
    """

    def __init__(self, rotation, translation):
        assert isinstance(rotation, R)
        assert isinstance(translation, (np.ndarray, list, torch.Tensor))

        self.rotation = rotation
        self.translation = np.asarray(translation, np.double)
        # self.parent_transform = None

    def as_matrix(self):
        """Represent as a 4x4 matrix."""
        rot_matrix = self.rotation.as_matrix()
        if rot_matrix.ndim == 2:
            return np.vstack(
                (np.c_[rot_matrix, self.translation], [0.0, 0.0, 0.0, 1.0])
            )
        elif rot_matrix.ndim == 3:
            transform_matrix = np.zeros((rot_matrix.shape[0], 4, 4), dtype=np.double)
            transform_matrix[:, :3, :3] = rot_matrix
            transform_matrix[:, :3, 3] = self.translation
            transform_matrix[:, 3, 3] = 1.0
            return transform_matrix
        else:
            raise ValueError(f"Invalid rotation matrix shape: {rot_matrix.shape}")

    def __mul__(self, other: 'Transform'):
        """Compose this transform with another."""
        rotation = self.rotation * other.rotation
        translation = self.rotation.apply(other.translation) + self.translation
        return self.__class__(rotation, translation)

    def transform_point(self, point):
        return self.rotation.apply(point) + self.translation

    def transform_vector(self, vector):
        return self.rotation.apply(vector)

    def inverse(self):
        """Compute the inverse of this transform."""
        rotation = self.rotation.inv()
        translation = -rotation.apply(self.translation)
        return self.__class__(rotation, translation)
    
    @classmethod
    def from_matrix(cls, m):
        """Initialize from a 4x4 matrix."""
        rotation = R.from_matrix(m[..., :3, :3])
        translation = m[..., :3, 3]
        return cls(rotation, translation)

    # ===== Avoid using these methods in data processing and policy learning
    # : order not compatible with our representation.====
    @classmethod
    def from_dict(cls, dictionary):
        rotation = R.from_quat(dictionary["rotation"])
        translation = np.asarray(dictionary["translation"])
        return cls(rotation, translation)

    @classmethod
    def from_list(cls, list):
        rotation = R.from_quat(list[:4])
        translation = list[4:]
        return cls(rotation, translation)

    def to_dict(self):
        """Serialize Transform object into a dictionary of quat and translation.
        {
            "rotation": [x, y, z, w],
            "translation": [x, y, z],
        }
        """
        return {
            "rotation": self.rotation.as_quat().tolist(),
            "translation": self.translation.tolist(),
        }

    def to_list(self):
        """TODO(deprecated): Not used: order not compatible with our representation.

        Returns:
            _type_: _description_
        """
        return np.r_[self.rotation.as_quat(), self.translation]


    @classmethod
    def identity(cls):
        """Initialize with the identity transformation."""
        rotation = R.from_quat([0.0, 0.0, 0.0, 1.0])
        translation = np.array([0.0, 0.0, 0.0])
        return cls(rotation, translation)

    @classmethod
    def look_at(cls, eye, center, up):
        """Initialize with a LookAt matrix.
        Returns:
            T_eye_ref, the transform from camera to the reference frame, w.r.t.
            which the input arguments were defined.
        """
        eye = np.asarray(eye)
        center = np.asarray(center)

        forward = center - eye
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        up = np.asarray(up) / np.linalg.norm(up)
        up = np.cross(right, forward)

        m = np.eye(4, 4)
        m[:3, 0] = right
        m[:3, 1] = -up
        m[:3, 2] = forward
        m[:3, 3] = eye

        return cls.from_matrix(m).inverse()

    @classmethod
    def look_at_ee(cls, ee_pos, center, y):
        """Initialize with a LookAt matrix.

        Args:
            ee (np.ndarray): (3,): ee translation point.
            center (np.ndarray): (3,): center point of gripper. z axis determined by (center - ee).
            y (np.ndarray): (3,): y axis positive direction. Eg for aarm, it's (gripper right - gripper left).
        """
        # ee_pos = np.asarray(ee_pos)
        # center = np.asarray(center)

        forward = center - ee_pos
        forward /= np.linalg.norm(forward, axis=-1, keepdims=True) # (N, 3)

        up = np.cross(y, forward, axis=-1)
        up /= np.linalg.norm(up, axis=-1, keepdims=True) # (N, 3)

        y = np.cross(forward, up, axis=-1)
        y /= np.linalg.norm(y, axis=-1, keepdims=True) # (N, 3)
        
        
        
        if ee_pos.ndim == 1:
            m = np.zeros((4, 4))
        else:
            m = np.zeros((ee_pos.shape[0], 4, 4))
        m[..., :3, 0] = up
        m[..., :3, 1] = y
        m[..., :3, 2] = forward
        m[..., :3, 3] = ee_pos
        
        
        # print("m.shape: ", m.shape, "up.shape: ", up.shape, "y.shape: ", y.shape, "forward.shape: ", forward.shape, "ee_pos.shape: ", ee_pos.shape)
        
        # m = np.eye(4, 4)
        # m[:3, 0] = up
        # m[:3, 1] = y
        # m[:3, 2] = forward
        # m[:3, 3] = ee_pos

        return cls.from_matrix(m)

    # ===== Unified methods for all representations. ====
    @classmethod
    def from_rep(cls, tfs, from_rep, seq=None, degrees=None, euler_in_rpy=True):
        """Supported representations:
            - euler: (n_action_steps, 6) or (6); (x, y, z, r, p, y) if euler_in_rpy is True.
            - quat: (n_action_steps, 7) or (7); (x, y, z, qx, qy, qz, qw)
            - matrix: (n_action_steps, 4, 4) or (4, 4)
            - axis_angle: (n_action_steps, 7) or (7); (x, y, z, ax, ay, az, angle)
            - rotation_6d: (n_action_steps, 9) or (9); (x, y, z, a00, a01, a02, a10, a11, a12)

        Args:
            tfs (np.ndarray): (n_action_steps, *tf_shape) or (*tf_shape)
            from_rep (str): "euler", "quat", "matrix"
            seq (str, optional): convention for euler angles. None for quat and matrix. Defaults to None.
            degrees (bool, optional): _description_. Defaults to None.

        Returns:
            Transform: Transform object
        """
        if from_rep == "euler":
            assert tfs.shape[-1] == 6, "action must be (n_action_steps, 6) for euler angles"
            # assert action.shape[1] == 6, "action must be (n_action_steps, 6) for euler angles"
            # assert pose.shape[1] == 6, "pose must be (n_pose_steps, 6) for euler angles"
            # assert seq is not None, "rep_convention must be provided for euler angles"
            if euler_in_rpy:
                # Order of euler angles is rpy in the arr. Very common for the states we get from robot.
                rotation = R.from_euler(
                    seq, tfs[..., _EULER_RPY_TO_SEQ_MAPPING[seq]+3], degrees=degrees
                )
            else:
                # Order of euler angles has been reordered to conform to the seq order.
                rotation = R.from_euler(
                    seq, tfs, degrees=degrees
                )
            return cls(
                rotation=rotation, translation=tfs[..., :3]
            )
        elif from_rep == "rotation_6d":
            assert tfs.shape[-1] == 9, "action must be (n_action_steps, 9) for rotation_6d"
            mtx = rot_6d_to_mtx(tfs[..., 3:9])
            return cls(
                rotation=R.from_matrix(mtx),
                translation=tfs[..., :3]
            )
        elif from_rep == "quat":
            assert tfs.shape[-1] == 7, "action must be (n_action_steps, 7) for quaternion"
            # assert action.shape[1] == 7, "action must be (n_action_steps, 7) for quaternion"
            # assert pose.shape[1] == 7, "pose must be (n_pose_steps, 7) for quaternion"
            return cls(
                rotation=R.from_quat(tfs[..., 3:7]),
                translation=tfs[..., :3]
            )
        elif from_rep == "matrix":
            assert tfs.shape[-2:] == (4, 4), "action must be (n_action_steps, 4, 4) for matrix"
            # assert action.shape[1:] == (4, 4), "action must be (n_action_steps, 4, 4) for matrix"
            # assert pose.shape[1:] == (4, 4), "pose must be (n_pose_steps, 4, 4) for matrix"
            return cls(
                rotation=R.from_matrix(tfs[..., :3, :3]),
                translation=tfs[..., :3, 3]
            )
        # elif from_rep == "axis_angle":
        #     # TODO: Test this axis angle support.
        #     raise NotImplementedError("Axis angle not supported yet.")
        #     return cls(
        #         rotation=R.from_rotvec(tfs[..., 3:7]),
        #         translation=tfs[..., :3]
        #     )
        else:
            raise ValueError(f"Invalid from_rep: {from_rep}, must be one of ['euler', 'quat', 'matrix']")

    def to_rep(self, to_rep, seq=None, degrees=None, euler_in_rpy=True):
        """Currently support euler, rotation_6d, axis_angle, quat and matrix.

        Args:
            to_rep (str): "euler", "quat", "matrix", "rotation_6d", "axis_angle"
            seq (str, optional): convention for euler angles.
                None for quat and matrix. Defaults to None.
            degrees (bool, optional): _description_. Defaults to None.
        """
        if to_rep == "euler":
            if euler_in_rpy:
                rotation = self.rotation.as_euler(
                    seq, degrees=degrees
                )[..., _EULER_SEQ_TO_RPY_MAPPING[seq]]
            else:
                rotation = self.rotation.as_euler(
                    seq, degrees=degrees
                )
            return np.hstack([self.translation, rotation])
        elif to_rep == "rotation_6d":
            rot_mtx = self.rotation.as_matrix()
            rot_6d = mtx_to_rot_6d(rot_mtx)
            return np.hstack([self.translation, rot_6d])
        elif to_rep == "quat":
            return np.hstack([self.translation, self.rotation.as_quat()])
        elif to_rep == "matrix":
            return self.as_matrix()
        elif to_rep == "axis_angle":
            # TODO: Test this axis angle support.
            raise NotImplementedError("Axis angle not supported yet.")
            return np.hstack([self.translation, self.rotation.as_rotvec()])
        else:
            raise ValueError(f"Invalid to_rep: {to_rep}, must be one of ['euler', 'quat', 'matrix']")

    @classmethod
    def convert(cls, tf: np.ndarray, from_rep, to_rep, seq=None, degrees=None, euler_in_rpy=True, gripper_scale=None, tcp_offset=None):
        """Class method to convert transform between different representations directly.

        Args:
            tf (np.ndarray): tf in either:
                - homogeneous matrix (N, 4, 4)
                - pos + euler (N, 6)
                - pos + quat (N, 7)
                - pos + rotation_6d (N, 9)
                - pos + axis_angle (N, 7),
                where N = 0, 1, or multiple.
            from_rep (str): "matrix", "euler", "quat", "rotation_6d", "axis_angle"
            to_rep (str): "matrix", "euler", "quat", "rotation_6d", "axis_angle"
            seq (str, optional): convention for euler angles.
            degrees (bool, optional): Whether euler angles are in degrees. Defaults to None.
            euler_in_rpy (bool, optional): Whether euler angles in the arrary are in rpy order. Defaults to True.
        """
        return cls.from_rep(
            tf, from_rep, seq, degrees, euler_in_rpy
        ).to_rep(
            to_rep, seq, degrees, euler_in_rpy
        )

    def to_relative(self, parent_transform: 'Transform'):
        # assert self.parent_transform is None,\
        #     "Parent transform is already set."
        # self.parent_transform = parent_transform
        relative_transform = parent_transform.inverse() * self
        # relative_transform.parent_transform = parent_transform
        return relative_transform

    def to_absolute(self, parent_transform: 'Transform'):
        return parent_transform * self

if __name__ == "__main__":

    # tf = Transform.look_at(
    #     eye = np.array([1.0, 1.0, 1.0]),
    #     center = np.array([1.0, 0.0, 0.0]),
    #     up = np.array([0.0, 0.0, 1.0])
    # )
    # print(tf.as_matrix())
    tf = Transform.look_at_ee(
        ee_pos=np.array([0.0, 0.0, 0.0]),
        center=np.array([1.0, 0.0, 0.0]),
        y=np.array([0.0, -1.0, 0.0]),
    )
    mtx1 = tf.as_matrix()

    tf = Transform.from_rep(
        tfs=np.array([0, 0, 0, -90, -90, -90]),
        from_rep="euler",
        seq="ZYX",
        degrees=True,
        euler_in_rpy=True,
    )
    print(np.allclose(mtx1, tf.as_matrix()))

    ee_pose_6d = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    right_ee_pose_quat = Transform.convert(
        tf=ee_pose_6d,    
        from_rep="rotation_6d",        
        to_rep= "quat"  
    )
    print(right_ee_pose_quat)