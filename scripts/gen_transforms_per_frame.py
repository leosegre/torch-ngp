import sys
import shutil
import os
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def write_json(filename, transform):
    # out = {
    #     "camera_angle_x": json_params['camera_angle_x'],
    #     "camera_angle_y": json_params['camera_angle_x'],
    #     "fl_x": json_params['fl_x'],
    #     "fl_y": json_params['fl_y'],
    #     "k1": json_params['k1'],
    #     "k2": json_params['k2'],
    #     "p1": json_params['p1'],
    #     "p2": json_params['p2'],
    #     "cx": json_params['cx'],
    #     "cy": json_params['cy'],
    #     "w": json_params['w'],
    #     "h": json_params['h'],
    #     "frames": json_params['frames'],
    # }

    output_path = os.path.join(filename)
    print(f"[INFO] writing {len(transform['frames'])} frames to {output_path}")
    with open(output_path, "w") as outfile:
        json.dump(transform, outfile, indent=2)


def get_transform_matrix(x, y, z):
    # Create a translation matrix to move the origin to (x,y,z)
    translation_matrix = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

    # Create a rotation matrix that rotates the vector (x,y,z) to the positive z-axis
    rotation_matrix = np.array([
        [x/np.sqrt(x**2 + y**2 + z**2), y/np.sqrt(x**2 + y**2 + z**2), 0, 0],
        [-y/np.sqrt(x**2 + y**2), x/np.sqrt(x**2 + y**2), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Multiply the translation and rotation matrices to get the transform matrix
    transform_matrix = np.dot(translation_matrix, rotation_matrix)

    return transform_matrix


def nerf_matrix_to_ngp(pose, scale=1, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def ngp_matrix_to_nerf(pose):
    # reverse the operations in nerf_matrix_to_ngp to get the original matrix
    new_pose = np.array([
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3]],
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def get_transform_matrix_by_radius(r, theta, phi):
    # Convert angles to radians
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)

    # Calculate x, y, and z coordinates
    x = r * np.sin(theta_rad) * np.cos(phi_rad)
    y = r * np.sin(theta_rad) * np.sin(phi_rad)
    z = r * np.cos(theta_rad)

    # Create a translation matrix to move the origin to (x,y,z)
    translation_matrix = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

    # Create a rotation matrix that rotates the vector (x,y,z) to the positive z-axis
    rotation_matrix = np.array([
        [x/np.sqrt(x**2 + y**2 + z**2), y/np.sqrt(x**2 + y**2 + z**2), 0, 0],
        [-y/np.sqrt(x**2 + y**2), x/np.sqrt(x**2 + y**2), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Multiply the translation and rotation matrices to get the transform matrix
    transform_matrix = np.dot(translation_matrix, rotation_matrix)

    return transform_matrix


def rand_poses(size, device, radius=1, theta_range=[0, 4 * (np.pi / 9)], phi_range=[0, 2 * np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''

    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size,
                                                                             1)  # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    # print(poses)

    return poses


def intrinsics_into_frames(transform):
    out = {
        "camera_angle_x": transform['camera_angle_x'],
        "camera_angle_y": transform['camera_angle_x'],
        "fl_x": transform['fl_x'],
        "fl_y": transform['fl_y'],
        "k1": transform['k1'],
        "k2": transform['k2'],
        "p1": transform['p1'],
        "p2": transform['p2'],
        "cx": transform['cx'],
        "cy": transform['cy'],
        "w": transform['w'],
        "h": transform['h'],
    }

    for frame in transform['frames']:
        for key in out:
            frame[key] = out[key]

    for key in out:
        transform.pop(key)



def main(file_name):
    with open(os.path.join(file_name), 'r') as f:
        transform = json.load(f)
    intrinsics_into_frames(transform)
    write_json(file_name, transform)

if __name__ == "__main__":
   main(sys.argv[1])
