import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
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
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10, num_classes=6, use_class_vector=False):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.num_classes = num_classes
        self.use_class_vector = use_class_vector

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                if self.opt.generate:
                    with open(os.path.join(self.root_path, f'transforms_to_generate.json'), 'r') as f:
                        print(f.name)
                        transform = json.load(f)
                else:
                    with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                        print(f.name)
                        transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size TODO: don't assume all images are in the same size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['frames'][0]['h']) // downscale
            self.W = int(transform['frames'][0]['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None

        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.poses = []
            self.images = []
            self.semantics = []
            self.intrinsics = []
            self.original_name = []
            if self.opt.dist_load:
                self.dist_images = []



            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])
                s_path = os.path.join(self.root_path, f['semantic_path'])
                if self.opt.dist_load:
                    d_path = os.path.join(self.root_path, f['dist_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path) and not self.opt.generate:
                    print("The file", f_path, "does not exist")
                    continue

                # there are non-exist paths in Semantic...
                if not os.path.exists(s_path) and not self.opt.generate:
                    print("The file", s_path, "does not exist")
                    continue

                # Dist from middle of shape
                if self.opt.dist_load:
                    # there are non-exist paths in Dist...
                    if not os.path.exists(d_path) and not self.opt.generate:
                        print("The file", d_path, "does not exist")
                        continue

                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                # if not self.opt.generate:
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)


                if self.opt.generate:
                    image = np.zeros((256, 256, 3), np.uint8)
                else:
                    image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]

                # Dist from middle of shape
                if self.opt.dist_load:
                    if self.opt.generate:
                        dist_image = np.zeros((256, 256, 1), np.uint8)
                    else:
                        dist_image = cv2.imread(d_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]

                if self.opt.generate:
                    semantic = np.zeros((256, 256, 1), np.uint8)
                else:
                    if self.use_class_vector:
                        for class_num in range(self.num_classes):
                            s_path_class = s_path.split('.')[0]+"_"+str(class_num)+"."+s_path.split('.')[-1]
                            temp_semantic = cv2.imread(s_path_class, cv2.IMREAD_GRAYSCALE) # [H, W]
                            if class_num == 0:
                                semantic = np.expand_dims(temp_semantic, axis=2)  # [H, W, 1]
                            else:
                                temp_semantic = np.expand_dims(temp_semantic, axis=2)  # [H, W, 1]
                                semantic = np.concatenate((semantic, temp_semantic), axis=2)
                        semantic = (semantic / 255)  # Normalize and use floating point
                    else:
                        semantic = cv2.imread(s_path, cv2.IMREAD_GRAYSCALE) # [H, W]
                        semantic = np.expand_dims(semantic, axis=2) # [H, W, 1]
                    if self.opt.use_class_vector:
                        semantic = torch.from_numpy(semantic).to(dtype=torch.float16)
                    if self.H is None or self.W is None:
                        self.H = image.shape[0] // downscale
                        self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    semantic = cv2.resize(semantic, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                # load intrinsics
                if 'fl_x' in f or 'fl_y' in f:
                    fl_x = (f['fl_x'] if 'fl_x' in f else f['fl_y']) / downscale
                    fl_y = (f['fl_y'] if 'fl_y' in f else f['fl_x']) / downscale
                elif 'camera_angle_x' in f or 'camera_angle_y' in f:
                    # blender, assert in radians. already downscaled since we use H/W
                    fl_x = self.W / (
                                2 * np.tan(f['camera_angle_x'] / 2)) if 'camera_angle_x' in f else None
                    fl_y = self.H / (
                                2 * np.tan(f['camera_angle_y'] / 2)) if 'camera_angle_y' in f else None
                    if fl_x is None: fl_x = fl_y
                    if fl_y is None: fl_y = fl_x
                else:
                    raise RuntimeError('Failed to load focal length, please check the transforms.json!')

                cx = (f['cx'] / downscale) if 'cx' in f else (self.W / 2)
                cy = (f['cy'] / downscale) if 'cy' in f else (self.H / 2)

                intrinsics = np.array([fl_x, fl_y, cx, cy])

                self.poses.append(pose)
                self.images.append(image)
                if self.opt.dist_load:
                    self.dist_images.append(dist_image)
                self.semantics.append(semantic)
                self.intrinsics.append(intrinsics)
                if self.opt.generate or opt.generate_dist:
                    self.original_name.append(f['file_path'])


        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        # Dist from middle of shape
        if self.intrinsics is not None:
            self.intrinsics = torch.from_numpy(np.stack(self.intrinsics, axis=0)) # [N, 4]
        if self.opt.dist_load:
            if self.dist_images is not None:
                self.dist_images = torch.from_numpy(np.stack(self.dist_images, axis=0)) # [N, H, W, 1]
        if self.semantics is not None:
            if self.use_class_vector:
                self.semantics = torch.stack(self.semantics, axis=0) # [N, H, W, 1]
            else:
                self.semantics = torch.from_numpy(np.stack(self.semantics, axis=0)) # [N, H, W, 1]
        if self.opt.generate or opt.generate_dist:
            self.original_name = np.stack(self.original_name, axis=0) # [N, 1]

        # Calculate number of semantic classes
        # if self.opt.use_class_vector:
        #     self.semantic_classes = torch.unique(self.semantics)
        # else:
        self.semantic_classes = np.unique(self.semantics)
        self.num_semantic_class = self.semantic_classes.shape[0]  # number of semantic classes, including the void class of 0
        print("num_semantic_class:", self.num_semantic_class)
        print("max_semantic_label:", self.semantics.max())
        print("Unique semantics:", self.semantic_classes)

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            self.intrinsics = self.intrinsics.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
                self.semantics = self.semantics.to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)




    def collate(self, index):

        B = len(index) # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]


        rays = get_rays(poses, self.intrinsics[index].squeeze(), self.H, self.W, self.num_rays, error_map, self.opt.patch_size)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        # Dist from middle of shape
        if self.opt.dist_load:
            if self.dist_images is not None:
                dist_images = self.dist_images[index].to(self.device) # [B, H, W, 1]
                if self.training:
                    dist_images = torch.gather(dist_images.view(B, -1, 1), 1, torch.stack([rays['inds']], -1)) # [B, N, 1]
                results['dist_images'] = dist_images

        if self.semantics is not None:
            semantics = self.semantics[index].to(self.device) # [B, H, W, 1]
            # TODO: change segmentation map to more general cases
            # seg_map = [0, 3, 11, 12, 13, 18, 19, 20, 29, 31, 37, 40, 44, 47, 59, 60, 63, 64, 65, 76, 78, 79, 80, 91, 92, 93, 95, 97, 98]
            # seg_map = [0, 1, 2, 3, 4, 5]
            seg_map = [0, 97, 193, 144, 157, 61]
            # for i in range(len(seg_map)):
            #     temp_semantics = (semantics == seg_map[i])
            #     temp_semantics = temp_semantics.cpu().numpy()[0] * 256
            #     cv2.imwrite("./labeling_test/"+str(seg_map[i])+".png", temp_semantics)

            if self.training:
                if self.use_class_vector:
                    semantics = torch.gather(semantics.view(B, -1, self.num_classes), 1, torch.stack(self.num_classes * [rays['inds']], -1)) # [B, N, 1]
                else:
                    semantics = torch.gather(semantics.view(B, -1, 1), 1, torch.stack([rays['inds']], -1)) # [B, N, num_classes]
            for i in range(len(seg_map)):
                semantics[semantics == seg_map[i]] = i
            results['semantics'] = semantics
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']

        if self.opt.generate or self.opt.generate_dist:
            results['original_name'] = self.original_name[index]
            
        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader