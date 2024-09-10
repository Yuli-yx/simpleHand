import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm

class SimplifiedFHBHandsDataset(Dataset):
    def __init__(self, dataset_folder, split='train', input_res=(1920, 1080)):
        self.input_res = input_res
        self.split = split

        # 摄像机外参和内参
        self.cam_extr = np.array([[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
                                  [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                                  [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
                                  [0, 0, 0, 1]])
        self.cam_intr = np.array([[1395.749023, 0, 935.732544], 
                                  [0, 1395.749268, 540.681030], 
                                  [0, 0, 1]])
        
        self.reorder_idx = np.array(
            [0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20]
        )

        # 分辨率缩放因子设为 1，因为我们使用原始尺寸
        self.reduce_factor = 1

        self.joints2d = []
        self.joints3d = []
        self.image_names = []

        # 加载关节信息
        self.load_dataset(dataset_folder)

    def load_dataset(self, dataset_folder):
        """加载图片路径和关节点信息"""
        skeleton_root = os.path.join(dataset_folder, "Hand_pose_annotation_v1")
        info_root = os.path.join(dataset_folder, "Subjects_info")
        info_split = os.path.join(dataset_folder, "data_split_action_recognition.txt")

        # 加载 subject 信息
        subjects_infos = self.load_subject_infos(info_root)
        skel_info = self.get_skeletons(skeleton_root, subjects_infos)

        # 加载数据分割信息
        with open(info_split, "r") as annot_f:
            lines_raw = annot_f.readlines()

        # print("lines_raw 0", lines_raw[0])
        train_list, test_list, all_infos = self.get_action_train_test(lines_raw, subjects_infos)
        
        print(f"Train samples: {len(train_list)}")
        print(f"Test samples: {len(test_list)}")

        # 根据split选择使用哪个数据集
        if self.split == "train":
            sample_list = train_list
        elif self.split == "test":
            sample_list = test_list
        else:
            raise ValueError(f"Split {self.split} 不合法，应为 [train|test]")

        for subject, action_name, seq_idx, frame_idx in sample_list:
            img_path = os.path.join(dataset_folder, "Video_files", subject, action_name, seq_idx, "color", f"color_{frame_idx:04d}.jpeg")
            self.image_names.append(img_path)

            # 获取并处理 3D 和 2D 关节坐标
            skel = skel_info[subject][(action_name, seq_idx)][frame_idx]
            skel = skel[self.reorder_idx]
            
            skel_homo = np.hstack([skel, np.ones((skel.shape[0], 1))])
            # skel_camcoords = self.cam_extr.dot(skel.transpose()).transpose()[:, :3].astype(np.float32)
            skel_camcoords = self.cam_extr.dot(skel_homo.T).T[:, :3].astype(np.float32)
            self.joints3d.append(skel_camcoords)
            hom_2d = np.array(self.cam_intr).dot(skel_camcoords.transpose()).transpose()
            skel2d = (hom_2d / hom_2d[:, 2:])[:, :2]
            self.joints2d.append(skel2d.astype(np.float32))

    def load_subject_infos(self, info_root):
        """加载 subjects 信息"""
        subjects_infos = {}
        for subject_file in os.listdir(info_root):
            subject_info_path = os.path.join(info_root, subject_file)
            subject_name = os.path.splitext(subject_file)[0][:9]
            with open(subject_info_path, "r") as subject_f:
                raw_lines = subject_f.readlines()
                subjects_infos[subject_name] = {}
                for line in raw_lines[3:]:
                    line = " ".join(line.split())
                    action, action_idx, length = line.strip().split(" ")
                    subjects_infos[subject_name][(action, action_idx)] = int(length)
        return subjects_infos

    def get_skeletons(self, skeleton_root, subjects_info):
        """加载 skeletons"""
        skelet_dict = defaultdict(dict)
        for subject, samples in tqdm(subjects_info.items(), desc="subj"):
            for (action, seq_idx) in tqdm(samples, desc="sample"):
                skeleton_path = os.path.join(skeleton_root, subject, action, seq_idx, "skeleton.txt")
                skeleton_vals = np.loadtxt(skeleton_path)
                if len(skeleton_vals):
                    assert np.all(
                        skeleton_vals[:, 0] == list(range(skeleton_vals.shape[0]))
                    ), f"row idxs should match frame idx failed at {skeleton_path}"
                    skelet_dict[subject][(action, seq_idx)] = skeleton_vals[:, 1:].reshape(
                        skeleton_vals.shape[0], 21, -1
                    )
        return skelet_dict

    def get_action_train_test(self, lines_raw, subjects_info):
        """加载训练和测试数据"""
        all_infos = []
        test_split = False
        test_samples = {}
        train_samples = {}
        for line in lines_raw:
            if line.startswith("Train"):
                test_split = False
                continue
            elif line.startswith("Test"):
                test_split = True
                continue
            subject, action_name, action_seq_idx = line.split(" ")[0].split("/")
            action_idx = line.split(" ")[1].strip()  # 动作分类索引
            frame_nb = int(subjects_info[subject][(action_name, action_seq_idx)])
            for frame_idx in range(frame_nb):
                sample_info = (subject, action_name, action_seq_idx, frame_idx)
                if test_split:
                    test_samples[sample_info] = action_idx
                else:
                    train_samples[sample_info] = action_idx
                all_infos.append(sample_info)
        return train_samples, test_samples, all_infos

    def get_image(self, idx):
        """获取图像"""
        img_path = os.path.join(self.image_names[idx])
        img = Image.open(img_path).convert("RGB")
        return img

    def get_joints3d(self, idx):
        """获取3D关节坐标"""
        return self.joints3d[idx] / 1000  # 转换为米单位

    def get_joints2d(self, idx):
        """获取2D关节坐标"""
        return self.joints2d[idx] * self.reduce_factor

    def get_joints25d(self, idx):
        """获取2.5D 关节坐标"""
        joints2d = self.get_joints2d(idx)
        joints3d = self.get_joints3d(idx)

        # 将 2D 坐标的 (x, y) 和 3D 的深度信息 (z) 组合起来，形成 2.5D 坐标
        joints25d = np.concatenate([joints2d, joints3d[:, 2:]], axis=1)
        return joints25d

    def __getitem__(self, idx):
        """获取单个样本，包括图像和对应的关节坐标"""
        img = self.get_image(idx)
        joints3d = self.get_joints3d(idx)
        joints2d = self.get_joints2d(idx)
        joints25d = self.get_joints25d(idx)

        sample = {
            'image': img,
            'joints2d': joints2d,
            'joints3d': joints3d,
            'joints25d': joints25d
        }
        return sample

    def __len__(self):
        return len(self.image_names)
    

if __name__ == "__main__":
    dataset_root = "/media/mldadmin/home/s123mdg31_07/Datasets/FPHAB"
    dataset = SimplifiedFHBHandsDataset(dataset_folder=dataset_root, split='train')

    # 获取第一个样本
    sample = dataset[0]
    image = sample['image']
    joints2d = sample['joints2d']
    joints3d = sample['joints3d']
    joints25d = sample['joints25d']

    # 打印关节点信息
    print(joints2d)
    print(joints3d)
    print(joints25d)