
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
import timm
from timm.models.layers import DropPath, Mlp
import hiera

from models.modules import MeshHead, JointsHead, AttentionBlock, IdentityBlock, SepConvBlock
from models.losses import mesh_to_joints
from models.losses import l1_loss



class HandNet(nn.Module):
    def __init__(self, cfg, pretrained=None):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["MODEL"]
        backbone_cfg = model_cfg["BACKBONE"]

        self.loss_cfg = model_cfg["LOSSES"]

        if pretrained is None:
            pretrained=backbone_cfg['pretrain']            

        if "hiera" in backbone_cfg['model_name']:
            self.backbone = hiera.__dict__[backbone_cfg['model_name']](pretrained=True, checkpoint="mae_in1k",  drop_path_rate=backbone_cfg['drop_path_rate'])
            self.is_hiera = True
        else:
            self.backbone = timm.create_model(backbone_cfg['model_name'], pretrained=pretrained, drop_path_rate=backbone_cfg['drop_path_rate'])
            self.is_hiera = False
            
        self.avg_pool = nn.AvgPool2d((7, 7), 1)            

        uv_cfg = model_cfg['UV_HEAD']
        depth_cfg = model_cfg['DEPTH_HEAD']

        self.keypoints_2d_head = nn.Linear(uv_cfg['in_features'], uv_cfg['out_features'])
        # self.depth_head = nn.Linear(depth_cfg['in_features'], depth_cfg['out_features'])
        # self.post_process_handjoints = To25DBranch(trans_factor=model_cfg['TRANS_FACTOR'], scale_factor=model_cfg['SCALE_FACTOR'])
        self.use_joint_head = False
        if "MESH_HEAD" in model_cfg.keys():
            
            mesh_head_cfg = model_cfg["MESH_HEAD"].copy()
            
            block_types_name = mesh_head_cfg['block_types']
            block_types = []
            block_map = {
                "attention": AttentionBlock,
                "identity": IdentityBlock,
                "conv": SepConvBlock,
            }
            
            for name in block_types_name:
                block_types.append(block_map[name])
            mesh_head_cfg['block_types'] = block_types
        
            self.mesh_head = MeshHead(**mesh_head_cfg)   
        elif "JOINTS_HEAD" in model_cfg.keys():
            self.use_joint_head = True
            joints_head_cfg = model_cfg["JOINTS_HEAD"].copy()
            
            block_types_name = joints_head_cfg['block_types']
            block_types = []
            block_map = {
                "attention": AttentionBlock,
                "identity": IdentityBlock,
                "conv": SepConvBlock,
            }
            
            for name in block_types_name:
                block_types.append(block_map[name])
            joints_head_cfg['block_types'] = block_types
        
            self.mesh_head = JointsHead(**joints_head_cfg)
        else:
            raise ValueError("No head is defined")


    def infer(self, image, batch_data):
        input_res = image.shape[2:]
        if self.is_hiera:
            x, intermediates = self.backbone(image, return_intermediates=True)
            features = intermediates[-1]
            features = features.permute(0, 3, 1, 2).contiguous()
        else:
            features = self.backbone.forward_features(image)
        
        assert not torch.isnan(features).any(), "NaN detected in backbone features"
        
        global_feature = self.avg_pool(features).squeeze(-1).squeeze(-1)
        uv = self.keypoints_2d_head(global_feature)     
        # depth = self.depth_head(global_feature)
        assert not torch.isnan(global_feature).any(), "NaN detected in global_feature"
        
        
        
        if self.use_joint_head:
            joints = self.mesh_head(features, uv)
            assert not torch.isnan(joints).any(), "NaN detected in joints before post-processing"
            # joints = self.post_process_handjoints(batch_data, joints, input_res)
            
            # for k, v in joints.items():
            #     error = []
            #     if torch.isnan(v).any():
            #         error.append(k)
            # assert len(error) == 0, f"NaN detected in joints after post-processing: {error}"
            
            
            
            return {
                "uv": uv,
                # "root_depth": depth,
                "joints": joints,
                # "vertices": vertices,            
            }

        else:
            vertices = self.mesh_head(features, uv)
            joints = mesh_to_joints(vertices)

            return {
                "uv": uv,
                # "root_depth": depth,
                "joints": joints,
                "vertices": vertices,            
            }


    def forward(self, image, target=None):
        """get training loss

        Args:
            inputs (dict): {
                'img': (B, 1, H, W), 
                "uv": [B, 21, 2],
                "xyz": [B,  21, 3],
                "hand_uv_valid": [B, 21],
                "gamma": [B, 1],    

                "vertices": [B, 778, 3],
                "xyz_valid": [B,  21],
                "verts_valid": [B, 1],
                "hand_valid": [B, 1],
            }     
        """
        image = image / 255 - 0.5
        output_dict = self.infer(image, target)
        if self.training:
            assert target is not None
            loss_dict = self._cal_single_hand_losses(output_dict, target, use_verticesloss=(self.use_joint_head==False))
            # loss_dict = self._cal_hand_loss(output_dict, target, use_verticesloss=(self.use_joint_head==False))
            return loss_dict

        return output_dict

    def _cal_hand_loss(self, pred_hand_dict, gt_hand_dict, use_verticesloss=True):

            # "rep2d": est_xy0, 
            # "rep_absz": est_Z0,
            # "rep3d": est_c3d,
        joints_2d_pred = pred_hand_dict['rep2d']
        joints_absz_pred = pred_hand_dict['rep_absz']
        joints_3d_pred = pred_hand_dict['rep3d']
        if use_verticesloss:
            vertices_pred = pred_hand_dict['vertices']


        joints_25d_gt = gt_hand_dict['joints25d']
        joints_3d_gt = gt_hand_dict['xyz']
        
        joints_2d_gt = joints_25d_gt[:, :, :2]
        joints_absz_gt = joints_25d_gt[:, :, 2:]
        # root_depth_pred = root_depth_pred.reshape(-1, 1).contiguous()
        uv_gt = gt_hand_dict['uv']

        hand_2d_valid = gt_hand_dict['uv_valid']
        hand_absz_valid = hand_2d_valid
        hand_xyz_valid = gt_hand_dict['xyz_valid'] # N, 1
        if use_verticesloss:
            vertices_gt = gt_hand_dict['vertices']

        uv_loss = l1_loss(joints_2d_pred, uv_gt, hand_2d_valid)
        depth_loss = l1_loss(joints_absz_pred, joints_absz_gt, hand_absz_valid)
        # joints_loss = l1_loss(joints_pred, joints_gt, valid=hand_xyz_valid)
        if use_verticesloss:
            vertices_loss = l1_loss(vertices_pred, vertices_gt, valid=hand_xyz_valid)


        # joints_3d_loss = l1_loss(joints_3d_pred, joints_3d_gt, valid=hand_xyz_valid)


        loss_dict = {
            "uv_loss": uv_loss * self.loss_cfg["UV_LOSS_WEIGHT"],
            "depth_loss": depth_loss * self.loss_cfg["DEPTH_LOSS_WEIGHT"],
            # "joints_3d_loss": joints_3d_loss * self.loss_cfg["JOINTS_LOSS_WEIGHT"],                      
        }
        if use_verticesloss:
            loss_dict["vertices_loss"] = vertices_loss * self.loss_cfg["VERTICES_LOSS_WEIGHT"]

        total_loss = 0
        for k in loss_dict:
            total_loss += loss_dict[k]

        loss_dict['total_loss'] = total_loss
        
        return loss_dict
        
        
    def _cal_single_hand_losses(self, pred_hand_dict, gt_hand_dict, use_verticesloss=True):
        """get training loss

        Args:
            pred_hand_dict (dict): {
                'uv': [B, 21, 2],
                'root_depth': [B, 1],
                'joints': [B, 21, 3],
                # 'vertices': [B, 778, 2],
            },
            gt_hand_dict (dict): {
                'uv': [B, 21, 2],
                'xyz': [B, 21, 3],
                'gamma': [B, 1],
                'uv_valid': [B, 21],
                # 'vertices': [B, 778, 3],
                # 'xyz_valid': [B, 21],
                # 'verts_valid': [B, 1],
            },            

        """
        uv_pred = pred_hand_dict['uv']
        # root_depth_pred = pred_hand_dict['root_depth']
        # print(pred_hand_dict)
        joints_pred = pred_hand_dict["joints"]
        if use_verticesloss:
            vertices_pred = pred_hand_dict['vertices']


        uv_pred = uv_pred.reshape(-1, 21, 2).contiguous()
        
        joints_pred = joints_pred.reshape(-1, 21, 3).contiguous()
        # root_depth_pred = root_depth_pred.reshape(-1, 1).contiguous()
        uv_gt = gt_hand_dict['uv']
        joints_gt = gt_hand_dict['xyz']
        # root_depth_gt = gt_hand_dict['gamma'].reshape(-1, 1).contiguous()
        hand_uv_valid = gt_hand_dict['uv_valid']
        hand_xyz_valid = gt_hand_dict['xyz_valid'] # N, 1
        if use_verticesloss:
            vertices_gt = gt_hand_dict['vertices']

        uv_loss = l1_loss(uv_pred, uv_gt, hand_uv_valid)
        joints_loss = l1_loss(joints_pred, joints_gt, valid=hand_xyz_valid)
        if use_verticesloss:
            vertices_loss = l1_loss(vertices_pred, vertices_gt, valid=hand_xyz_valid)


        # root_depth_loss = (torch.abs(root_depth_pred- root_depth_gt)).mean()
        # root_depth_loss = root_depth_loss.mean()


        loss_dict = {
            "uv_loss": uv_loss * self.loss_cfg["UV_LOSS_WEIGHT"],
            "joints_loss": joints_loss * self.loss_cfg["JOINTS_LOSS_WEIGHT"],
            # "root_depth_loss": root_depth_loss * self.loss_cfg["DEPTH_LOSS_WEIGHT"],
                        
        }
        if use_verticesloss:
            loss_dict["vertices_loss"] = vertices_loss * self.loss_cfg["VERTICES_LOSS_WEIGHT"]

        total_loss = 0
        for k in loss_dict:
            total_loss += loss_dict[k]

        loss_dict['total_loss'] = total_loss
        
        return loss_dict



class To25DBranch(nn.Module):
    def __init__(self, trans_factor=1, scale_factor=1):
        """
        Args:
            trans_factor: Scaling parameter to insure translation and scale
                are updated similarly during training (if one is updated 
                much more than the other, training is slowed down, because
                for instance only the variation of translation or scale
                significantly influences the final loss variation)
            scale_factor: Scaling parameter to insure translation and scale
                are updated similarly during training
        """
        super(To25DBranch, self).__init__()
        self.trans_factor = trans_factor
        self.scale_factor = scale_factor
        self.inp_res = [256, 256]

    def forward(self, sample, scaletrans, input_res=(224, 224)):     
        batch_size = scaletrans.shape[0]
        trans = scaletrans[:, :, :2]
        scale = scaletrans[:, :, 2]
        final_trans = trans.view(batch_size,-1, 2)* self.trans_factor               # xy0 transfactor=100
        final_scale = scale.view(batch_size,-1, 1)* self.scale_factor               # z0 scalefactor=0.0001
        assert not torch.isnan(final_trans).any(), "NaN in final_trans"
        assert not torch.isnan(final_scale).any(), "NaN in final_scale"
        camintr = sample["cam_intr"]
        
        est_xy0,est_Z0, est_c3d=recover_3d_from_25d_pinhole(camintr=camintr,depth=final_scale,joints2d=final_trans,input_res=input_res)
        
        assert not torch.isnan(est_xy0).any(), "NaN in est_xy0"
        assert not torch.isnan(est_Z0).any(), "NaN in est_Z0"
        assert not torch.isnan(est_c3d).any(), "NaN in est_c3d"
        
        return {
            "rep2d": est_xy0, 
            "rep_absz": est_Z0,
            "rep3d": est_c3d,
        }
        
def recover_3d_from_25d_pinhole(camintr, depth, joints2d, off_z=0.4, input_res=(224, 224)):
    focal = camintr[:, :1, :1]
    batch_size = joints2d.shape[0]
    num_joints = joints2d.shape[1]
    focal = focal.view(batch_size, 1, 1)
    depth = depth.view(batch_size, -1, 1)# z factor
    joints2d = joints2d.view(batch_size, -1, 2)# 2D x,y, img_center as 0,0

    # depth is homogeneous to object scale change in pixels
    est_Z0 = focal * depth + off_z
    assert not torch.isnan(est_Z0).any(), "NaN in est_Z0"
    
    cam_centers = camintr[:, :2, 2].view(batch_size,1,2).repeat(1,num_joints,1)
    # img_centers = (cam_centers.new(input_res) / 2).view(1, 1, 2).repeat(batch_size,num_joints, 1)
    img_centers = (torch.tensor(input_res, dtype=cam_centers.dtype, device=cam_centers.device) / 2).view(1, 1, 2).repeat(batch_size, num_joints, 1)

    est_xy0= joints2d+img_centers
    assert not torch.isnan(est_xy0).any(), "NaN in est_xy0"
    est_XY0=(est_xy0-cam_centers) * est_Z0 / focal
    assert not torch.isnan(est_XY0).any(), "NaN in est_XY0"
    est_c3d = torch.cat([est_XY0, est_Z0], -1)
    return est_xy0,est_Z0, est_c3d

if __name__ == "__main__":
    import pickle
    import numpy as np
    # from cfg import _CONFIG
    from cfg_jointshead import _CONFIG




    print('test forward')
    x = np.random.uniform(0, 255, (4, 3, 224, 224)).astype(np.float32)
    x = Tensor(x)

    print(x.shape)

    # model = timm.create_model("convnext_tiny", pretrained=True)
    # print(model)

    # out = model.forward_features(x)
    # print(out.shape)

    net = HandNet(_CONFIG)
    # net.eval()
    # print(net)

    print("get losses")
    target = {
        'img': torch.rand(4, 3, 128, 128).cuda(),
        'uv': torch.rand(4, 21, 2).cuda(),
        'xyz': torch.rand(4, 21, 3).cuda(),
        'gamma': torch.rand(4, 1).cuda(), 
        'uv_valid': torch.rand(4, 21).cuda(),
        'xyz_valid': torch.rand(4, 1).cuda(),
        'cam_intr': torch.rand(4, 3, 3).cuda(),
        'joints25d': torch.rand(4, 21, 3).cuda(),
    }
    net.cuda()
    x = x.cuda()
    # target = {k: v.cuda() for k, v in target.items()}
    loss = net(x, target)
    print(loss)

    
    

    # path = 'batch_data.pkl'
    # with open(path, 'rb') as f:
    #     batch_data = pickle.load(f)
    #     for k in batch_data:
    #         batch_data[k] = Tensor(batch_data[k]).float()
    #         print(k, batch_data[k].shape, batch_data[k].max(), batch_data[k].min())

    # losses_dict = net(batch_data['img'],batch_data)
    # for key in losses_dict:
    #     print(key, losses_dict[key].item())


    # loss = losses_dict['total_loss']
    # loss.backward()


