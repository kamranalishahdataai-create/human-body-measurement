"""
TensorFlow 2.x compatible wrapper for RunModel
Uses TF1 compatibility mode to run the existing TF1 model code
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from os.path import exists

from .tf_smpl import projection as proj_util
from .tf_smpl.batch_smpl import SMPL
from .models import get_encoder_fn_separate


class RunModel(object):
    def __init__(self, sess=None):
        """
        TensorFlow 2.x compatible model runner
        Args:
          sess: TensorFlow session (optional)
        """
        self.load_path = 'models/model.ckpt-667589'
        
        # Data
        self.batch_size = 1
        self.img_size = 224
        
        self.data_format = 'NHMC'
        self.smpl_model_path = 'models/neutral_smpl_with_cocoplus_reg.pkl'
        
        input_size = (self.batch_size, self.img_size, self.img_size, 3)
        self.images_pl = tf.placeholder(tf.float32, shape=input_size)

        # Model Settings
        self.num_stage = 3
        self.model_type = 'resnet_fc3_dropout'
        self.joint_type = 'cocoplus'
 
        # Camera
        self.num_cam = 3
        self.proj_fn = proj_util.batch_orth_proj_idrot

        self.num_theta = 72        
        # Theta size: camera (3) + pose (24*3) + shape (10)
        self.total_params = self.num_cam + self.num_theta + 10

        self.smpl = SMPL(self.smpl_model_path, joint_type=self.joint_type)

        self.build_test_model_ief()

        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
        else:
            self.sess = sess
        
        # Load data.
        self.saver = tf.train.Saver()
        self.prepare()        

    def build_test_model_ief(self):
        # Load mean value
        self.mean_var = tf.Variable(
            tf.zeros((1, self.total_params)), 
            name="mean_param", 
            dtype=tf.float32
        )

        img_enc_fn, threed_enc_fn = get_encoder_fn_separate(self.model_type)
        
        # Extract image features.        
        self.img_feat, self.E_var = img_enc_fn(
            self.images_pl,
            is_training=False,
            reuse=False
        )
        
        # Start loop
        self.all_verts = []
        self.all_kps = []
        self.all_cams = []
        self.all_Js = []
        self.final_thetas = []
        
        theta_prev = tf.tile(self.mean_var, [self.batch_size, 1])
        
        for i in np.arange(self.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            state = tf.concat([self.img_feat, theta_prev], 1)

            if i == 0:
                delta_theta, _ = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=False)
            else:
                delta_theta, _ = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=True)
                
            # Compute new theta
            theta_here = theta_prev + delta_theta
            
            # cam = N x 3, pose N x 72, shape: N x 10
            cams = theta_here[:, :self.num_cam]
            poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            shapes = theta_here[:, (self.num_cam + self.num_theta):]

            verts, Js, _ = self.smpl(shapes, poses, get_skin=True)
            
            # Project to 2D!
            pred_kp = self.proj_fn(Js, cams, name='proj_2d_stage%d' % i)
            self.all_verts.append(verts)
            self.all_kps.append(pred_kp)
            self.all_cams.append(cams)
            self.all_Js.append(Js)
            self.final_thetas.append(theta_here)
            
            # Finally)update to end iteration.
            theta_prev = theta_here

    def prepare(self):
        print('Restoring checkpoint %s..' % self.load_path)
        self.saver.restore(self.sess, self.load_path)
        self.mean_value = self.sess.run(self.mean_var)

    def predict(self, images, get_theta=False):
        """
        Predict 3D pose and shape from image
        
        Args:
            images: Input image(s)
            get_theta: If True, return theta parameters
            
        Returns:
            joints: 2D joint predictions
            verts: 3D mesh vertices  
            cams: Camera parameters
            joints3d: 3D joint positions
            theta: Shape/pose parameters (if get_theta=True)
        """
        results = self.predict_dict(images)
        
        if get_theta:
            return (
                results['joints'], 
                results['verts'], 
                results['cams'],
                results['joints3d'], 
                results['theta']
            )
        else:
            return (
                results['joints'],
                results['verts'],
                results['cams'],
                results['joints3d']
            )

    def predict_dict(self, images):
        """
        Predict and return results as dictionary
        
        Args:
            images: Input image(s)
            
        Returns:
            dict with keys: joints, verts, cams, joints3d, theta
        """
        feed_dict = {
            self.images_pl: images,
        }
        
        fetch_dict = {
            'joints': self.all_kps[-1],
            'verts': self.all_verts[-1],
            'cams': self.all_cams[-1],
            'joints3d': self.all_Js[-1],
            'theta': self.final_thetas[-1]
        }

        results = self.sess.run(fetch_dict, feed_dict)
        
        # Return numpy arrays
        return results

    def close(self):
        """Close the TensorFlow session"""
        if hasattr(self, 'sess') and self.sess is not None:
            self.sess.close()
            self.sess = None
