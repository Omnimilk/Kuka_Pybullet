import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import pybullet as p
import pybullet_data
import numpy as np
from numpy.random import normal,uniform
import random
import copy
import math
import time
import cv2
import matplotlib.pyplot as plt
from pprint import pprint
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
from pybullet_envs.bullet.kuka import Kuka
from sim_env_reconfigured import KukaCamGymEnv_Reconfigured
from utils import commands_iterator,get_data_paths
#gym envrionment information: 
#   plane base [0,0,-1]
#   tray base: [0.640000,0.075000,-0.190000], tray orientation: [0.000000,0.000000,1.000000,0.000000]
#   table base:  [0.5000000,0.00000,-.820000], table orientation: [0.000000,0.000000,0.0,1.0]
#   kuka base： [-0.100000,0.000000,0.070000], kuka orientation： [0.000000,0.000000,0.000000,1.000000]
TRAY_RESCALE_FACTOR = 1.8



def add_one_obj_to_scene(num,pos=None, orientation = None, global_scaling=1):
    """
    Inputs:
        num: int, serial number of the object
        pos: 3D vector, pose of the object
        orientation: quaternion, orientation of the object
    Output:
        obj: pybullet object
    """
    assert num<1000, "random object number cannot exceed 1000!"
    assert pos is not None, "pos cannot be empty!"
    if orientation is None:
        obj = p.loadURDF("random_urdfs/{0:0>3}/{0:0>3}.urdf".format(num,num),pos,globalScaling=global_scaling)
    else:
        obj = p.loadURDF("random_urdfs/{0:0>3}/{0:0>3}.urdf".format(num,num),pos,orientation,globalScaling=global_scaling)
    return obj

def add_objs_to_scene(nums,poses,orientations=None):
    """
    Inputs:
        nums: list of ints, serial numbers of the objects
        poses: list of 3D vector, a list of poses of the objects
        orientations: list of quaternions, optional, a list of quaternions of the objects
    Outputs:
        objs: a list of pybullet objects
    """
    assert len(nums)==len(poses), "number of objects should match number of poses!"
    if orientations is not None:
        assert len(orientations)==len(nums), "size of orientations should match size of objects!"
        return [add_one_obj_to_scene(num,pos,orientation) for num,pos,orientation in zip(nums,poses,orientations)]
    else:
        return [add_one_obj_to_scene(num,pos) for num,pos in zip(nums,poses)]
        
def add_random_objs_to_scene(size,pos_mean=[0,0],pos_height=1,orientation_mean=[0,0,0,1],use_uniform=True):
    if use_uniform:
        nums = [random.randint(0,999) for _ in range(size)]
        tray_width = 0.145*TRAY_RESCALE_FACTOR
        tray_height = 0.2*TRAY_RESCALE_FACTOR
        poses = [[uniform(0.64-0.055 - tray_width,0.64-0.055 +tray_width),uniform(0.075-tray_height,0.075 +tray_height),uniform(-0.189,-0.185)] for _ in range(size)]#container center position (0.64,0.075,-0.19)
        # poses = [[uniform]]

        #randomize orientations
        # orientations = [normal(size=(4)) for _ in range(size)]
        # objs = add_objs_to_scene(nums,poses,orientations)

        #without randomize orientations
        objs = add_objs_to_scene(nums,poses)
    else:
        nums = [random.randint(0,999) for _ in range(size)]
        poses = [[normal(0.64,0.14),normal(scale=0.14),-0.182] for _ in range(size)]#container center position (0.6,0,0)
        # poses = [[uniform]]
        orientations = [normal(size=(4)) for _ in range(size)]
        objs = add_objs_to_scene(nums,poses,orientations)
    # objs = add_objs_to_scene(nums,poses)
    return objs

def write_from_imgarr(imgarr,serial_number,path="sim_images/{0:0>6}.jpeg"):
    bgra =imgarr[2]#imgarr[3] depth image; imgarr[4] segmentation mask
    img = np.reshape(bgra, (512, 640, 4)).astype(np.uint8)#BGRA
    img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)#RGB
    cv2.imwrite(path.format(serial_number),img)#write original image
    # print(len(imgarr[4]))
    segmentation_mask = imgarr[4]
    segmentation_mask = np.reshape(segmentation_mask,(512,640,1)).astype(np.uint8)
    # plt.imshow(segmentation_mask[:,:,0])
    # plt.show()
    segmentation_mask_path = path.format(serial_number)[:-5] + "_segmentation.jpeg"
    # print("segmentation path: {}, shape of seg mask {}".format(segmentation_mask_path,segmentation_mask.shape))
    cv2.imwrite(segmentation_mask_path,segmentation_mask)

def write_from_npimg(npimg,serial_number,path="sim_images/{0:0>6}.jpeg"):
    img = np.reshape(npimg, (512, 640, 4)).astype(np.uint8)#BGRA
    img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)#RGB
    cv2.imwrite(path.format(serial_number),img)

def get_randomized_ViewMat(sigma=None):
    if sigma is not None:
        mean = [0,0,0]
        cov = sigma*np.eye(3)
        camEyePos = np.array([0.42,0.2,0.54]) + np.random.multivariate_normal(mean,cov)
        targetPos = np.array([0.55,0.2,-0.180000]) + np.random.multivariate_normal(mean,cov)
        # camEyePos = np.array([0.03,0.236,0.54]) + np.random.multivariate_normal(mean,cov)
        # # targetPos = np.array([0.640000,0.075000,-0.190000]) + np.random.multivariate_normal(mean,cov)
        # targetPos = np.array([0.640000,0.0000,-0.190000]) + np.random.multivariate_normal(mean,cov)
    else:#fixed camera angle to get naive dataset
        camEyePos = np.array([0.42,0.2,0.54])
        targetPos = np.array([0.55,0.2,-0.180000])
        # camEyePos = np.array([0.45,0.25,0.54])
        # targetPos = np.array([0.54,0.25,-0.180000])
        # camEyePos = np.array([0.03,0.236,0.54])
        # targetPos = np.array([0.640000,0.0000,-0.190000])
    cameraUp = np.array([0,0,1])
    viewMat = p.computeViewMatrix(camEyePos,targetPos,cameraUp)
    return viewMat

def main_usingEnvOnly():
    environment = KukaGymEnv(renders=True,isDiscrete=False, maxSteps = 10000000)
    #p.resetBasePositionAndOrientation(self.kukaUid,[-0.100000,0.000000,0.070000],[0.000000,0.000000,0.000000,1.000000])
    randomObjs = add_random_objs_to_scene(10)	  
    motorsIds=[]
    motorsIds.append(environment._p.addUserDebugParameter("posX",0.4,0.75,0.537))
    motorsIds.append(environment._p.addUserDebugParameter("posY",-.22,.3,0.0))
    motorsIds.append(environment._p.addUserDebugParameter("posZ",0.1,1,0.2))
    motorsIds.append(environment._p.addUserDebugParameter("yaw",-3.14,3.14,0))
    motorsIds.append(environment._p.addUserDebugParameter("fingerAngle",0,0.3,.3))

    # dv = 0.01 
    # motorsIds.append(environment._p.addUserDebugParameter("posX",-dv,dv,0))
    # motorsIds.append(environment._p.addUserDebugParameter("posY",-dv,dv,0))
    # motorsIds.append(environment._p.addUserDebugParameter("posZ",-dv,dv,0))
    # motorsIds.append(environment._p.addUserDebugParameter("yaw",-dv,dv,0))
    # motorsIds.append(environment._p.addUserDebugParameter("fingerAngle",0,0.3,.3))

    done = False
    #According to the hand-eye coordination paper, the camera is mounted over the shoulder of the arm
    camEyePos = [0.03,0.236,0.54]
    targetPos = [0.640000,0.075000,-0.190000]
    cameraUp = [0,0,1]
    viewMat = p.computeViewMatrix(camEyePos,targetPos,cameraUp)
    camInfo = p.getDebugVisualizerCamera()
    projMatrix = camInfo[3]
    # viewMat = [-0.5120397806167603, 0.7171027660369873, -0.47284144163131714, 0.0, -0.8589617609977722, -0.42747554183006287, 0.28186774253845215, 0.0, 0.0, 0.5504802465438843, 0.8348482847213745, 0.0, 0.1925382763147354, -0.24935829639434814, -0.4401884973049164, 1.0]
    # projMatrix = [0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
    img_arr = p.getCameraImage(640,512,viewMatrix=viewMat,projectionMatrix=projMatrix)#640*512*3 
    # write_from_imgarr(img_arr, 1)
    
    while (not done):   
        action=[]
        for motorId in motorsIds:
            action.append(environment._p.readUserDebugParameter(motorId))        
        state, reward, done, info = environment.step2(action)
        obs = environment.getExtendedObservation()

    # physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    # p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    # p.setGravity(0,0,-10)
    # planeId = p.loadURDF("plane.urdf")
    # objects = p.loadSDF(os.path.join(pybullet_data.getDataPath(),"kuka_iiwa/kuka_with_gripper2.sdf"))
    
    # cubeStartPos = [0,0,1]
    # cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
    # # randomObj = add_one_obj_to_scene(80, [2,2,0],cubeStartOrientation)
    # randomObjs = add_random_objs_to_scene(10)

    # boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
    # for i in range (10000):
    #     p.stepSimulation()
    #     time.sleep(1./240.)
    # cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    # print(cubePos,cubeOrn)
    # p.disconnect()

def main():
    environment = KukaCamGymEnv(renders=True,isDiscrete=False)  
    # environment = KukaCamGymEnv_Reconfigured(renders=True,isDiscrete=False)  
    # environment._reset()
    # num_of_objects = 50
    # num_of_objects_var = 10
    num_of_objects = 20
    num_of_objects_var = 4
    randomObjs = add_random_objs_to_scene(num_of_objects)
    motorsIds = []
    #addUserDebugParameter(paramName,rangeMin,rangeMax,startValue)
    #return the most up-to-date reading of the parameter

    #TODO: add visual randomization(texture,color,lighting,brightness) and dynamics randomization(mass, friction)
    #usefule methods: setDebugObjectColor, changeDynamics

    #motorsIds.append(environment._p.addUserDebugParameter("posX",0.4,0.75,0.537))
	#motorsIds.append(environment._p.addUserDebugParameter("posY",-.22,.3,0.0))
	#motorsIds.append(environment._p.addUserDebugParameter("posZ",0.1,1,0.2))
	#motorsIds.append(environment._p.addUserDebugParameter("yaw",-3.14,3.14,0))
	#motorsIds.append(environment._p.addUserDebugParameter("fingerAngle",0,0.3,.3))
    # dv = 1 
    # motorsIds.append(environment._p.addUserDebugParameter("posX",-dv,dv,0))#(paramName, rangeMin, rangeMax, startValue)
    # motorsIds.append(environment._p.addUserDebugParameter("posY",-dv,dv,0))
    # motorsIds.append(environment._p.addUserDebugParameter("posZ",-dv,dv,0))
    # motorsIds.append(environment._p.addUserDebugParameter("yaw",-dv,dv,0))
    # motorsIds.append(environment._p.addUserDebugParameter("fingerAngle",0,0.3,.3))	
    
    done = False

    try:
        with open("sim_images/serial_num_log.txt",'r') as f:
            img_serial_num = int(f.read())
    except:
        print("read serial number failed!")
        img_serial_num = 0
    step = 0
    snapshot_interval = 1#50#42#20 before, 42 would make about 10 snapshot per try, like in the real world dataset
    # viewMat = [[ 775.71899414,    0.,           0.        ],
    #            [   0.,          775.71899414,    0.        ],
    #            [ 335.3380127,   232.45100403,    1.        ]]
    viewMat = get_randomized_ViewMat()#sigma = 0.001
    camInfo = p.getDebugVisualizerCamera()
    # viewMat = camInfo[2]
    projMatrix = camInfo[3]
    # projMatrix = [  [-0.0210269,  -0.99247903,  0.120598,    0.26878399],
    #                 [-0.88814598, -0.0368459,  -0.45808199,  0.293412  ],
    #                 [ 0.45908001, -0.116741,   -0.88069099,  0.71057898],
    #                 [ 0.,          0.,          0.,          1.        ]]
    action_keys = ["grasp/0/commanded_pose/transforms/base_T_endeffector/vec_quat_7", "grasp/1/commanded_pose/transforms/base_T_endeffector/vec_quat_7"]
    data_folder = "/Users/bozai/Desktop/PixelDA/PixelDA/Data/tfdata"
    file_tail = "22"
    data_path = get_data_paths(data_folder,file_tail)
    commands = commands_iterator(data_path)
    while (not done):    
        # action=[]#5-D control signal
        # for motorId in motorsIds:
        #     action.append(environment._p.readUserDebugParameter(motorId))
        # if step%snapshot_interval==0:
        #     #get cameta image
        #     print("Saving image... Current image count: {}".format(img_serial_num))
        #     img_arr = p.getCameraImage(640,512,viewMatrix=viewMat,projectionMatrix=projMatrix)#640*512*3 
        #     write_from_imgarr(img_arr, img_serial_num)
        #     # np_img = environment.getExtendedObservation()#shape is not correct
        #     # write_from_npimg(np_img,img_serial_num)
        #     img_serial_num +=1
        #     # environment._kuka.
        # step +=1
        # state, reward, done, info = environment.step(action)
        #obs = environment.getExtendedObservation()
        # print("done: {}, step: {}".format(done,step))
        print("helloooooo")
        attemption = commands.__next__()
        print(attemption)
        for action_with_quaternion in attemption:
            print("action_with_quaternion: {}".format(action_with_quaternion))
            quaternion = action_with_quaternion[0][3:]
            print("Quaternion {}".format(quaternion))
            euler = p.getEulerFromQuaternion(quaternion)#[yaw,pitch,roll]
            action = action_with_quaternion[0][:3].tolist()
            action.append(euler[0])
            action.append(euler[2])
            print("action: {}".format(action))
            if step%snapshot_interval==0:
                #get cameta image
                print("Saving image... Current image count: {}".format(img_serial_num))
                img_arr = p.getCameraImage(640,512,viewMatrix=viewMat,projectionMatrix=projMatrix)#640*512*3 
                write_from_imgarr(img_arr, img_serial_num)
                # np_img = environment.getExtendedObservation()#shape is not correct
                # write_from_npimg(np_img,img_serial_num)
                img_serial_num +=1       
            step +=1
            print("here")
            state, reward, done, info = environment.step(action)
            print("action executed")

        if done:
            environment._reset()
            randomObjs = add_random_objs_to_scene(num_of_objects+random.choice(range(-num_of_objects_var,num_of_objects_var+1)))
            viewMat = get_randomized_ViewMat(sigma = 0.0001)#change view per try, not per image，0.003                                            
            done =False
            print("Environment reseted!")
            with open("sim_images/serial_num_log.txt","w") as f:
                f.write(str(img_serial_num))
if __name__ == '__main__':
    main()
    #main_usingEnvOnly()
    