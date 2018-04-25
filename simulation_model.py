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
from sim_env_reconfigured import KukaCamGymEnv_Reconfigured,Kuka_Reconfigured,TRAY_RESCALE_FACTOR
from utils import commands_iterator,get_data_paths
from extract_foreground import concat_simfore_realback
#gym envrionment information: 
#   plane base [0,0,-1]
#   tray base: [0.640000,0.075000,-0.190000], tray orientation: [0.000000,0.000000,1.000000,0.000000]
#   table base:  [0.5000000,0.00000,-.820000], table orientation: [0.000000,0.000000,0.0,1.0]
#   kuka base： [-0.100000,0.000000,0.070000], kuka orientation： [0.000000,0.000000,0.000000,1.000000]




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
        obj_Uid = p.loadURDF("random_urdfs/{0:0>3}/{0:0>3}.urdf".format(num),pos,globalScaling=global_scaling)
    else:
        obj_Uid = p.loadURDF("random_urdfs/{0:0>3}/{0:0>3}.urdf".format(num),pos,orientation,globalScaling=global_scaling)
    return obj_Uid

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
        poses = [[normal(0.64,0.14),normal(scale=0.14),-0.182] for _ in range(size)]#container center position (0.64,0.075,-0.19)
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

def substitute_from_imgarr(imgarr,real_image):
    bgra =imgarr[2]#imgarr[3] depth image; imgarr[4] segmentation mask
    img = np.reshape(bgra, (512, 640, 4)).astype(np.uint8)#BGRA
    img = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)#RGB
    segmentation_mask = imgarr[4]
    segmentation_mask = np.reshape(segmentation_mask,(512,640)).astype(np.uint8)
    segmentation_mask = np.dstack((segmentation_mask,segmentation_mask,segmentation_mask))
    substitued = concat_simfore_realback(img, segmentation_mask, real_image)
    return substitued

def get_randomized_ViewMat(sigma=None):
    if sigma is not None:
        mean = [0,0,0]
        cov = sigma*np.eye(3)
        camEyePos = np.array([0.42,0.2,0.8]) + np.random.multivariate_normal(mean,cov)
        targetPos = np.array([0.55,0.2,-0.180000]) + np.random.multivariate_normal(mean,cov)
        # camEyePos = np.array([0.03,0.236,0.54]) + np.random.multivariate_normal(mean,cov)
        # # targetPos = np.array([0.640000,0.075000,-0.190000]) + np.random.multivariate_normal(mean,cov)
        # targetPos = np.array([0.640000,0.0000,-0.190000]) + np.random.multivariate_normal(mean,cov)
    else:#fixed camera angle to get naive dataset
        eye_x = 0.21#0.16
        eye_y = 0.24
        eye_z = 0.64

        target_x = 0.52
        target_y = 0.19
        target_z = 0
        camEyePos = np.array([eye_x,eye_y,eye_z]) #arm base [-0.1,-0.075,0.070000], tray base (0.640000,0.075000,-0.190000)
        targetPos = np.array([target_x,target_y,target_z])
        # camEyePos = np.array([0.24,0.2,0.54]) #arm base [-0.1,-0.075,0.070000], tray base (0.640000,0.075000,-0.190000)
        # targetPos = np.array([0.64,0.2,-0.190000])
        # camEyePos = np.array([0.42,0.2,0.54])
        # targetPos = np.array([0.55,0.2,-0.180000])
        # camEyePos = np.array([0.45,0.25,0.54])
        # targetPos = np.array([0.54,0.25,-0.180000])
        # camEyePos = np.array([0.03,0.236,0.54])
        # targetPos = np.array([0.640000,0.0000,-0.190000])
    #cameraUp = np.array([ (target_x-eye_x)/(eye_z-target_z), (target_y - eye_y)/ (eye_z - target_z) ,1])
    cameraUp = np.array([0,0,1])
    viewMat = p.computeViewMatrix(camEyePos,targetPos,cameraUp)
    return viewMat

def get_ProjMat():
    projMat = p.computeProjectionMatrixFOV(fov=46.5, aspect=1.25, nearVal=0.001, farVal=100)
    return projMat
        

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

def main():
    # environment = KukaCamGymEnv(renders=True,isDiscrete=False)  
    environment = KukaCamGymEnv_Reconfigured(renders=True,isDiscrete=False)  
    environment._reset()
    # num_of_objects = 50
    # num_of_objects_var = 10
    num_of_objects = 20
    num_of_objects_var = 4
    randomObjs = add_random_objs_to_scene(num_of_objects)
    done = False

    try:
        with open("sim_images/serial_num_log.txt",'r') as f:
            img_serial_num = int(f.read())
    except:
        print("read serial number failed!")
        img_serial_num = 0
    step = 0
    snapshot_interval = 50#42#20 before, 42 would make about 10 snapshot per try, like in the real world dataset
    viewMat = get_randomized_ViewMat()#sigma = 0.001
    camInfo = p.getDebugVisualizerCamera()
    # viewMat = camInfo[2]
    projMatrix = camInfo[3]
    action_keys = ["grasp/0/commanded_pose/transforms/base_T_endeffector/vec_quat_7", "grasp/1/commanded_pose/transforms/base_T_endeffector/vec_quat_7"]
    data_folder = "/Users/bozai/Desktop/PixelDA/PixelDA/Data/tfdata"
    file_tail = "22"
    data_path = get_data_paths(data_folder,file_tail)
    commands = commands_iterator(data_path)
    while (not done):    
        attemption = commands.__next__()
        for action_with_quaternion in attemption:
            quaternion = action_with_quaternion[0][3:]
            euler = p.getEulerFromQuaternion(quaternion)#[yaw,pitch,roll]
            action = action_with_quaternion[0][:3].tolist()
            action.append(euler[0])
            action.append(euler[2])
            if step%snapshot_interval==0:
                #get cameta image
                print("Saving image... Current image count: {}".format(img_serial_num))
                img_arr = p.getCameraImage(640,512,viewMatrix=viewMat,projectionMatrix=projMatrix)#640*512*3 
                write_from_imgarr(img_arr, img_serial_num)
                img_serial_num +=1       
            step +=1
            state, reward, done, info = environment.step(action)#state: (256, 341, 4), info: empty dict
            print("step: {} done: {} reward: {}".format(step, done, reward))
        if done:
            environment._reset()
            randomObjs = add_random_objs_to_scene(num_of_objects+random.choice(range(-num_of_objects_var,num_of_objects_var+1)))
            viewMat = get_randomized_ViewMat(sigma = 0.0001)#change view per try, not per image，0.003                                            
            done =False
            print("Environment reseted!")
            with open("sim_images/serial_num_log.txt","w") as f:
                f.write(str(img_serial_num))


def reset_sim_env(time_step = 1./240.):
    # p.disconnect()
    urdfRoot=pybullet_data.getDataPath()
    p.connect(p.DIRECT)#DIRECT
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(time_step)
    p.loadURDF("plane.urdf",[0,0,-1])

    p.setGravity(0,0,-10)
    # kuka_arm = Kuka(urdfRootPath=urdfRoot, timeStep=time_step)
    kuka_arm = Kuka_Reconfigured(urdfRootPath=urdfRoot, timeStep=time_step)
    kuka_arm.useSimulation = 0
    return kuka_arm

def setting_simulation_env():
    time_step = 1./240.
    num_of_objects = 20
    num_of_objects_var = 4
    # p.connect(p.GUI)
    reset_sim_env()
   
    viewMat = get_randomized_ViewMat()#sigma = 0.001
    # projMatrix = [  [-0.0210269,  -0.99247903,  0.120598,    0.26878399],
    #             [-0.88814598, -0.0368459,  -0.45808199,  0.293412  ],
    #             [ 0.45908001, -0.116741,   -0.88069099,  0.71057898],
    #             [ 0.,          0.,          0.,          1.        ]]
    # camInfo = p.getDebugVisualizerCamera()# viewMat = camInfo[2]
    # projMatrix = camInfo[3]
    projMatrix = get_ProjMat()
    # action_keys = ["grasp/0/commanded_pose/transforms/base_T_endeffector/vec_quat_7", "grasp/1/commanded_pose/transforms/base_T_endeffector/vec_quat_7"]
    data_folder = "/Users/bozai/Desktop/PixelDA/PixelDA/Data/tfdata"
    file_tail = "22"
    data_path = get_data_paths(data_folder,file_tail)
    commands = commands_iterator(data_path)
    # try:
    #     with open("sim_images/serial_num_log.txt",'r') as f:
    #         img_serial_num = int(f.read())
    # except:
    #     print("read serial number failed!")
    #     img_serial_num = 0
    img_serial_num = 0
    kukaEndEffectorIndex = 6
    #joint damping coefficents
    jd=[0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001]
    while True:
        kuka_arm = reset_sim_env()
        kukaUid = kuka_arm.kukaUid
        # numJoints = kuka_arm.numJoints
        
        attemption, image = commands.__next__()
        randomObjs = add_random_objs_to_scene(num_of_objects+random.choice(range(-num_of_objects_var,num_of_objects_var+1)))
        for action_with_quaternion in attemption:
            quaternion = action_with_quaternion[0][3:]
            pos = action_with_quaternion[0][:3]
            jointPoses = p.calculateInverseKinematics(kukaUid,kukaEndEffectorIndex,pos,quaternion,jointDamping=jd)
            # print("numJoints : {}, jointPoses: {}".format(numJoints,len(jointPoses)))
            for i in range (12):
                p.resetJointState(kukaUid,i,jointPoses[i])    
            # time.sleep(30)
            # quaternion = action_with_quaternion[0][3:]
            # euler = p.getEulerFromQuaternion(quaternion)#[yaw,pitch,roll]
            # action = action_with_quaternion[0][:3].tolist()
            # action.append(euler[0])
            # action.append(euler[2])
            # kuka_arm.applyAction(action)#bug in the module implementation
            print("Saving image... Current image count: {}".format(img_serial_num))
            img_arr = p.getCameraImage(640,512,viewMatrix=viewMat,projectionMatrix=projMatrix,lightDirection=[1,1,1])#640*512*3 
            # write_from_imgarr(img_arr, img_serial_num)
            subed = substitute_from_imgarr(img_arr,image)
            subed = cv2.cvtColor(subed, cv2.COLOR_RGB2BGR)
            cv2.imwrite("sim_backSubed/{0:0>6}_subed.jpeg".format(img_serial_num),subed)
            img_serial_num +=1
        if img_serial_num>10000:#40000
            with open("sim_images/serial_num_log.txt","w") as f:
                f.write(str(img_serial_num))
            break     
            
            # randomObjs = add_random_objs_to_scene(num_of_objects+random.choice(range(-num_of_objects_var,num_of_objects_var+1)))
            # viewMat = get_randomized_ViewMat(sigma = 0.0001)#change view per try, not per image，0.003                                            
            # done =False
            # print("Environment reseted!")
            # with open("sim_images/serial_num_log.txt","w") as f:
            #     f.write(str(img_serial_num))

if __name__ == '__main__':
    # main()
    #main_usingEnvOnly()
    setting_simulation_env()
    