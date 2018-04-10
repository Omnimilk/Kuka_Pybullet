import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import pybullet as p
import pybullet_data
import  numpy as np
import random
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
from pybullet_envs.bullet.kuka import Kuka
#gym envrionment information: 
#   plane base [0,0,-1]
#   tray base: [0.640000,0.075000,-0.190000], tray orientation: [0.000000,0.000000,1.000000,0.000000]
#   table base:  [0.5000000,0.00000,-.820000], table orientation: [0.000000,0.000000,0.0,1.0]
#   kuka base： [-0.100000,0.000000,0.070000], kuka orientation： [0.000000,0.000000,0.000000,1.000000]
#   tray and arm are off-proportional, use globalScaling argument when loading tray to get similar images

class Kuka_Reconfigured(Kuka):
    def reset(self):
        objects = p.loadSDF(os.path.join(self.urdfRootPath,"kuka_iiwa/kuka_with_gripper2.sdf"))
        self.kukaUid = objects[0]
        #Arm base reconfigured
        p.resetBasePositionAndOrientation(self.kukaUid,[-0.1,-0.075,0.070000],[0.000000,0.000000,0.000000,1.000000])
        # p.resetBasePositionAndOrientation(self.kukaUid,[-0.100000,0.000000,0.070000],[0.000000,0.000000,0.000000,1.000000])
        self.jointPositions=[ 0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048, -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200 ]
        self.numJoints = p.getNumJoints(self.kukaUid)
        for jointIndex in range (self.numJoints):
            p.resetJointState(self.kukaUid,jointIndex,self.jointPositions[jointIndex])
            p.setJointMotorControl2(self.kukaUid,jointIndex,p.POSITION_CONTROL,targetPosition=self.jointPositions[jointIndex],force=self.maxForce)
        #rescale the tray
        self.trayUid = p.loadURDF(os.path.join(self.urdfRootPath,"tray/tray.urdf"), basePosition = (0.640000,0.075000,-0.190000), baseOrientation = (0.000000,0.000000,1.000000,0.00000), globalScaling = 1.4)
        self.endEffectorPos = [0.537,0.0,0.5]
        self.endEffectorAngle = 0

        self.motorNames = []
        self.motorIndices = []
        
        for i in range (self.numJoints):
            jointInfo = p.getJointInfo(self.kukaUid,i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)

class KukaCamGymEnv_Reconfigured(KukaCamGymEnv):
    def _reset(self):
        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])

        p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.820000,0.000000,0.000000,0.0,1.0)

        xpos = 0.5 +0.2*random.random()
        ypos = 0 +0.25*random.random()
        ang = 3.1415925438*random.random()
        orn = p.getQuaternionFromEuler([0,0,ang])
        self.blockUid =p.loadURDF(os.path.join(self._urdfRoot,"block.urdf"), xpos,ypos,-0.1,orn[0],orn[1],orn[2],orn[3])

        p.setGravity(0,0,-10)
        #Kuka arm class altered
        self._kuka = Kuka_Reconfigured(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        #self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)