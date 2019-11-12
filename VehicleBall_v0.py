import pybullet as p
import time
import pybullet_data
import random
import gym

import numpy as np 
from operator import add

import tkinter
# import cv2
# import PIL.Image, PIL.ImageTk
import math

# Action constants
NULL_ACTION = 0

ACCEL_LEFT = 1
BRAKE_LEFT = 2
NULL_LEFT = 3

ACCEL_RIGHT = 4
BRAKE_RIGHT = 5
NULL_RIGHT = 6

ACCEL_NULL = 7
BRAKE_NULL = 8

NUMBER_OF_ACTIONS = 9

# useful constants
MAX_SPEED = 40
MAX_FORCE = 500
MAX_STEER = .6
NUM_BALLS = 1

clamp = lambda x, l, u: l if x < l else u if x > u else x

class BallData :
    def __init__( self, id, pos=[0,0,0] ) :
        self.id = id
        self.pos = pos 
        self.distance_to_vehicle = None

    def movedTo( self, pos, vehpos ) :
        dx = pos[0]-self.pos[0] 
        dy = pos[1]-self.pos[1] 
        self.pos = pos
        distance_moved = math.sqrt( dx*dx + dy*dy )     
        distance_closed = 0
        
        if vehpos :
            dx = vehpos[0]-self.pos[0] 
            dy = vehpos[1]-self.pos[1] 
            distance_to_vehicle = math.sqrt( dx*dx + dy*dy )
            if self.distance_to_vehicle :
                distance_closed = self.distance_to_vehicle - distance_to_vehicle

            self.distance_to_vehicle = distance_to_vehicle

        return distance_moved, distance_closed

    def reset( self, vehpos ) :
        startOrientation = [ 0,0,0,1 ]
        self.distance_to_vehicle = None 
        self.movedTo( [ random.randrange(2,6), random.randrange(-4,4), .25 ] , vehpos )
        p.resetBasePositionAndOrientation( self.id, self.pos, startOrientation )


class Camera :
    def __init__( self ) :
        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=[0, 0, 1],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 1, 0] )

        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.01,
            farVal=12.0 )

    def photo( self ) :
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=224, 
            height=224,
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix )
        return rgbImg

    def move( self, pos, quat ) :
        rc = p.multiplyTransforms(
            pos,
            quat, 
            [1, 0, .6],
            [0,0,0,1]
        )
        tgt = rc[0]

        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=[ pos[0], pos[1], .6 ],
            cameraTargetPosition=rc[0],
            cameraUpVector=[0, 0, 1])


class VehicleBallEnvSpec( gym.Env ) :
    '''
    This class defines an environment to be used in openai gym.
    We have to define the public interface required by that:
        reset,step,seed, render

    The render is special for pybullet, currently we have to 
    specify mode in the constructor. It's not good, but working on it
    '''
    def __init__( self, render_mode='rgb_array', num_balls=NUM_BALLS, max_speed=MAX_SPEED, max_force=MAX_FORCE, max_steer=MAX_STEER ) :

        self.mode = render_mode
        self.max_speed = max_speed      
        self.max_steer = max_steer
        self.maxForce = max_force       # Force to apply to joints
        # Other instance atts. are set in reset()

        # Connect to pybullet
        self.physicsClient = p.connect( p.DIRECT if render_mode == 'rgb_array' else p.GUI )
        p.setGravity(0,0,-9.81)

        # Extra niceties
        p.setAdditionalSearchPath( pybullet_data.getDataPath() ) 

        # Ground
        self.planeId = p.loadURDF("plane.urdf")

        self.balls = []                 # Current balls to display

        for _ in range( num_balls ) :
            ballId = p.loadURDF("sphere-car-env.urdf", globalScaling=.5 )
            self.balls.append(  BallData( ballId ) )

        self.vehicle = p.loadURDF("racecar/racecar.urdf", globalScaling=3.0 ) 
        # Set no force on front wheels, it's a RWD racecar!
        p.setJointMotorControl2(self.vehicle, 5, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=0 ) #self.maxForce )   # RL
        p.setJointMotorControl2(self.vehicle, 7, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=0 ) #self.maxForce )   # RR

        # Used for reset later
        self.reset()

        # Useful in debug...
        # numJoints = p.getNumJoints( self.vehicle )
        # self.jointInfo = list( map( lambda i : p.getJointInfo( self.vehicle, i ), range( numJoints ) ) )

        # how many possible actions ?
        self.action_space = gym.spaces.Discrete( NUMBER_OF_ACTIONS )
        # obs shape is pos + orientation + speed of vehicle + each ball position
        state = self.state()
        # self.observation_space = gym.spaces.Box( float("inf"), float("-inf"), shape=[ len(state) ] ) 
        self.observation_space = gym.spaces.Box( -1.0, 1.0, shape=[ len(state) ] ) 
        
        # Did we open a window
        self.window = None

        # View matrix - will calculate later
        self.vm = None
        self.camera = Camera()


    def reset( self ) :
        '''
        public interface
        Start a new episode - new ball position(s) - move car
        to beginning and reinitialize local attributes - score etc.
        '''
        self.text = None

        self.score = 0 
        self.speed = 0
        self.steer = 0
        self.frame = 0 
        self.done = False


        # Vehicle start position
        self.vehicle_pos = [0,0,0]
        self.vehicle_orientation = [0,0,0,1]
        p.resetBasePositionAndOrientation( self.vehicle, self.vehicle_pos, self.vehicle_orientation )

        for ball in self.balls :
            ball.reset( self.vehicle_pos )


        return self.state()


    def step( self, action ) :
        '''
        public interface
        take an action  and step the physics engine.
        Return the standard openai gym return tuple
        '''
        self.frame = self.frame + 1
   
        if ( action==ACCEL_LEFT or action==BRAKE_LEFT or action==NULL_LEFT ) and self.steer < 1.0 :
            self.steer = self.steer + .01
        if ( action==ACCEL_RIGHT or action==BRAKE_RIGHT or action==NULL_RIGHT  ) and self.steer > -1.0 :
            self.steer = self.steer - .01
        if ( action==BRAKE_LEFT or action==BRAKE_RIGHT or action==BRAKE_NULL ) and self.speed > -1.0 :
            self.speed = self.speed - .01
        if ( action==ACCEL_LEFT or action==ACCEL_RIGHT or action==ACCEL_NULL ) and self.speed < 1.0 :
            self.speed = self.speed + .01

        # First do the wheel speed
        p.setJointMotorControl2(self.vehicle, 2, controlMode=p.VELOCITY_CONTROL, targetVelocity=(self.speed*self.max_speed), force=self.maxForce )  # FR
        p.setJointMotorControl2(self.vehicle, 3, controlMode=p.VELOCITY_CONTROL, targetVelocity=(self.speed*self.max_speed), force=self.maxForce )  # FL
        # Then adjust steering position
        p.setJointMotorControl2(self.vehicle, 4, controlMode=p.POSITION_CONTROL, targetPosition=(self.steer*self.max_steer), force=self.maxForce ) # Right wheel steer
        p.setJointMotorControl2(self.vehicle, 6, controlMode=p.POSITION_CONTROL, targetPosition=(self.steer*self.max_steer), force=self.maxForce ) # Left wheel steer
   
        # Tell physics engine to do it's magic
        p.stepSimulation()

        data = p.getBasePositionAndOrientation( self.vehicle )
        self.vehicle_pos = data[0]
        self.vehicle_orientation = data[1] 

        # Have we fallen off the edge?
        done = self.vehicle_pos[2] < -0.1
        
        reward = -5 if done else 0  # fallen off - bad reward

        # check for each contact with a ball & vehicle
        for ball in self.balls :

            data = p.getBasePositionAndOrientation( ball.id )
            ballpos = data[0]

            # fell below plane ? ... over the edge
            if ball.pos[2] < 0.1 :
                ball.reset( self.vehicle_pos )
                reward = reward + 10
            else :
                distance_ball_moved , distance_closed = ball.movedTo( ballpos, self.vehicle_pos ) 
                reward = reward + distance_ball_moved * 5 + distance_closed


        obs = self.state() 
        info = self.make_info( obs, reward )
        reward = reward if not self.done else 0

        # Return openai.gym format
        rc = ( obs, reward, self.done, info )

        self.done = self.done | done
        return rc


    def state( self ) :
        '''
        State is an array of data
 
        '''
        
        # state.append( self.vehicle_pos[0] )  # X
        # state.append( self.vehicle_pos[1] )  # Y 

        # quat = data[1] 
        # ori = p.getEulerFromQuaternion( quat )

        # print( quat[0], quat[1], quat[2], quat[3], sep="\n" )
        # find [ x  y ] direction vector from quaternion
        qw, qx, qy, qz = self.vehicle_orientation

        # x = 2 * ( qx*qz + qw*qy )
        # y = 2 * ( qy*qz - qw*qx )
        # z = 1 - 2 * ( qx*qx + qy*qy )

        x = 2 * ( qx*qy - qw*qz )
        y = 1 - 2 * ( qx*qx + qz*qz )
        z = 2 * ( qy*qz + qw*qx )

        # z = 2 * ( qx*qz - qw*qy )
        # y = 2 * ( qy*qx + qw*qz )
        # x = 1 - 2 * ( qz*qz + qy*qy )

        vehhdg = math.atan2(z,-y)   # yaw ( around Z )
        state = []

        state.append( vehhdg / math.pi )   # convert -1 to 1
        # print( "Hdg:", x, y, z, math.degrees( math.atan2(z,-y) ), math.degrees( ori[2] ) )
        # state.append( self.speed )
        state.append( self.steer )
        ang = -math.pi

        for ball in self.balls :
            ballpos = ball.pos
            dx = ballpos[0] - self.vehicle_pos[0]  
            dy = ballpos[1] - self.vehicle_pos[1]  
            ang = math.atan2( dy, dx )

            state.append( ang / math.pi )
            in_front = abs( ang ) < math.pi/2.0
        
            dist = clamp( ball.distance_to_vehicle, 0, 20 ) / 20.0
            state.append( dist if in_front else -dist )

        if self.mode == 'human' and (self.frame & 31 == 0) :
            if self.text :
                p.removeAllUserDebugItems()

            txt = "Sp  {:.3f} St {:.3f}  He {:.3f}  Be {:.3f} Sc {:.3f}".format( 
                    self.speed * self.max_speed ,
                    self.steer * self.max_steer * 57.296 ,
                    vehhdg * 57.296 ,
                    ang * 57.296 ,
                    self.score 
                    ) 
            self.text = p.addUserDebugText( txt, [-5,-5, 1], [0,0,0] )
        return state

    def make_info( self, obs, reward ) :
        '''
        Make the info return from step. This is NOT for
        training, but so clients can see what's happening.
        It relies on the observations being present, so check
        there if observations have 'moved' in the obs array
        '''
        vehdg = obs[0]
        ballhdg = obs[2]

        ideal_steer = ballhdg - vehdg   
        # if abs( ideal_steer ) < 1e-4 :
        steer_error = ideal_steer - self.steer * self.max_steer
        # reward = reward - clamp( abs(steer_error), 0, .1 ) / 10.0

        # Update the score ...
        self.score = self.score + reward 
        return { "score": self.score, "ideal_steer" : ideal_steer, "frame" : self.frame  }


    def __enter__(self) :
        '''
        Used for with Env() as env
        '''
        return self 

    def __exit__(self, exc_type, exc_value, traceback):
        '''
        Used for with Env() as env 
        '''
        self.close()

    def seed( self, s ) :
        '''
        public interface called by contexts - sets RNG seed
        '''
        random.seed( s ) 

    def close( self ) :
        '''
        public interface - clean up
        '''
        if self.window :
            self.window.close()
        p.disconnect() 


    def render( self, mode='human', width=400, height=400 ) :
        '''
        public interface - draw the screen (mode = human )
        or return an RGB image ( mode = rgb_array )
        '''
        if mode == 'rgb_array' :
            if not self.vm :
                self.vm = p.computeViewMatrixFromYawPitchRoll( [0, 0, 0], 17, 50, -40, 0, 2 )
                self.pm = p.computeProjectionMatrixFOV( 80, 1, 0, 1 )

            w,h,img,_,_ = p.getCameraImage( width, height, self.vm, self.pm )

            return img

        if mode == 'human' :
            self.camera.move( self.vehicle_pos, self.vehicle_orientation )
            self.camera.photo()



# Call this to register the class with openai.gym
def init() :
    gym.envs.registration.register(
        id='VehicleBall-v0',
        entry_point=VehicleBallEnvSpec,
        max_episode_steps=5000,
    )

