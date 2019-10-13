import pybullet as p
import time
import pybullet_data
import random
import gym

import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import math

# Action constants
NULL_ACTION = 0
STEER_LEFT = 1
STEER_RIGHT = 2
ACCELERATE = 3
BRAKE = 4
NUMBER_OF_ACTIONS = 5

# useful constants
MAX_SPEED = 40
MAX_FORCE = 500
MAX_STEER = .6
NUM_BALLS = 1

clamp = lambda x, l, u: l if x < l else u if x > u else x

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
        self.frame = 0 
        self.max_speed = max_speed      
        self.max_steer = max_steer
        self.speed = 1                  # Current speed of car 0 = stopped
        self.steer = 0                  # Current steering angle 0 = straight
        self.maxForce = max_force       # Force to apply to joints
        # debug on screen items
        self.text = None

        # Connect to pybullet
        self.physicsClient = p.connect( p.DIRECT if render_mode == 'rgb_array' else p.GUI )
        p.setGravity(0,0,-9.81)

        # Extra niceties
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

        # Ground
        self.planeId = p.loadURDF("plane.urdf")

        self.balls = []                 # Current balls to display
        self.lostBalls = []             # Balls gone over the edge are stored here
        for _ in range( num_balls ) :
            self.balls.append( p.loadURDF("sphere-car-env.urdf", globalScaling=.5 ) )

        # Vehicle start position
        startPos = [0,0,0] 

        self.vehicle = p.loadURDF("racecar/racecar.urdf", startPos, globalScaling=3.0 ) 

        # Used for reset later
        self.stateId = p.saveState()
        self.reset()

        # Set no force on front wheels, it's a RWD racecar!
        p.setJointMotorControl2(self.vehicle, 5, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=0 ) #self.maxForce )   # RL
        p.setJointMotorControl2(self.vehicle, 7, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=0 ) #self.maxForce )   # RR

        # Useful in debug...
        # numJoints = p.getNumJoints( self.vehicle )
        # self.jointInfo = list( map( lambda i : p.getJointInfo( self.vehicle, i ), range( numJoints ) ) )

        # how many possible actions ?
        self.action_space = gym.spaces.Discrete( NUMBER_OF_ACTIONS )
        # obs shape is pos + orientation + speed of vehicle + each ball position
        state = self.state()
        # self.observation_space = gym.spaces.Box( float("inf"), float("-inf"), shape=[ len(state) ] ) 
        self.observation_space = gym.spaces.Box( 2*math.pi, -2*math.pi, shape=[ len(state) ] ) 
        # self.observation_space = gym.spaces.Box( float("inf"), float("-inf"), shape=[ len(state) ] ) 
        
        # Did we open a window
        self.window = None

        # View matrix - will calculate later
        self.vm = None
        # Total reward this episode
        self.score = 0 


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

    def reset( self ) :
        '''
        public interface
        Start a new episode - new ball position(s) - move car
        to beginning and reinitialize local attributes - score etc.
        '''
        self.score = 0 
        self.speed = 1
        self.steer = 0
        self.frame = 0 
        self.done = False
        self.lostBalls = []
        p.restoreState( self.stateId )
        for ball in self.balls :
            startPos = [ random.randrange(2,6), random.randrange(-4,4), .25 ]
            startOrientation = p.getQuaternionFromEuler([0,0,random.random()] )

            p.resetBasePositionAndOrientation( ball, startPos, startOrientation )
        return self.state()

    def step( self, action ) :
        '''
        public interface
        take an action  and step the physics engine.
        Return the standard openai gym return tuple
        '''
        self.frame = self.frame + 1
        # can be passed in a 1 elem array or a single number
        if hasattr(action, '__iter__') :
            action = action[0]
        
        if ( action==STEER_LEFT ) and self.steer < 1 :
            self.steer = self.steer + .01
        elif ( action==STEER_RIGHT ) and self.steer > -1 :
            self.steer = self.steer - .01
        elif ( action==BRAKE ) and self.speed > 0 :
            self.speed = self.speed - .01
        elif ( action==ACCELERATE ) and self.speed < 1 :
            self.speed = self.speed + .01

        # First do the wheel speed
        p.setJointMotorControl2(self.vehicle, 2, controlMode=p.VELOCITY_CONTROL, targetVelocity=(self.speed*self.max_speed), force=self.maxForce )  # FR
        p.setJointMotorControl2(self.vehicle, 3, controlMode=p.VELOCITY_CONTROL, targetVelocity=(self.speed*self.max_speed), force=self.maxForce )  # FL
        # Then adjust steering position
        p.setJointMotorControl2(self.vehicle, 4, controlMode=p.POSITION_CONTROL, targetPosition=(self.steer*self.max_steer), force=self.maxForce ) # Right wheel steer
        p.setJointMotorControl2(self.vehicle, 6, controlMode=p.POSITION_CONTROL, targetPosition=(self.steer*self.max_steer), force=self.maxForce ) # Left wheel steer
        # Tell engine to do it's magic
        p.stepSimulation()

        done = False
        reward = -0.001

        # Have we fallen off the edge?
        contacts = p.getContactPoints( self.vehicle, self.planeId )
        # vehicle not in contact with anything ... it fell off the edge - we're done
        if len(contacts) == 0 :  
            # sometimes the car wheelies and flies - make sure we're below the plane
            data = p.getBasePositionAndOrientation( self.vehicle )
            vehpos = data[0]
            done = vehpos[2] < 0
            if done :   # if we fell off - that's a negative reward
                reward = reward - 10000
        
        # check for each contact with a ball
        for ball in self.balls :
            contacts = p.getContactPoints( self.vehicle, ball )
            reward = reward + len(contacts)    # add to score if in contact with any ball
            
        for ball in self.balls :
            # has a ball has fallen off the world - only check 1st time 
            if ball not in self.lostBalls :
                contacts = p.getContactPoints( self.planeId, ball )
                if len(contacts) == 0 :
                    data = p.getBasePositionAndOrientation( ball )
                    ballpos = data[0]
                    if ballpos[2] < 0 :
                        self.lostBalls.append( ball )
                        reward = reward + 40
                        # done = True

        obs = self.state() 

        vehdg = obs[0]
        ballhdg = obs[2]
        ideal_steer = ballhdg - vehdg   
        # if abs( ideal_steer ) < 1e-4 :
        steer_error = ideal_steer - self.steer
        reward = reward - clamp( abs(steer_error), 0, .1 )

        # Update the score ...
        self.score = self.score + ( reward if not self.done else 0 )
        info = { "score": self.score, "ideal_steer" : ideal_steer }

        # Return openai.gym format
        rc = ( obs, reward, self.done, info )
        self.done = done
        return rc


    def state( self ) :
        '''
        State is an array of data
 
        '''
        data = p.getBasePositionAndOrientation( self.vehicle )

        vehpos = data[0]
        # state.append( vehpos[0] )  # X
        # state.append( vehpos[1] )  # Y 

        # quat = data[1] 
        # ori = p.getEulerFromQuaternion( quat )

        # print( quat[0], quat[1], quat[2], quat[3], sep="\n" )
        # find [ x  y ] direction vector from quaternion
        qw, qx, qy, qz = data[1] 

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

        state.append( vehhdg )     
        # print( "Hdg:", x, y, z, math.degrees( math.atan2(z,-y) ), math.degrees( ori[2] ) )
        # state.append( self.speed )
        state.append( self.steer )

        for ball in self.balls :
            data = p.getBasePositionAndOrientation( ball )
            ballpos = data[0]
            dx = ballpos[0] - vehpos[0]  
            dy = ballpos[1] - vehpos[1]  
            ang = math.atan2( dy, dx )
            state.append( ang )
            # state.append( math.sqrt(dx*dx + dy*dy) )

        if self.mode == 'human' and (self.frame & 31 == 0) :
            if self.text :
                p.removeAllUserDebugItems()

            txt = "St {:.3f}  He {:.3f}  Be {:.3f} Sc {:.3f}".format( 
                    self.steer * self.max_steer * 57.296 ,
                    vehhdg * 57.296 ,
                    ang * 57.296 ,
                    self.score 
                    ) 
            self.text = p.addUserDebugText( txt, [-5,-5, 1], [0,0,0] )
        return state

# Call this to register the class with openai.gym
def init() :
    gym.envs.registration.register(
        id='VehicleBall-v0',
        entry_point=VehicleBallEnvSpec,
        max_episode_steps=5000,
    )

