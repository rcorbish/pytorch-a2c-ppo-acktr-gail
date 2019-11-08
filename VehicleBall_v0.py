import pybullet as p
import time
import pybullet_data
import random
import gym

import numpy as np 

import tkinter
import cv2
import PIL.Image, PIL.ImageTk
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
    def __init__( self, id, pos ) :
        self.id = id
        self.pos = pos 

    def movedTo( self, pos ) :
        dx = pos[0]-self.pos[0] 
        dy = pos[1]-self.pos[1] 
        self.pos = pos
        distance_moved = math.sqrt( dx*dx + dy*dy )        
        return distance_moved


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
        self.prev_distance_to_ball = None

        # Connect to pybullet
        self.physicsClient = p.connect( p.DIRECT if render_mode == 'rgb_array' else p.GUI )
        p.setGravity(0,0,-9.81)

        # Extra niceties
        p.setAdditionalSearchPath( pybullet_data.getDataPath()) #optionally

        # Ground
        self.planeId = p.loadURDF("plane.urdf")

        self.balls = []                 # Current balls to display
        self.lostBalls = []             # Balls gone over the edge are stored here

        for _ in range( num_balls ) :
            ballId = p.loadURDF("sphere-car-env.urdf", globalScaling=.5 )
            position = [ 0,0,0 ]
            self.balls.append(  BallData( ballId, position ) )

        self.vehicle = p.loadURDF("racecar/racecar.urdf", globalScaling=3.0 ) 

        # Used for reset later
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
        self.observation_space = gym.spaces.Box( -1.0, 1.0, shape=[ len(state) ] ) 
        
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
        self.text = None
        self.prev_distance_to_ball = None

        self.score = 0 
        self.speed = 0
        self.steer = 0
        self.frame = 0 
        self.balls.extend( self.lostBalls )
        self.lostBalls = []

        startOrientation = [ 0,0,0,1 ]

        for ball in self.balls :
            ball.movedTo( [ random.randrange(2,6), random.randrange(-4,4), .25 ] )
            p.resetBasePositionAndOrientation( ball.id, ball.pos, startOrientation )

        # Vehicle start position
        startPos = [ 0,0,0 ]
        p.resetBasePositionAndOrientation( self.vehicle, startPos, startOrientation )

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
        vehpos = data[0]


        done = False
        # Have we fallen off the edge?
        contacts = p.getContactPoints( self.vehicle, self.planeId )
        # vehicle not in contact with anything ... it fell off the edge - we're done
        if len(contacts) == 0 :  
            # sometimes the car wheelies and flies - 
            # ignore 'done' if we're not below the plane
            done = vehpos[2] < 0
        
        reward = 0
        # check for each contact with a ball & vehicle
        for ball in self.balls :

            data = p.getBasePositionAndOrientation( ball.id )
            ballpos = data[0]

            dx = vehpos[0]-ballpos[0] 
            dy = vehpos[1]-ballpos[1] 
            distance_to_ball = math.sqrt( dx*dx + dy*dy )

            if self.prev_distance_to_ball :
                distance_closed = self.prev_distance_to_ball - distance_to_ball 

                # reward for being closer to ball than prev. step
                if distance_closed > 0 :
                    reward = reward + distance_closed

            self.prev_distance_to_ball = distance_to_ball

            # fell below plane ? ... over the edge
            if ball.pos[2] < 0.1 :
                self.lostBalls.append( ball )

            distance_moved = ball.movedTo( ballpos ) 
            reward = reward + distance_moved * 5

        # remove lost balls from the self.balls list
        self.balls = [ elem for elem in self.balls if elem not in self.lostBalls ]

        obs = self.state() 

        vehdg = obs[0]

        ballhdg = obs[2]

        ideal_steer = ballhdg - vehdg   
        # if abs( ideal_steer ) < 1e-4 :
        steer_error = ideal_steer - self.steer * self.max_steer
        # reward = reward - clamp( abs(steer_error), 0, .1 ) / 10.0

        # Update the score ...
        self.score = self.score + reward 
        info = { "score": self.score, "ideal_steer" : ideal_steer, "frame" : self.frame  }

        # Return openai.gym format
        rc = ( obs, reward, done, info )
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

        state.append( vehhdg / math.pi )   # convert -1 to 1
        # print( "Hdg:", x, y, z, math.degrees( math.atan2(z,-y) ), math.degrees( ori[2] ) )
        # state.append( self.speed )
        state.append( self.steer )
        ang = -math.pi

        for ball in self.balls :
            ballpos = ball.pos
            dx = ballpos[0] - vehpos[0]  
            dy = ballpos[1] - vehpos[1]  
            ang = math.atan2( dy, dx )
            state.append( ang / math.pi )
            state.append( clamp( math.sqrt(dx*dx + dy*dy), 0, 10 ) / 10.0 )

        for ball in self.lostBalls :
            state.append( 0 )
            state.append( 1 )

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


# Call this to register the class with openai.gym
def init() :
    gym.envs.registration.register(
        id='VehicleBall-v0',
        entry_point=VehicleBallEnvSpec,
        max_episode_steps=5000,
    )

