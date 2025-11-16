import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation 

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class DSLPIDControl(BaseControl):
    """PID control class for Crazyflies.

    Contributors: SiQi Zhou, James Xu, Tracy Du, Mario Vukosavljev, Calvin Ngan, and Jingyuan Hou.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()

        # You can initialize more parameters here



        # Your code ends here

        ######################################################
        # Do not change these parameters below
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
    
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3),
                       target_acc = np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_dslPIDPositionControl()` and `_dslPIDAttitudeControl()`.
        Parameter `cur_ang_vel` is unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        mass = self._getURDFParameter('m')
        target_thrust, computed_target_rpy, pos_e, cur_rotation = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel,
                                                                         target_acc,
                                                                         mass=mass
                                                                         )
        scalar_thrust = max(0., np.dot(target_thrust, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        rpm = self._dslPIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          cur_ang_vel,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )

        return rpm, pos_e, computed_target_rpy
    
    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel,
                               target_acc,
                               mass = 0.29
                               ):
        """DSL's CF2.x PID position control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray
            (3,1)-shaped array of floats containing the desired velocity.
        target_acc : ndarray
            (3,1)-shaped array of floats containing the desired acceleration.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.
        ndarray
            (3,3)-shaped array of floats representing the current rotation matrix (from quaternion).
        """
        target_thrust = np.zeros(3)
        target_rpy = np.zeros(3)
        pos_e_vec = np.zeros(3)
        cur_rotation = np.zeros((3,3))

        ########################################################################
        # ---------- DSL PID Position Control Implementation ----------
        g = 9.81  # gravity constant (m/s^2)

        # ---------- Convert quaternion → rotation matrix ----------        
        rot = Rotation.from_quat(cur_quat)
        cur_rotation = rot.as_matrix()

        # ---------- Step 2: Compute position and velocity errors ----------
        pos_e_vec = target_pos - cur_pos
        vel_e_vec = target_vel - cur_vel

        # ---------- Step 3: PD + feedforward + gravity compensation ----------
        # Kp = np.array([1,1,10.0])
        # Kd = np.array([1,1,8])

        # Kp = np.array([1,1,10.0])
        # Kd = np.array([1,1,8])
         # Kp = np.array([0.4,0.4,5])
        # Kd = np.array([0.4,0.4,8])
        
        Kp = np.array([0.15,0.15,8.0])
        Kd = np.array([0.15,0.15,4.5])

        # ------------- calculating the upward thrust --------------------------
        F_g = np.array([0.0, 0.0, mass * g])
        F_d = Kp*pos_e_vec + Kd*vel_e_vec + mass * target_acc + F_g  # desired total thrust (world frame)

        # -------------Desired body z-axis (thrust direction)------------------
        if np.linalg.norm(F_d) > 1e-6:
            zb_desired = F_d / np.linalg.norm(F_d)
        else:
            zb_desired = np.array([0.,0.,1.])

        # -------------Calculating the Desired yaw -----------------------------
        yaw_desired = target_rpy[2]
        xc_desired = np.array([np.cos(yaw_desired), np.sin(yaw_desired), 0.0])

        yb_desired = np.cross(zb_desired, xc_desired)

        if np.linalg.norm(yb_desired) > 1e-6:
            yb_desired /= np.linalg.norm(yb_desired)
        else:
            yb_desired = np.array([0., 1., 0.])
        xb_desired = np.cross(yb_desired, zb_desired)

        # -----------------Desired rotation matrix ---------------------------------
        # Rd = [xb;yb;zb]
        R_des = np.column_stack((xb_desired, yb_desired, zb_desired))

        # Convert rotation matrix to Euler angles (roll, pitch, yaw)
        target_rpy = Rotation.from_matrix(R_des).as_euler('ZXY')

        # target thrust output 
        target_thrust = F_d







        #Your code ends here

        return target_thrust, target_rpy, pos_e_vec, cur_rotation
    
        # return target_thrust, target_rpy_out.flatten(), er.flatten(), R_curr
        #############################################################################

        


    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               cur_ang_vel,
                               target_euler,
                               target_rpy_rates
                               ):
        """DSL's CF2.x PID attitude control.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_euler : ndarray
            (3,1)-shaped array of floats containing the computed target Euler angles.
        target_rpy_rates : ndarray
            (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """

        #Write your code here

        target_torques = np.zeros(3)   

        ####################################################################################
        R = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        Rd = Rotation.from_euler('ZXY', target_euler,degrees=False).as_matrix()

        # Rotation error: e_R
        Re = Rd.T @ R    

        
        R_T_Rd = R.T @ Rd
        eR = 0.5 * np.array([
            Re[2, 1] - R_T_Rd[2, 1],  
            Re[0, 2] - R_T_Rd[0, 2],  
            Re[1, 0] - R_T_Rd[1, 0]   
        ])

        # Angular velocity error: ew (in body frame)
        ew = target_rpy_rates - cur_ang_vel

        # PD gains (tune these for your hardware)
        # Kp_att = np.array([2, 2, 1])   # Example values
        # Kd_att = np.array([4, 4, 1])
        # Kp_att = np.array([10, 10, 50])   
        # Kd_att = np.array([10, 10, 50])
        # Kp_att = np.array([100, 100, 150])   
        # Kd_att = np.array([100, 100, 200])
        # Kp_att = np.array([50000, 50000, 10000])   
        # Kd_att = np.array([25000, 25000, 9000])
        # Kp_att = np.array([36000, 36000, 35000])   
        # Kd_att = np.array([10000, 10000, 7000])
        Kp_att = np.array([35000, 35000, 30000])   
        Kd_att = np.array([10500, 10500, 6500])

        # self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        # self.I_COEFF_TOR = np.array([.0, .0, 500.])
        # self.D_COEFF_TOR = np.array([20000., 20000., 12000.])

        # Compute desired moments
        target_torques = -Kp_att * eR + Kd_att * ew
     
        

    ##################################################################################    


       


        #Your code ends here

    ################################################################################

        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)

        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    ################################################################################

    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()
