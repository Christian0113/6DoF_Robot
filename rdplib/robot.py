import json
import numpy as np
import pathlib
from scipy.integrate import solve_ivp
import yaml

from . import math_helpers as mh

# define class to allow links to be attributes of links
class Link:
    pass

class Link:
    """one body of a serial robot with rotational joints


    Vectors
    ------------------
    - r position
    - v translational velocity
    - a translational acceleration
    - omega angular velocity
    - alpha angular acceleration

    Points in the indizes
    - i origin of the body
    - p origin of the predecessor
    - s origin of the successor
    - CoM center of mass of this body
    
    The indices after one underscore give the two points/bodies between which the vector is the difference. E.G., r_iCoM__i is the vector from i to CoM. A leading 0 is neglected, e.g., v_i__i is the translational velocity of i with respect to 0.

    The index after two underscores gives the cosy in which the vector is represented.
    E.G., r_iCoM__i is represented in the body coordinate system i


    Units
    -----
    - lengths are given in meter
    - angle are given in radians
    - times are given in seconds

    Attributes
    ----------
    dof: int
        number of this body
        - number of the DoF starting at the base
        - first body: dof=1
        - base: dof=0

    alpha_p: float
        angle alpha, DH parameter, time-independent

    a_p: float
        length a, DH parameter, time-independent

    d_i: float
        length d, DH parameter, time-independent

    r_pi__p: np.array((3))
        see module description for an explanation of vectors

    r_i__0: np.array((3))
        see module description for an explanation of vectors

    r_iCoM__i: np.array((3))
        see module description for an explanation of vectors

    A_ip: np.array((3,3))
        base transformation from predecessor cosy to this body cosy

    A_i0: np.array((3,3))
        base transformation from world fixed cosy 0 to this body cosy

    omega_pi__i: np.array((3))
        see module description for an explanation of vectors

    omega_i__i: np.array((3))
        see module description for an explanation of vectors

    v_pi__i: np.array((3))
        see module description for an explanation of vectors

    v_i__i: np.array((3))
        see module description for an explanation of vectors

    alpha_i__i: np.array((3))
        see module description for an explanation of vectors

    a_i__i: np.array((3))
        see module description for an explanation of vectors

    a_CoM__i: np.array((3))
        see module description for an explanation of vectors

    J_Ti__i: np.array((3,n))
        translational jacobi matrix of the origin i represented in the body cosy

    J_R__i: np.array((3,n))
        rotational jacobi matrix of this body represented in the body cosy

    Theta_i_i__i: np.array((3,3))
        mass moment of inertia with respect to the origin of the body cosy

    m: float
        mass in kilo grams

    dM: np.array((size,size))
        part of the total mass matrix in kilo grams

    tau_id: np.array((size))
        part of this link to the joint torques (used in inverse dynamics)
    """
    # number of this body
    dof = -1
    # fix DH parameters
    alpha_p = 0.0
    a_p = 0.0
    d_i = 0.0
    # position and orientation
    r_pi__p = np.zeros((3))
    r_i__0 = np.zeros((3))
    r_iCoM__i = np.zeros((3))
    A_ip = np.eye(3)
    A_i0 = np.eye(3)
    # velocities
    omega_i__i = np.zeros((3))
    v_i__i = np.zeros((3))
    # accelerations
    alpha_i__i = np.zeros((3))
    a_i__i = np.zeros((3))
    a_CoM__i = np.zeros((3))
    # Jacobi matrices
    J_R__i = np.array
    J_Ti__i = np.array
    # inertia parameters
    Theta_i_i__i = np.zeros((3,3))
    m = 0.0
    dM = np.array
    # for id
    tau_id = np.array

    def __init__(self, dof: int, size: int) -> None:
        """constructor

        Parameters
        ----------
        dof: int
            number of this body
            - number of the DoF starting at the base
            - first body: dof=1
            - base: dof=0

        size: int
            number of DoFs of the total robot

        Returns
        ----------
        None
        """
        self.dof = dof
        self.J_R__i = np.zeros((3, size))
        self.J_Ti__i = np.zeros((3, size))
        self.dM = np.zeros((size, size))
        self.tau_id = np.zeros((size))

    def set_dh_parameters(self, alpha_p: float, a_p: float, d_i: float) -> None:
        """initialization of the DH parameters

        Parameters
        ----------
        alpha_p: float
            alpha_p in radians

        a_p: float
            a_p in meter

        d_i: float
            d_i in meter

        Returns
        ----------
        None
        """
        self.alpha_p = alpha_p
        self.a_p = a_p
        self.d_i = d_i

    def set_inertia_parameters(self, mass: float, Theta_i_i__i: np.array, r_iCoM__i: np.array) -> None:
        """initialization of inertia parameters

        Parameters
        ----------
        mass: float
            mass in kilo grams

        Theta_i_i__i: np.array((3,3))
            mass moment of inertia with respect to the origin of the body cosy

        r_iCoM__i: np.array((3))
            positionvector from the origin of the body cosy to the center of mass

        Returns
        ----------
        None
        """
        self.m = mass
        self.Theta_i_i__i = Theta_i_i__i
        self.r_iCoM__i = r_iCoM__i

    def calculate_positions(self, prev: Link, q: float)->None:
        """calculation of the position and orientation of this link based on the position and orientation of the predecessor

        Parameters
        -------
        prev: Link
            predecessor

        q: float
            joint angle

        Returns
        ------
        None

        """


        ###Workspace Start###
        # Self-z joint convention:
        # The joint variable q rotates this link around its own z_i axis.
        # The predecessor-to-link offset is fixed by alpha_p, a_p and d_i;
        # only the orientation is changed by q about z_i.
        ca = np.cos(self.alpha_p)
        sa = np.sin(self.alpha_p)
        cq = np.cos(q)
        sq = np.sin(q)

        # vector from predecessor to this link, represented in predecessor cosy p
        self.r_pi__p = np.array([self.a_p, self.d_i * sa, self.d_i * ca])

        # A_ip = Rz(-q) @ Rx(alpha): passive map p->i for active +q about z_i (self-z)
        self.A_ip = np.array([
            [cq,        sq * ca,  -sq * sa],
            [-sq,       cq * ca,  -cq * sa],
            [0.0,       sa,        ca    ]
        ])

        # absolute position and orientation
        self.A_i0 = self.A_ip @ prev.A_i0
        self.r_i__0 = prev.r_i__0 + prev.A_i0.T @ self.r_pi__p
        ###Workspace End###

    def calculate_velocities(self, prev: Link, dot_q: float)-> None:
        """calculation of the velocity of this link based on the velocity of the predecessor

        Prerequisites
        -------
        - calculate_positions(args)

        Parameters
        -------
        prev: Link
            predecessor


        dot_q: float
            joint velocity

        Returns
        ------
        None

        """

        ###Workspace Start###
        z__i = np.array([0.0, 0.0, 1.0])

        # difference angular velocity this link minus predecessor
        # revolute joint axis is the current link z_i-axis
        self.omega_pi__i = dot_q * z__i

        # angular velocity
        omega_p__i = self.A_ip @ prev.omega_i__i
        self.omega_i__i = omega_p__i + self.omega_pi__i

        # translational velocity
        self.v_i__i = self.A_ip @ (
            prev.v_i__i + np.cross(prev.omega_i__i, self.r_pi__p)
        )
        ###Workspace End###

    def calculate_accelerations(self, prev: Link, ddot_q: float) -> None:
        """calculation of the acceleration of this link based on the acceleration of the predecessor

        Prerequisites
        -------
        - calculate_positions(args)
        - calculate_velocities(args)

        Parameters
        -------
        prev: Link
            predecessor

        ddot_q: float
            joint acceleration

        Returns
        ------
        None

        """


        ###Workspace Start
        z__i = np.array([0.0, 0.0, 1.0])

        omega_p__i = self.A_ip @ prev.omega_i__i
        alpha_p__i = self.A_ip @ prev.alpha_i__i

        # angular acceleration
        self.alpha_i__i = (
            alpha_p__i
            + ddot_q * z__i
            # transport term because prev.omega is first expressed in this link frame
            + np.cross(omega_p__i, self.omega_pi__i)
        )

        # translational acceleration
        self.a_i__i = self.A_ip @ (
            prev.a_i__i
            + np.cross(prev.alpha_i__i, self.r_pi__p)
            + np.cross(prev.omega_i__i, np.cross(prev.omega_i__i, self.r_pi__p))
        )

        # translation acceleration of CoM
        self.a_CoM__i = (
            self.a_i__i
            + np.cross(self.alpha_i__i, self.r_iCoM__i)
            + np.cross(self.omega_i__i, np.cross(self.omega_i__i, self.r_iCoM__i))
        )
        ###Workspace End###

    def calculate_jacobis(self, prev: Link) -> None:
        """calculation of the jacobis based on the jacobis of the predecessor

        Prerequisites
        --------
        - calculate_positions(args)

        Parameters
        -------
        prev: Link
            predecessor

        Returns
        -----
        None

        """


        ###Workspace Start###
        # rotational
        # self.J_R__i = ...
        # translational
        # self.J_Ti__i = ...
        ###Workspace End###

    def calculate_mass_matrix(self)->None:
        """calculation of the contribution of this link to the total mass matrix

        Prerequisites
        --------
        - calculate_positions(args)
        - calculate_jacobis(args)

        Returns
        ------
        None

        """


        ###Workspace Start###
        #self.dM = ...
        ###Workspace End###

    def calculate_id(self, g__0: np.array((3))) -> None:
        """inverse dynamics, calculation of the joint torques

        Prerequisites
        -------
        - calculate_positions(args)
        - calculate_velocities(args)
        - calculate_accelerations(args)
        - calculate_jacobis()

        Parameters
        -------
        g__0: np.array((3))
            gravity vector in cosy 0

        Returns
        -------
        None

        """

        ###Workspace Start###
        # newton
        # F = ...
        # euler
        # T = ...
        # projection into direction compatible with the constraints and add to tau_id
        # self.tau_id = ...
        ###Workspace End###

class Robot:
    """robot class for serial robots with revolute joints

    Attributes
    ----------
    size: int
        number of joints

    links: list[Link]
        list of links

    r_NTCP__N: np.array((3))
        vector from origin of the last cosy to the TCP represented in the last cosy

    g__0: np.array((3))
        gravity vector in cosy 0

    home_q: np.array((size))
        home position in joint space in radians, default: Nullvektor

    base: Link
        base of the robot

    r_TCP__0 : np.array((3))
        position of the TCP in cosy 0 represented in cosy 0 in meter

    v_TCP__0 : np.array((3))
        translational velocity of the TCP represented in cosy 0 in meter/seconds

    a_TCP__0 : np.array((3)): np.array((3))
        translation acceleration of the TCP reperesented in cosy 0 in meter/seconds²

    q: np.array((size))
        joint angles in radians

    dot_q: np.array((size))
        joint velocities in radians/seconds

    ddot_q: np.array((size))
        joint accelerations in radians/seconds²

    J_TTCP__0: np.array((3,size))
        translational jacobian of the TCP in cosy 0

    J_RTCP__0: np.array((3,size))
        rotational jacobian of the last link in cosy 0

    M: np.array((size,size))
        massmatrix

    tau_id: np.array((size))
        joint torques

    tau_drive: np.array((size))
        drive torques
    """
    size = 0
    links : list[Link]
    r_NTCP__N : np.array((3))
    g__0 = np.array((3))
    q_home = np.array
    q_max = np.array
    q_min = np.array
    q_comfort = np.array
    base : Link
    r_TCP__0 : np.array((3))
    v_TCP__0 : np.array((3))
    a_TCP__0 : np.array((3))
    q : np.array
    dot_q : np.array
    ddot_q : np.array
    J_RTCP__0 : np.array
    J_TTCP__0 : np.array
    M : np.array
    tau_id : np.array
    tau_antrieb : np.array

    def __init__(self, links: list[Link]) -> None:
        self.size = len(links)
        self.links = links
        self.r_NTCP__N = np.zeros((3))
        self.g__0 = np.zeros((3))
        self.base = Link(0, self.size)
        self.r_TCP__0 = np.zeros((3))
        self.v_TCP__0 = np.zeros((3))
        self.a_TCP__0 = np.zeros((3))
        self.q = np.zeros((self.size))
        self.dot_q = np.zeros((self.size))
        self.ddot_q = np.zeros((self.size))
        self.J_RTCP__0 = np.zeros((3,self.size))
        self.J_TTCP__0 = np.zeros((3,self.size))
        self.M = np.zeros((self.size,self.size))
        self.tau_id = np.zeros((self.size))
        self.tau_antrieb = np.zeros((self.size))
        self.q_home = np.zeros((self.size))
        self.q_max = np.pi * np.ones((self.size))
        self.q_min = -np.pi * np.ones((self.size))
        self.q_comfort = np.zeros((self.size))

    def calculate_positions(self, q: np.array) -> None:
        """direct kinematics (position)
        calculation of the position and orientation of all links and the TCP

        Parameters
        ----------
        q: np.array
            Gelenkwinkel in [rad]

        Returns
        -------
        None
        """

        for link in self.links:
            if link.dof == 1:
                link.calculate_positions(self.base, q[link.dof-1])
            else:
                link.calculate_positions(self.links[link.dof-2], q[link.dof-1])


        ###Workspace Start###
        last = self.links[-1]
        self.r_TCP__0 = last.r_i__0 + last.A_i0.T @ self.r_NTCP__N
        ###Workspace End###

        self.q = np.copy(q)

    def calculate_velocities(self, dot_q: np.array) -> None:
        """direct kinematics (velocity)
        calculation of the velocities of all links and the TCP

        Prerequisites
        ----------
        - calculate_positions(args)

        Parameters
        ----------
        dot_q: np.array
            joint velocities

        Returns
        -------
        None
        """


        ###Workspace Start###
        # calculation of the velocity of all links
        for link in self.links:
            if link.dof == 1:
                link.calculate_velocities(self.base, dot_q[link.dof-1])
            else:
                link.calculate_velocities(self.links[link.dof-2], dot_q[link.dof-1])

        # calculation of the velocity of the TCP
        last = self.links[-1]
        v_TCP__N = last.v_i__i + np.cross(last.omega_i__i, self.r_NTCP__N)
        self.v_TCP__0 = last.A_i0.T @ v_TCP__N
        ###Workspace End###

        self.dot_q = np.copy(dot_q)

    def calculate_accelerations(self, ddot_q: np.array) -> None:
        """direct kinematics (acceleration)
        calculation of the accelerations of all links and the TCP

        Prerequisites
        ----------
        - calculate_positions(args)
        - calculate_velocities(args)

        Parameters
        ----------
        ddot_q: np.array
            Gelenkwinkelbeschleunigungen in [rad/s²]

        Returns
        -------
        None
        """


        ###Workspace Start###
        # calculation of the acceleration of all links
        for link in self.links:
            if link.dof == 1:
                link.calculate_accelerations(self.base, ddot_q[link.dof-1])
            else:
                link.calculate_accelerations(self.links[link.dof-2], ddot_q[link.dof-1])

        # calculation of the acceleration of the TCP
        last = self.links[-1]
        a_TCP__N = (
            last.a_i__i
            + np.cross(last.alpha_i__i, self.r_NTCP__N)
            + np.cross(last.omega_i__i, np.cross(last.omega_i__i, self.r_NTCP__N))
        )
        self.a_TCP__0 = last.A_i0.T @ a_TCP__N
        ###Workspace End###

        self.ddot_q = np.copy(ddot_q)

    def calculate_jacobis(self) -> None:
        """direct kinematics (jacobian)
        calculation of the jacobians of all links and the TCP

        Prerequisites
        ----------
        - calculate_positions(args)

        Returns
        -------
        None
        """


        ###Workspace Start###
        # calculation of the jacobians of all links
        # ...
        # calculation of the jacobians of the TCP
        # self.J_RTCP__0 = ...
        # self.J_TTCP__0 = ...
        ###Workspace End###

    def calculate_mass_matrix(self) -> None:
        """calculation of the mass matrix

        Prerequisites
        ----------
        - calculate_positions(args)
        - calculate_jacobis()

        Returns
        -------
        None
        """


        ###workspace Start###
        # build total mass matrix M from the contribution of each link
        ###Workspace End###

    def calculate_id(self) -> None:
        """inverse dynamics
        calculation of the joint torques

        Prerequisites
        ----------
        - calculate_positions(args)
        - calculate_velocities(args)
        - calculate_accelerations(args)
        - calculate_jacobis()

        Returns
        -------
        None
        """

        ###Workspace Start###
        # add the contribution of each link to the total joint torques
        ###Workspace End###

    def calculate_eom(self) -> np.array:
        """direct dynamics
        calculation of the joint accelerations from the joint torques

        Prerequisites
        ---------
        - calculate_positions(args)
        - calculate_velocities(args)
        - calculate_jacobis()

        Returns
        --------
        ddot_q: np.array((size))
            joint accelerations

        """


        ###Workspace Start###
        # calculatio h-vector
        # a...
        # h = ...
        # calculate mass matrix
        # ...
        # solve system of equations
        # ddot_q = ...
        # return ddot_q
        ###Workspace End###

    def calculate_tmstp_euler(self, dt: float) -> None:
        """calculation of the joint accelerations, velocities and positions
        through time integration with an explicit Euler method

        Parameters
        --------

        dt: float
            time step in seconds

        Returns
        -------
        None

        """


        ###Workspace Start###
        # calculation of the joint accelerations
        # self.ddot_q = ...
        # time integration for velocities and positions (explicite euler)
        # q = ...
        # dot_q = ...
        ###Workspace End###

        # direct kinematics with new positions and velocities
        self.calculate_positions(q)
        self.calculate_velocities(dot_q)
        self.calculate_jacobis()


    ###Workspace Start###
    # def ode_func(self, t: float, y: np.array) -> np.array:
    #     """statespace formulation of the equations of motion for an ode solver
    #     (e.g. scipy.integrate.RK45 -> fun)
    #     ... """
    #     ...
    ###Workspace End###

    def calculate_tmstp_rk45(self, dt: float) -> None:
        """calculation of the joint accelerations, velocities and positions
        through time integration with the Runge-Kutta method (RK45)

        Parameters
        --------

        dt: float
            time step in seconds

        Returns
        -------
        None

        """


        ###Workspace Start###
        # get q, dot_q, ddot_q from time integration with RK45
        # ...
        # direct kinematics with new positions and velocities
        # ...
        # self.calculate_positions(???)
        # self.calculate_velocities(???)
        ###Workspace End###

        self.calculate_jacobis()


def robot_factory(path: str) -> Robot:
    """create a robot object based on a robot config file
    
    Supported formats
    --------
    - .yaml
    - .json

    Parameters
    ---------

    path: string
        path to the robot config file

    Returns
    --------

    new_robot: robot
        new robot object
    """

    #Lesen der Datei
    data = {}
    tmp = pathlib.Path(path).suffix
    match tmp:
        case '.json':
            with open (path,'r') as f:
                data = json.load(f)
        case '.yaml':
            with open (path,'r') as f:
                data = yaml.load(f,Loader=yaml.loader.SafeLoader)
        case _:
            print('config file type not valid')
            return None
    
    #Erstellen des Roboters
    return __parse_dict_to_robot(data)

def __eval_expression(expression: str) -> any:
    """Reliable evaluation of strings from the configuration data \\
    - Lengths are specified in meter by default
    - Angles are specified in radians by default
    - Angles can be formulated as a function of pi, e.g. alpha_p = pi/2
    - If degrees are to be used, this must be explicitly specified by \'deg\', e.g. alpha_p = 90 deg

    Parameters
    ---------
    expression: str
        expression to evaluate
    
    Returns
    --------
    evaluated: float | np.array
        resulting number
    """

    # replace \'def\' by pi/180
    expression = expression.replace('deg','* pi/180.0')
    # remove \'
    expression = expression.replace('\'','')
    # define alias for pi
    allowed_names = {'pi': np.pi}
    code = compile(expression, '<string>', 'eval')
    for name in code.co_names:
        if name not in allowed_names:
            raise NameError(f'Use of {name} not allowed')
    return np.array(eval(code, {'__builtins__': {}}, allowed_names))

def __set_attributes_to_object(object: any, obj_dict: dict, optional: list, required: list, \
                             disable_warnings: bool = True):
    """Assigns a value from a dictionary to the attribute of a class instance
    
    Parameters
    --------

    object: any
        object that gets the attributes
    
    obj_dict: dict
        dictionary with the attributes

    optional: list[str]
        list with the names of the optional attributes, if these are missing in the dictionary, a warning is issued

    required: list[str]
        list with the names of the required attributes; if these are missing in the dictionary, an error is displayed
    
    disable_warnings: bool
        warnings are deactivated
    """

    for param in required:
        expr = obj_dict.get(param)
        if expr != None:
            object.__setattr__(param,__eval_expression(str(expr)))
        else:
            raise ValueError(f'{param} is missing, but is a required parameter for a link')
    for param in optional:
        expr = obj_dict.get(param)
        if expr != None:
            object.__setattr__(param,__eval_expression(str(expr)))
        elif not disable_warnings:
            print(f'{param} is missing, using default value:\n {object.__getattribute__(param)} \n')

def __parse_dict_to_robot(rob_dict: dict) -> Robot:

    # create empty list for links
    link_dict = rob_dict.get('links')
    if link_dict == None:
        raise ValueError('robot without a list of links is not valid')
    links = list()

    # iteration over the links
    size = len(link_dict)
    for i in range(size):
        links.append(Link(i+1, size))
        # set necessary and optional parameters for a link
        required = ['alpha_p', 'a_p', 'd_i'] # necessary -> error if not in config file
        optional = ['m', 'Theta_i_i__i', 'r_iCoM__i'] # optional -> warning if not in confi file
        __set_attributes_to_object(links[i], link_dict[i], optional, required)
    
    # create robot object
    new_robot = Robot(links)
    # set necessary and optional parameters for a robot
    required = ['r_NTCP__N']
    optional = ['g__0']
    __set_attributes_to_object(new_robot, rob_dict, optional, required)
    
    # read angle configurations
    angle_configs = ['q_home', 'q_max', 'q_min', 'q_comfort']
    for name in angle_configs:
        angles = rob_dict.get(name)
        if angles != None:
            angles = __eval_expression(str(rob_dict[name]))
            if np.size(angles) < size:
                diff = size-np.size(angles)
                angles = np.concatenate((angles,np.zeros(diff)))
            else:
                angles = np.array(angles[0:size])
            new_robot.__setattr__(name, angles)
        else:
            print(f'{name} is missing, using default value:\n {new_robot.__getattribute__(name)}\n')

    return new_robot 
