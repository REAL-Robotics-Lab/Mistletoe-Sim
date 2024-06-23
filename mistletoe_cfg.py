import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import DCMotorCfg,ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
import os
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
dirname, filename = os.path.split(os.path.abspath(__file__))

# Actuator Configs 

HAA_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*HAA"],
    effort_limit=24,
    velocity_limit=10,
    stiffness={".*": 100.0},
    damping={".*": 10.0},
)

KFE_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*KFE"],
    effort_limit=9,
    velocity_limit=10,
    stiffness={".*": 100.0},
    damping={".*": 10.0},
)

HFE_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*HFE"],
    effort_limit=9,
    velocity_limit=10,
    stiffness={".*": 100.0},
    damping={".*": 10.0},
)

#Articulation Configs

MISTLETOE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-B/anymal_b.usd",        

        usd_path= dirname + '/assets/mistletoe_usd/mistletoe.usd',

        # Adapted from Anymal Velocity Tracking Example's Config
        activate_contact_sensors=True,
        
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),


    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            "LEG_[14]_HFE": 0.523599,  # both front HFE
            "LEG_[23]_HFE": -0.523599,  # both hind HFE
            "LEG_[23]_KFE": -0.872665,  # both front KFE
            "LEG_[14]_KFE": 0.872665,  # both hind KFE
        },
        joint_vel={".*": 0.0},
    ),

    actuators={
        "HAA": HAA_ACTUATOR_CFG,
        "HFE": HFE_ACTUATOR_CFG,
        "KFE": KFE_ACTUATOR_CFG
    },

    soft_joint_pos_limit_factor=0.9
)