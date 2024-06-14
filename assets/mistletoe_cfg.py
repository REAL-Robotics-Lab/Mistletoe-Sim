import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import DCMotorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

# Actuator Configs 

HAA_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*HAA"],
    saturation_effort=24,
    effort_limit=24,
    velocity_limit=7.5,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
)

KFE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*KFE"],
    saturation_effort=9,
    effort_limit=9,
    velocity_limit=7.5,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
)

HFE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*HFE"],
    saturation_effort=9,
    effort_limit=9,
    velocity_limit=7.5,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
)

#Articulation Configs

MISTLETOE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path='./mistletoe_usd/mistletoe.usd',
        
        # Copied from the anymal example
        
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            joint_pos={
                ".*HAA": 0.0,  # all HAA
                "HFE[12]": 0.4,  # both front HFE
                "HFE[34]": -0.4,  # both hind HFE
                "KFE[12]": -0.8,  # both front KFE
                "KFE[34]": 0.8,  # both hind KFE
            },
        ),

        actuators={
            "HAA": HAA_ACTUATOR_CFG,
            "HFE": HFE_ACTUATOR_CFG,
            "KFE": KFE_ACTUATOR_CFG
        },

        soft_joint_pos_limit_factor=0.95
    )
)