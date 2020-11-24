from import_helper import *


def get_plant():
    '''
    :return: context, single_
    '''
    builder = DiagramBuilder()
    plant = builder.AddSystem(MultibodyPlant(0.0))
    file_name = "leg_v2.urdf"
    Parser(plant=plant).AddModelFromFile(file_name)
    plant.Finalize()
    single_leg = plant.ToAutoDiffXd()

    plant_context = plant.CreateDefaultContext()
    context = single_leg.CreateDefaultContext()

    return context, single_leg, plant, plant_context


def get_limits(n_u, n_x, plant):
    # Store the actuator limits here
    effort_limits = np.zeros(n_u)
    for act_idx in range(n_u):
        effort_limits[act_idx] = \
            plant.get_joint_actuator(JointActuatorIndex(act_idx)).effort_limit()
    vel_limits = 15 * np.ones(n_x // 2)

    return effort_limits, vel_limits


def get_transform(plant, context, parent_frame_name, child_frame_name):
    parent = plant.GetFrameByName(parent_frame_name)
    child = plant.GetFrameByName(child_frame_name)

    transform = plant.CalcRelativeTransform(context, parent, child)

    return transform.translation(), transform.rotation().ToQuaternion()

def get_world_position(context, single_leg, plant, plant_context, frame_name, x):
    world_frame = single_leg.world_frame()
    base_frame = single_leg.GetFrameByName(frame_name)

    # Choose plant and context based on dtype.
    if x.dtype == float:
        # print("using float")
        plant_eval = plant
        context_eval = plant_context
    else:
        # print("using auto")
        # Assume AutoDiff.
        plant_eval = single_leg
        context_eval = context

    plant_eval.SetPositionsAndVelocities(context_eval, x)

    # Do forward kinematics.
    frame_xyz = plant_eval.CalcRelativeTransform(context_eval, resolve_frame(plant_eval, world_frame),
                                               resolve_frame(plant_eval, base_frame))

    return frame_xyz.translation()

def resolve_frame(plant, F):
    """Gets a frame from a plant whose scalar type may be different."""
    return plant.GetFrameByName(F.name(), F.model_instance())


def drake_quat_to_floats(drake_quat):
    return drake_quat.w(), drake_quat.x(), drake_quat.y(), drake_quat.z()


def get_angle_from_context(plant, context, joint_name):
    return plant.GetJointByName(joint_name).get_angle(context)


def set_angle_in_context(plant, context, joint_name, angle):
    joint = plant.GetJointByName(joint_name)
    joint.set_angle(context, angle)

    return context


def create_context_from_angles(plant, name_to_angle_dict):
    context = plant.CreateDefaultContext()

    for name, angle in name_to_angle_dict.items():
        joint = plant.GetJointByName(name)
        joint.set_angle(context, angle)

    return context
