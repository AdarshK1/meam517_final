from import_helper import *
from matplotlib import pyplot as plt

def build_viz_plant():
    # Create a MultibodyPlant for the arm
    file_name = "leg_v2.urdf"
    builder = DiagramBuilder()
    scene_graph = builder.AddSystem(SceneGraph())
    single_leg = builder.AddSystem(MultibodyPlant(0.0))
    single_leg.RegisterAsSourceForSceneGraph(scene_graph)
    Parser(plant=single_leg).AddModelFromFile(file_name)
    single_leg.Finalize()
    return single_leg, builder, scene_graph


def assemble_visualizer(builder, scene_graph, single_leg, x_traj_source):
    demux = builder.AddSystem(Demultiplexer(np.array([3, 3])))
    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(single_leg))
    zero_inputs = builder.AddSystem(ConstantVectorSource(np.zeros(3)))

    builder.Connect(zero_inputs.get_output_port(), single_leg.get_actuation_input_port())
    builder.Connect(x_traj_source.get_output_port(), demux.get_input_port())
    builder.Connect(demux.get_output_port(0), to_pose.get_input_port())
    builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(single_leg.get_source_id()))
    builder.Connect(scene_graph.get_query_output_port(), single_leg.get_geometry_query_input_port())

    ConnectDrakeVisualizer(builder, scene_graph)


def draw_trace(x_traj, visualizer, tf, num_points=1000, plot=False):
    context, single_leg, plant, plant_context = get_plant()

    toe_pos_array = np.zeros((3, num_points))
    lower_pos_array = np.zeros((3, num_points))
    upper_pos_array = np.zeros((3, num_points))
    i = 0
    toe_to_obst_x = []
    toe_to_obst_y = []
    toe_to_obst_z = []
    toe_to_obst_euc = []
    for t in np.linspace(0, tf, num_points):
        x = x_traj.value(t)
        toe_pos = get_world_position(context, single_leg, plant, plant_context, "toe0", x)
        toe_pos_array[:, i] = toe_pos

        toe_to_obst = toe_pos - np.array([0.25, 0.25, 0.1])
        toe_to_obst_x.append(toe_to_obst[0])
        toe_to_obst_y.append(toe_to_obst[1])
        toe_to_obst_z.append(toe_to_obst[2])
        toe_to_obst_euc.append(toe_to_obst.dot(toe_to_obst) ** 0.5)

        lower_pos = get_world_position(context, single_leg, plant, plant_context, "lower0", x)
        lower_pos_array[:, i] = lower_pos

        upper_pos = get_world_position(context, single_leg, plant, plant_context, "upper0", x)
        upper_pos_array[:, i] = upper_pos

        # print(lower_pos, toe_pos)

        i += 1
    if plot:
        plt.plot(toe_to_obst_x)
        plt.plot(toe_to_obst_y)
        plt.plot(toe_to_obst_z)
        plt.plot(toe_to_obst_euc)
        plt.legend(["x", "y", "z", "total"])
        plt.show()

    visualizer.vis['toe_traj_line'].set_object(
        geom.Line(geom.PointsGeometry(toe_pos_array), geom.LineBasicMaterial(color=0x0000ff, linewidth=2)))
    visualizer.vis['lower_traj_line'].set_object(
        geom.Line(geom.PointsGeometry(lower_pos_array), geom.LineBasicMaterial(color=0xff0000, linewidth=2)))
    visualizer.vis['upper_traj_line'].set_object(
        geom.Line(geom.PointsGeometry(upper_pos_array), geom.LineBasicMaterial(color=0x00ff00, linewidth=2)))

    print("Initial toe position:", toe_pos_array[:, 0])
    print("Final toe position:", toe_pos_array[:, -1])


def do_viz(x_traj, u_traj, tf, n_play=1, obstacles=None):
    server_args = ['--ngrok_http_tunnel']

    zmq_url = "tcp://127.0.0.1:6000"
    web_url = "http://127.0.0.1:7000/static/"

    single_leg, builder, scene_graph = build_viz_plant()

    # Create meshcat
    visualizer = ConnectMeshcatVisualizer(
        builder,
        scene_graph,
        scene_graph.get_pose_bundle_output_port(),
        zmq_url=zmq_url,
        server_args=server_args,
        delete_prefix_on_load=False)

    x_traj_source = builder.AddSystem(TrajectorySource(x_traj))
    u_traj_source = builder.AddSystem(TrajectorySource(u_traj))

    assemble_visualizer(builder, scene_graph, single_leg, x_traj_source)

    diagram = builder.Build()
    diagram.set_name("diagram")

    # Visualize obstacles
    if obstacles is not None:
        visualizer.vis.delete()
        obstacles.draw(visualizer)

    # Draw trace of computed trajectory
    draw_trace(x_traj, visualizer, tf)

    visualizer.load()
    print("\n!!!Open the visualizer by clicking on the URL above!!!")

    # Visualize the motion for `n_playback` times
    for i in range(n_play):
        print("Started view: ", i)
        # Set up a simulator to run this diagram.
        simulator = Simulator(diagram)
        initialized = simulator.Initialize()
        simulator.set_target_realtime_rate(1.0)
        simulator.AdvanceTo(tf)
        time.sleep(2)
