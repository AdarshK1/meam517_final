from import_helper import *

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
        obstacles.draw(visualizer)

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