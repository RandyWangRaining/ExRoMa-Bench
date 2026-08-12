"""Simulation collection and evaluation entry point."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from collections import Counter
from pathlib import Path

from isaaclab.app import AppLauncher

from exroma_bench.paths import PROJECT_ROOT
from exroma_bench.sim.defaults import DEFAULT_EVALUATION_EPISODES, resolve_task_seed
from exroma_bench.tasks.benchmark_suite.registry import BENCHMARK_TASKS

TASKS = tuple(BENCHMARK_TASKS)
SCENES = (
    "ground_plane",
    "lunalab",
    "moon_surface",
    "mars_surface",
    "procedural_moon",
    "procedural_mars",
    "oberpfaffenhofen",
)
ROBOTTWIN_TASKS = {"stack_blocks_two", "handover_block", "scan_object", "scan_rock"}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "collect", "evaluate"))
    parser.add_argument("--task", choices=TASKS, default="stack_blocks_two")
    parser.add_argument("--scene", choices=SCENES, default="procedural_moon")
    parser.add_argument("--attempts", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVALUATION_EPISODES)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Task randomization seed; defaults to 100000 for evaluate and 0 otherwise.",
    )
    parser.add_argument(
        "--terrain-seed",
        type=int,
        default=0,
        help="SimForge terrain seed; independent from task randomization.",
    )
    parser.add_argument(
        "--terrain-size",
        type=float,
        default=32.0,
        help="Width and length in metres for procedural SimForge terrain.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--material", choices=("lunar", "original"), default="lunar")
    parser.add_argument("--strict-collision-check", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--record-fps", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--keep-failed-recordings",
        action="store_true",
        help="Keep failed collect attempts as HDF5/video episodes marked success=False.",
    )
    parser.add_argument(
        "--skip-base-approach",
        action="store_true",
        help="Start at the manipulation pose and skip the rover approach phase.",
    )
    parser.add_argument(
        "--policy-host",
        help="WebSocket policy server host. Valid with evaluate only.",
    )
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--policy-connect-timeout", type=float, default=30.0)
    parser.add_argument("--policy-response-timeout", type=float, default=30.0)
    parser.add_argument(
        "--policy-frequency",
        type=float,
        default=10.0,
        help="Remote policy action frequency in Hz.",
    )
    parser.add_argument(
        "--policy-action-horizon",
        type=int,
        default=16,
        help="Maximum number of returned actions consumed from each policy query.",
    )
    parser.add_argument("--policy-jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--record-policy-video",
        action="store_true",
        help="Record three-view video during remote-policy evaluation.",
    )
    parser.add_argument(
        "--record-policy-video-limit",
        type=int,
        help="Maximum number of remote-policy episodes to record; defaults to all episodes.",
    )
    parser.add_argument(
        "--policy-video-max-seconds",
        type=float,
        help="Stop each policy video after this many simulated seconds without stopping evaluation.",
    )
    parser.add_argument(
        "--policy-video-only",
        action="store_true",
        help="Delete the temporary episode HDF5 after successfully exporting its MP4.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _preload_curobo() -> None:
    import importlib

    for module_name in (
        "torch",
        "curobo.curobolib.geom_cu",
        "curobo.curobolib.kinematics_fused_cu",
        "curobo.curobolib.lbfgs_step_cu",
        "curobo.curobolib.line_search_cu",
        "curobo.curobolib.tensor_step_cu",
    ):
        importlib.import_module(module_name)

    print("[EXROMA][CUROBO]: CUDA extensions preloaded", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    run_exit_code = 1
    args.seed = resolve_task_seed(args.command, args.seed)
    if args.attempts <= 0 or args.episodes <= 0:
        parser.error("--attempts and --episodes must be positive")
    if args.terrain_size <= 0.0:
        parser.error("--terrain-size must be positive")
    if args.policy_host and args.command != "evaluate":
        parser.error("--policy-host is only valid with the evaluate command")
    if not 1 <= args.policy_port <= 65535:
        parser.error("--policy-port must be between 1 and 65535")
    if args.policy_connect_timeout <= 0.0 or args.policy_response_timeout <= 0.0:
        parser.error("policy timeouts must be positive")
    if args.policy_frequency <= 0.0 or args.policy_action_horizon <= 0:
        parser.error("policy frequency and action horizon must be positive")
    if not 1 <= args.policy_jpeg_quality <= 100:
        parser.error("--policy-jpeg-quality must be between 1 and 100")
    if args.record_policy_video and not args.policy_host:
        parser.error("--record-policy-video requires --policy-host")
    if args.record_policy_video_limit is not None and args.record_policy_video_limit <= 0:
        parser.error("--record-policy-video-limit must be positive")
    if args.policy_video_max_seconds is not None and args.policy_video_max_seconds <= 0.0:
        parser.error("--policy-video-max-seconds must be positive")
    if (
        args.record_policy_video_limit is not None
        or args.policy_video_max_seconds is not None
        or args.policy_video_only
    ) and not args.record_policy_video:
        parser.error("policy video options require --record-policy-video")
    automatic = args.command in {"collect", "evaluate"}
    remote_policy_evaluation = args.command == "evaluate" and bool(args.policy_host)
    if automatic and not remote_policy_evaluation:
        _preload_curobo()
    args.enable_cameras = bool(args.command == "collect" or args.policy_host)
    args.exroma_enable_cameras = args.enable_cameras
    sim_gpu = int(os.environ.get("EXROMA_SIM_GPU", "0"))
    args.active_gpu = sim_gpu
    args.physics_gpu = sim_gpu
    args.multi_gpu = _env_flag("EXROMA_MULTI_GPU", default=False)
    if args.device == "cuda:0" and sim_gpu != 0:
        args.device = f"cuda:{sim_gpu}"
    # AppLauncher receives the parsed namespace directly. Keep ExRoMa's positional
    # command and task arguments away from Kit's own command-line parser.
    sys.argv = [sys.argv[0]]
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        result = _run(args, simulation_app)
        run_exit_code = int(result)
        print("[EXROMA]: runtime loop finished", flush=True)
        return result
    except BaseException:
        print("[EXROMA][ERROR]: runtime failed", flush=True)
        traceback.print_exc()
        raise
    finally:
        print("[EXROMA]: closing Isaac Sim", flush=True)
        try:
            simulation_app.close(
                wait_for_replicator=not args.headless,
                skip_cleanup=bool(args.headless),
            )
        except TypeError as exc:
            if "skip_cleanup" not in str(exc):
                raise
            if args.headless and os.environ.get("EXROMA_FAST_HEADLESS_EXIT") == "1":
                print("[EXROMA]: fast headless shutdown for Isaac Sim 4.5", flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(run_exit_code)
            simulation_app.close(wait_for_replicator=not args.headless)


def _run(args, simulation_app) -> int:
    print("[EXROMA]: loading Isaac Lab runtime modules", flush=True)
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Usd, UsdGeom, UsdPhysics

    from exroma_bench.tasks.benchmark_suite import apply_benchmark_materials

    from .assembly import assemble_dual_piper
    from .base_approach import BasePoseApproachController
    from .domains import domain_for_scene
    from .rover_drive import PragyanDrive
    from .scene import (
        TABLE_HEIGHT,
        TABLE_LENGTH,
        TABLE_WIDTH,
        TABLE_X,
        TABLE_Y,
        build_scene_cfg,
        ensure_terrain_collisions,
    )
    from .scene_alignment import snap_roots_to_terrain

    print("[EXROMA]: building simulation context", flush=True)
    domain = domain_for_scene(args.scene)
    task_goal_y = {
        "stack_blocks_two": -0.10,
        "beat_block_hammer": -0.05,
    }
    goal_y = task_goal_y.get(args.task, 0.05)
    robot_start_y = goal_y if args.skip_base_approach else goal_y + 0.60
    sim_cfg = sim_utils.SimulationCfg(
        device=args.device,
        dt=0.02,
        render_interval=2,
        gravity=(0.0, 0.0, -domain.gravity),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        render=sim_utils.RenderCfg(
            enable_translucency=True,
            enable_reflections=True,
        ),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view((2.4, 2.2, 1.8), (0.0, -0.35, 0.5))
    print("[EXROMA]: composing scene configuration", flush=True)
    scene_cfg = build_scene_cfg(
        scene_name=args.scene,
        task_name=args.task,
        robot_start_y=robot_start_y,
        spacing=args.terrain_size,
        material_style=args.material,
        enable_cameras=args.exroma_enable_cameras,
        terrain_seed=args.terrain_seed,
        fix_dual_piper_root=args.skip_base_approach,
    )
    if args.command in {"collect", "evaluate"}:
        scene_cfg.dual_piper.actuators["arms"].effort_limit_sim = 100.0
        scene_cfg.dual_piper.actuators["arms"].stiffness = 160.0
        scene_cfg.dual_piper.actuators["arms"].damping = 16.0
        scene_cfg.dual_piper.actuators["grippers"].stiffness = 220.0
        scene_cfg.dual_piper.actuators["grippers"].damping = 30.0
        scene_cfg.dual_piper.actuators["grippers"].velocity_limit_sim = 0.08
    scene = InteractiveScene(scene_cfg)
    if args.scene == "ground_plane":
        terrain_meshes, collision_apis_added = 0, 0
    else:
        terrain_meshes, collision_apis_added = ensure_terrain_collisions(
            get_current_stage(), "/World/envs/env_0/terrain"
        )
    print(
        f"[EXROMA]: terrain meshes={terrain_meshes}, collision APIs added={collision_apis_added}",
        flush=True,
    )
    terrain_source = (
        "simforge_foundry"
        if args.scene in {"procedural_moon", "procedural_mars"}
        else "external_asset"
    )
    print(
        f"[EXROMA]: terrain source={terrain_source}, "
        f"size={args.terrain_size:g} m, seed={args.terrain_seed}",
        flush=True,
    )
    if args.scene in {"moon_surface", "procedural_moon"}:
        snap_result = snap_roots_to_terrain(
            get_current_stage(),
            terrain_path="/World/envs/env_0/terrain",
            root_paths=(
                "/World/envs/env_0/lunar_outpost_part_1",
                "/World/envs/env_0/lunar_outpost_part_2",
            ),
        )
        if snap_result is None:
            print("[EXROMA][WARN]: lunar outpost could not be snapped to terrain", flush=True)
        else:
            print(
                "[EXROMA]: lunar outpost grounded; "
                f"sampled_z=[{snap_result.minimum_height:.3f}, "
                f"{snap_result.maximum_height:.3f}] m, "
                f"shift={snap_result.shift_z:+.3f} m",
                flush=True,
            )
    print("[EXROMA]: assembling rover and dual arms", flush=True)
    if args.skip_base_approach:
        print("[EXROMA]: dual-arm root fixed at the manipulation pose", flush=True)
    else:
        assemble_dual_piper()
    _hide_pragyan_solar_panel(get_current_stage(), Usd, UsdGeom, UsdPhysics)
    sim.reset()
    _write_articulation_defaults(rover=scene["rover"], robot=scene["dual_piper"])
    scene.reset()
    scene["dual_piper"].set_joint_position_target(scene["dual_piper"].data.default_joint_pos)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    print("[EXROMA]: simulation initialized", flush=True)

    if args.task in ROBOTTWIN_TASKS:
        bound = apply_benchmark_materials(get_current_stage(), args.task, args.material)
        if bound:
            print(f"[EXROMA]: lunar material bindings={bound}", flush=True)

    rover = scene["rover"]
    robot = scene["dual_piper"]
    drive = PragyanDrive(rover)
    joint_targets = robot.data.default_joint_pos.clone()
    initial_state = scene.get_state(is_relative=False)
    cameras = {}
    if args.exroma_enable_cameras:
        cameras = {
            "mast": scene["mast_camera"],
            "front_left": scene["front_left_camera"],
            "front_right": scene["front_right_camera"],
        }

    if args.command == "preview":
        print(
            f"[EXROMA]: preview ready task={args.task}, scene={args.scene}, "
            f"gravity={domain.gravity:.5f} m/s^2",
            flush=True,
        )
        steps = 0
        while simulation_app.is_running():
            robot.set_joint_position_target(joint_targets)
            drive.stop()
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
            steps += 1
            if args.max_steps is not None and steps >= args.max_steps:
                break
        return 0

    import torch

    from exroma_bench.recording import MobileAlohaEpisodeRecorder
    from exroma_bench.tasks.beat_block_hammer import (
        CuroboBeatBlockHammerConfig,
        CuroboBeatBlockHammerController,
    )
    from exroma_bench.tasks.benchmark_suite import (
        benchmark_record_objects,
        create_benchmark_controller,
        sample_task_prompt,
        write_task_prompt_file,
    )
    from exroma_bench.tasks.test_tube_rack import (
        CuroboTestTubeRackConfig,
        CuroboTestTubeRackController,
    )

    controller, record_objects = _create_controller(
        args,
        robot,
        joint_targets,
        scene,
        create_benchmark_controller,
        benchmark_record_objects,
        CuroboTestTubeRackConfig,
        CuroboTestTubeRackController,
        CuroboBeatBlockHammerConfig,
        CuroboBeatBlockHammerController,
        TABLE_X,
        TABLE_Y,
        TABLE_HEIGHT,
        TABLE_LENGTH,
        TABLE_WIDTH,
    )
    if args.skip_base_approach:
        _relax_base_settle_gates(controller)

    base_controller = BasePoseApproachController(
        rover,
        goal_x=0.0,
        goal_y=goal_y,
    )
    rng = random.Random(args.seed)
    prompt_rng = random.Random(args.seed + 104729)
    target_attempts = args.attempts if args.command == "collect" else args.episodes
    all_env_ids = torch.arange(scene.num_envs, device=scene.device, dtype=torch.int32)
    output_dir = args.output or (
        PROJECT_ROOT
        / ("datasets" if args.command == "collect" else "evaluations")
        / f"{args.task}_{args.scene}_seed{args.seed}_run{target_attempts}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "evaluate" and args.policy_host:
        policy_cameras = {
            "cam_high": cameras["mast"],
            "cam_left_wrist": cameras["front_left"],
            "cam_right_wrist": cameras["front_right"],
        }
        return _run_remote_policy_evaluation(
            args=args,
            simulation_app=simulation_app,
            sim=sim,
            scene=scene,
            rover=rover,
            robot=robot,
            drive=drive,
            joint_targets=joint_targets,
            initial_state=initial_state,
            controller=controller,
            cameras=policy_cameras,
            record_objects=record_objects,
            rng=rng,
            prompt_rng=prompt_rng,
            target_episodes=target_attempts,
            output_dir=output_dir,
            domain_name=domain.name,
        )
    recorder = None
    if args.command == "collect":
        recorder = MobileAlohaEpisodeRecorder(
            output_dir,
            fps=args.record_fps,
            jpeg_quality=args.jpeg_quality,
            task_instruction=BENCHMARK_TASKS[args.task].instruction,
            format_name=f"exroma.dual_piper.{args.task}.v1",
            joint_encoding=MobileAlohaEpisodeRecorder.DUAL_PIPER_COMPACT_ENCODING,
            export_video=not args.no_video,
        )
        if args.task in ROBOTTWIN_TASKS:
            write_task_prompt_file(output_dir, args.task)

    attempts = 0
    successes = 0
    records: list[dict[str, object]] = []
    elapsed = 0.0
    current_sample = None
    current_prompt = BENCHMARK_TASKS[args.task].instruction
    current_prompt_index = 0
    current_prompt_source = "exroma_builtin"

    def reset_attempt() -> None:
        nonlocal current_sample, current_prompt, current_prompt_index, current_prompt_source
        scene.reset_to(initial_state, env_ids=all_env_ids, is_relative=False)
        scene.reset()
        joint_targets.copy_(robot.data.default_joint_pos)
        controller.joint_targets = joint_targets
        robot.set_joint_position_target(joint_targets)
        drive.stop()
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        current_sample = controller.randomize_and_start(rng)
        if args.task in ROBOTTWIN_TASKS:
            current_prompt, current_prompt_index, current_prompt_source = sample_task_prompt(
                args.task, prompt_rng
            )
        if args.skip_base_approach:
            base_controller.complete()
        else:
            base_controller.reset()
        if recorder is not None:
            recorder.start(
                metadata={
                    "benchmark": "ExRoMa-Bench",
                    "task": args.task,
                    "scene": args.scene,
                    "domain": domain.name,
                    "seed": args.seed,
                    "terrain": {
                        "source": terrain_source,
                        "seed": args.terrain_seed,
                        "size_m": args.terrain_size,
                    },
                    "sampled_object_pose": current_sample,
                    "task_prompt": {
                        "text": current_prompt,
                        "index": current_prompt_index,
                        "source": current_prompt_source,
                    },
                    "collision_check": {
                        "curobo_table_obstacle": True,
                        "curobo_arm_self_collision": bool(args.strict_collision_check),
                        "isaac_physx_contacts": True,
                    },
                    "policy_interface": {
                        "joint_encoding": "dual_piper_compact",
                        "robot_state_dim": 16,
                        "action_dim": 16,
                        "mast_fixed": True,
                    },
                },
                joint_names=list(robot.joint_names),
                camera_names=list(cameras),
                task_instruction=current_prompt,
            )

    reset_attempt()
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running() and attempts < target_attempts:
        if base_controller.is_active:
            hold = getattr(controller, "hold_pre_manipulation_objects", None)
            if callable(hold):
                hold()
            base_controller.update(sim_dt)
            linear_command, angular_command = base_controller.base_command()
        elif base_controller.is_failed:
            linear_command, angular_command = 0.0, 0.0
            if not controller.is_terminal:
                fail = getattr(controller, "_fail", None)
                if callable(fail):
                    fail(base_controller.failure_reason)
                else:
                    controller.failure_reason = base_controller.failure_reason
                    controller.state = "failed"
        else:
            controller.update(sim_dt)
            linear_command, angular_command = controller.base_command()

        mast_ids, _ = robot.find_joints(["camera_stand_.*_joint"], preserve_order=True)
        joint_targets[:, mast_ids] = robot.data.default_joint_pos[:, mast_ids]
        robot.set_joint_position_target(joint_targets)
        drive.apply(linear_command, angular_command)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        elapsed += sim_dt

        if recorder is not None and recorder.is_recording:
            recorder.capture(
                sim_time=elapsed,
                robot=robot,
                base_robot=rover,
                joint_targets=joint_targets,
                base_command=(linear_command, angular_command),
                cameras=cameras,
                objects=record_objects,
            )

        controller.report_terminal_once()
        if not controller.is_terminal:
            continue
        attempts += 1
        succeeded = bool(controller.succeeded)
        if succeeded:
            successes += 1
            if recorder is not None:
                recorder.finish(success=True)
        elif recorder is not None:
            if args.keep_failed_recordings:
                recorder.finish(success=False)
            else:
                recorder.discard()
        records.append(
            {
                "attempt": attempts,
                "success": succeeded,
                "failure_reason": controller.failure_reason,
                "sample": current_sample,
                "prompt": current_prompt,
                "prompt_index": current_prompt_index,
                "prompt_source": current_prompt_source,
            }
        )
        _write_summary(
            output_dir,
            args,
            domain.name,
            attempts,
            successes,
            target_attempts,
            records,
        )
        print(
            f"[EXROMA][METRICS]: attempts={attempts}/{target_attempts}, "
            f"successes={successes}, success_rate={successes / attempts:.3f}",
            flush=True,
        )
        if attempts < target_attempts:
            reset_attempt()

    if recorder is not None and recorder.is_recording:
        if args.keep_failed_recordings:
            recorder.finish(success=False)
        else:
            recorder.discard()
    print(
        f"[EXROMA][DONE]: successes={successes}, attempts={attempts}, "
        f"success_rate={successes / attempts if attempts else 0.0:.3f}",
        flush=True,
    )
    return 0


def _run_remote_policy_evaluation(
    *,
    args,
    simulation_app,
    sim,
    scene,
    rover,
    robot,
    drive,
    joint_targets,
    initial_state,
    controller,
    cameras,
    record_objects,
    rng,
    prompt_rng,
    target_episodes: int,
    output_dir: Path,
    domain_name: str,
) -> int:
    """Evaluate a remote vision policy while Isaac Sim owns all robot I/O."""

    import time
    from collections import deque

    import torch

    from exroma_bench.policy.client import RemotePolicyClient
    from exroma_bench.policy.sim_codec import DualPiperPolicyCodec, PolicyTaskMonitor
    from exroma_bench.recording import MobileAlohaEpisodeRecorder
    from exroma_bench.tasks.benchmark_suite import sample_task_prompt

    sim_dt = sim.get_physics_dt()
    control_steps = max(1, round(1.0 / (args.policy_frequency * sim_dt)))
    max_steps = args.max_steps or max(1, round(120.0 / sim_dt))
    progress_interval_steps = max(1, round(1.0 / sim_dt))
    max_episode_time = max_steps * sim_dt
    codec = DualPiperPolicyCodec(robot, rover)
    monitor = PolicyTaskMonitor(controller, args.task, max_steps=max_steps)
    records: list[dict[str, object]] = []
    successes = 0
    attempts = 0
    all_env_ids = torch.arange(scene.num_envs, device=scene.device, dtype=torch.int32)
    recorder = None
    if args.record_policy_video:
        recorder = MobileAlohaEpisodeRecorder(
            output_dir,
            fps=args.record_fps,
            jpeg_quality=args.jpeg_quality,
            task_instruction=BENCHMARK_TASKS[args.task].instruction,
            format_name=f"exroma.remote_policy_replay.{args.task}.v1",
            joint_encoding=MobileAlohaEpisodeRecorder.DUAL_PIPER_COMPACT_ENCODING,
            export_video=True,
            keep_hdf5=not args.policy_video_only,
        )
    record_cameras = {
        "mast": cameras["cam_high"],
        "front_left": cameras["cam_left_wrist"],
        "front_right": cameras["cam_right_wrist"],
    }

    print(
        f"[EXROMA][POLICY]: connecting to ws://{args.policy_host}:{args.policy_port}; "
        f"state=16, action=16, cameras=3, frequency={1.0 / (control_steps * sim_dt):.2f} Hz",
        flush=True,
    )
    with RemotePolicyClient(
        args.policy_host,
        args.policy_port,
        connect_timeout=args.policy_connect_timeout,
        response_timeout=args.policy_response_timeout,
        jpeg_quality=args.policy_jpeg_quality,
    ) as client:
        print(
            f"[EXROMA][POLICY]: connected policy={client.metadata['policy_name']!r}",
            flush=True,
        )
        while simulation_app.is_running() and attempts < target_episodes:
            scene.reset_to(initial_state, env_ids=all_env_ids, is_relative=False)
            scene.reset()
            joint_targets.copy_(robot.data.default_joint_pos)
            controller.joint_targets = joint_targets
            robot.set_joint_position_target(joint_targets)
            drive.stop()
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            sample = controller.randomize_and_start(rng)
            if args.task in ROBOTTWIN_TASKS:
                prompt, prompt_index, prompt_source = sample_task_prompt(args.task, prompt_rng)
            else:
                prompt = BENCHMARK_TASKS[args.task].instruction
                prompt_index = 0
                prompt_source = "exroma_builtin"
            robot.set_joint_position_target(joint_targets)
            drive.stop()
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            client.reset()
            monitor.reset()
            record_this_episode = recorder is not None and (
                args.record_policy_video_limit is None
                or attempts < args.record_policy_video_limit
            )
            if record_this_episode:
                recorder.start(
                    metadata={
                        "benchmark": "ExRoMa-Bench",
                        "mode": "remote_policy_evaluation",
                        "task": args.task,
                        "scene": args.scene,
                        "seed": args.seed,
                        "sampled_object_pose": sample,
                        "task_prompt": {
                            "text": prompt,
                            "index": prompt_index,
                            "source": prompt_source,
                        },
                        "policy_server": f"ws://{args.policy_host}:{args.policy_port}",
                    },
                    joint_names=list(robot.joint_names),
                    camera_names=list(record_cameras),
                    task_instruction=prompt,
                )
            action_queue = deque()
            episode_time = 0.0
            query_count = 0
            round_trip_ms: list[float] = []
            server_infer_ms: list[float] = []
            policy_finished = False
            episode_number = attempts + 1
            next_progress_step = progress_interval_steps
            print(
                f"[EXROMA][POLICY][PROGRESS]: episode={episode_number}/{target_episodes}, "
                f"step=0/{max_steps}, elapsed=0.0/{max_episode_time:.1f}s, status=RUNNING",
                flush=True,
            )

            while simulation_app.is_running() and not monitor.is_terminal:
                if not action_queue:
                    state = codec.state()
                    images = codec.images(cameras)
                    query_start = time.perf_counter()
                    actions, timing = client.infer(
                        state=state,
                        images=images,
                        prompt=prompt,
                        timestamp=episode_time,
                    )
                    round_trip_ms.append((time.perf_counter() - query_start) * 1000.0)
                    if "infer_ms" in timing:
                        server_infer_ms.append(float(timing["infer_ms"]))
                    policy_status = timing.get("policy_status", {})
                    policy_finished = bool(policy_status.get("finished", False))
                    action_queue.extend(actions[: args.policy_action_horizon])
                    query_count += 1

                action = action_queue.popleft()
                linear_command, angular_command = codec.apply_action(action, joint_targets)
                for _ in range(control_steps):
                    robot.set_joint_position_target(joint_targets)
                    drive.apply(linear_command, angular_command)
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(sim_dt)
                    episode_time += sim_dt
                    if recorder is not None and recorder.is_recording:
                        recorder.capture(
                            sim_time=episode_time,
                            robot=robot,
                            base_robot=rover,
                            joint_targets=joint_targets,
                            base_command=(linear_command, angular_command),
                            cameras=record_cameras,
                            objects=record_objects,
                        )
                        if (
                            args.policy_video_max_seconds is not None
                            and episode_time >= args.policy_video_max_seconds
                        ):
                            recorder.finish(success=False)
                    monitor.update(sim_dt)
                    if (
                        not monitor.is_terminal
                        and monitor.steps >= next_progress_step
                    ):
                        print(
                            f"[EXROMA][POLICY][PROGRESS]: "
                            f"episode={episode_number}/{target_episodes}, "
                            f"step={monitor.steps}/{max_steps}, "
                            f"elapsed={monitor.elapsed:.1f}/{max_episode_time:.1f}s, "
                            "status=RUNNING",
                            flush=True,
                        )
                        next_progress_step += progress_interval_steps
                    if monitor.is_terminal or not simulation_app.is_running():
                        break
                if policy_finished and not action_queue and not monitor.is_terminal:
                    monitor.failure_reason = "policy action trajectory exhausted"

            if not simulation_app.is_running() and not monitor.is_terminal:
                break
            drive.stop()
            attempts += 1
            succeeded = bool(monitor.succeeded)
            successes += int(succeeded)
            failure_reason = "" if succeeded else monitor.failure_reason
            result_status = "SUCCESS" if succeeded else "FAILURE"
            result_reason = failure_reason or "success criteria satisfied"
            print(
                f"[EXROMA][POLICY][RESULT]: episode={attempts}/{target_episodes}, "
                f"status={result_status}, step={monitor.steps}/{max_steps}, "
                f"elapsed={monitor.elapsed:.1f}/{max_episode_time:.1f}s, "
                f"reason={result_reason}",
                flush=True,
            )
            diagnostics_fn = getattr(controller, "success_diagnostics", None)
            success_diagnostics = diagnostics_fn() if callable(diagnostics_fn) else None
            if recorder is not None and recorder.is_recording:
                recorder.finish(success=succeeded)
            record = {
                "attempt": attempts,
                "success": succeeded,
                "failure_reason": failure_reason,
                "sample": sample,
                "prompt": prompt,
                "prompt_index": prompt_index,
                "prompt_source": prompt_source,
                "policy_queries": query_count,
                "episode_steps": monitor.steps,
                "episode_time_s": monitor.elapsed,
                "mean_round_trip_ms": (
                    sum(round_trip_ms) / len(round_trip_ms) if round_trip_ms else None
                ),
                "mean_server_infer_ms": (
                    sum(server_infer_ms) / len(server_infer_ms) if server_infer_ms else None
                ),
            }
            if success_diagnostics is not None:
                record["success_diagnostics"] = success_diagnostics
                print(
                    "[EXROMA][POLICY][SUCCESS-DIAGNOSTICS]: "
                    + ", ".join(f"{key}={value}" for key, value in success_diagnostics.items()),
                    flush=True,
                )
            records.append(record)
            _write_summary(
                output_dir,
                args,
                domain_name,
                attempts,
                successes,
                target_episodes,
                records,
            )
            print(
                f"[EXROMA][POLICY][METRICS]: episodes={attempts}/{target_episodes}, "
                f"successes={successes}, success_rate={successes / attempts:.3f}",
                flush=True,
            )

    if recorder is not None and recorder.is_recording:
        recorder.finish(success=False)
    drive.stop()
    print(
        f"[EXROMA][POLICY][DONE]: successes={successes}, episodes={attempts}, "
        f"success_rate={successes / attempts if attempts else 0.0:.3f}",
        flush=True,
    )
    return 0


def _create_controller(
    args,
    robot,
    joint_targets,
    scene,
    create_benchmark_controller,
    benchmark_record_objects,
    tube_cfg_type,
    tube_controller_type,
    hammer_cfg_type,
    hammer_controller_type,
    table_x,
    table_y,
    table_height,
    table_length,
    table_width,
):
    if args.task in ROBOTTWIN_TASKS:
        controller = create_benchmark_controller(
            robot,
            joint_targets,
            scene,
            args.task,
            table_x=table_x,
            table_y=table_y,
            table_height=table_height,
            table_length=table_length,
            table_width=table_width,
            strict_collision_check=args.strict_collision_check,
            planner_enabled=not bool(args.policy_host),
        )
        return controller, benchmark_record_objects(scene, args.task)
    if args.task == "test_tube_rack":
        target = scene["test_tube"]
        cfg = tube_cfg_type(
            arm="fl",
            table_x=table_x,
            table_y=table_y,
            table_height=table_height,
            table_length=table_length,
            table_width=table_width,
            target_x=table_x + 0.18,
            target_y=table_y,
            spawn_x_range=(-0.08, 0.10),
            spawn_y_range=(-0.05, 0.12),
            robot_config_stem="dual_piper",
            align_base=False,
            planner_self_collision_check=args.strict_collision_check,
            planner_enabled=not bool(args.policy_host),
        )
        return tube_controller_type(robot, joint_targets, target, cfg), {"test_tube": target}
    hammer = scene["robotwin_hammer"]
    block = scene["hammer_block"]
    cfg = hammer_cfg_type(
        arm="auto",
        table_x=table_x,
        table_y=table_y,
        table_height=table_height,
        table_length=table_length,
        table_width=table_width,
        planner_self_collision_check=args.strict_collision_check,
        planner_enabled=not bool(args.policy_host),
    )
    return (
        hammer_controller_type(
            robot,
            joint_targets,
            hammer,
            block,
            scene["hammer_contact_sensor"],
            cfg,
        ),
        {"robotwin_hammer": hammer, "hammer_block": block},
    )


def _relax_base_settle_gates(controller) -> None:
    configs = []
    for step in getattr(controller, "steps", ()):
        cfg = getattr(step, "cfg", None)
        if cfg is not None:
            configs.append(cfg)
    for worker in getattr(controller, "workers", {}).values():
        cfg = getattr(worker, "cfg", None)
        if cfg is not None:
            configs.append(cfg)
    for attr_name in ("object_worker",):
        worker = getattr(controller, attr_name, None)
        cfg = getattr(worker, "cfg", None)
        if cfg is not None:
            configs.append(cfg)
    for cfg in configs:
        if hasattr(cfg, "base_settle_linear_velocity"):
            cfg.base_settle_linear_velocity = max(cfg.base_settle_linear_velocity, 10.0)
        if hasattr(cfg, "base_settle_angular_velocity"):
            cfg.base_settle_angular_velocity = max(cfg.base_settle_angular_velocity, 10.0)
        if hasattr(cfg, "base_stable_steps"):
            cfg.base_stable_steps = 0
        if hasattr(cfg, "settle_time"):
            cfg.settle_time = 0.0
        if hasattr(cfg, "base_settle_timeout"):
            cfg.base_settle_timeout = max(cfg.base_settle_timeout, 30.0)
        if hasattr(cfg, "max_execution_base_translation"):
            cfg.max_execution_base_translation = max(cfg.max_execution_base_translation, 1.0)
        if hasattr(cfg, "max_execution_base_rotation"):
            cfg.max_execution_base_rotation = max(cfg.max_execution_base_rotation, 3.14159)


def _write_summary(
    output_dir: Path,
    args,
    domain: str,
    attempts: int,
    successes: int,
    target_attempts: int,
    records: list[dict[str, object]],
) -> None:
    failures = Counter(
        record["failure_reason"] or "unspecified" for record in records if not record["success"]
    )
    payload = {
        "benchmark": "ExRoMa-Bench",
        "mode": args.command,
        "task": args.task,
        "scene": args.scene,
        "domain": domain,
        "seed": args.seed,
        "terrain": {
            "source": (
                "simforge_foundry"
                if args.scene in {"procedural_moon", "procedural_mars"}
                else "external_asset"
            ),
            "seed": args.terrain_seed,
            "size_m": args.terrain_size,
        },
        "strict_curobo_collision_check": bool(args.strict_collision_check),
        "target_attempts": target_attempts,
        "attempts": attempts,
        "successes": successes,
        "failures": attempts - successes,
        "success_rate": successes / attempts if attempts else 0.0,
        "failure_counts": dict(failures),
        "complete": attempts >= target_attempts,
        "records": records,
    }
    if getattr(args, "policy_host", None):
        payload["policy_interface"] = {
            "protocol": "exroma.policy.v1",
            "server": f"ws://{args.policy_host}:{args.policy_port}",
            "state_dim": 16,
            "action_dim": 16,
            "cameras": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            "mast_fixed": True,
            "frequency_hz": args.policy_frequency,
            "action_horizon": args.policy_action_horizon,
        }
    destination = output_dir / "collection_summary.json"
    temporary = output_dir / ".collection_summary.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(destination)


def _hide_pragyan_solar_panel(stage, Usd, UsdGeom, UsdPhysics) -> None:
    root = stage.GetPrimAtPath("/World/envs/env_0/rover")
    if not root.IsValid():
        root = stage.GetPrimAtPath("/World/envs/env_0/robot")
    for prim in Usd.PrimRange(root):
        if "solar_panel" not in prim.GetPath().pathString.lower():
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeInvisible()
        collision = UsdPhysics.CollisionAPI(prim)
        if collision:
            collision.GetCollisionEnabledAttr().Set(False)


def _write_articulation_defaults(*, rover, robot) -> None:
    """Make Isaac Lab config defaults the canonical episode initial state."""

    for articulation in (rover, robot):
        root_state = articulation.data.default_root_state.clone()
        articulation.write_root_pose_to_sim(root_state[:, :7])
        articulation.write_root_velocity_to_sim(root_state[:, 7:])
        articulation.write_joint_state_to_sim(
            articulation.data.default_joint_pos,
            articulation.data.default_joint_vel,
        )


if __name__ == "__main__":
    raise SystemExit(main())
