"""Learning exercises that remain importable without local GPU libraries."""


def main() -> None:
    """CPU lesson configurations, grid baselines, and dynamics replay."""
    import argparse
    import json

    from .workspace import config_dict

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", type=int, choices=tuple(range(1, 17)), default=1)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--train-layouts", type=int, default=8)
    parser.add_argument("--fresh-cases", type=int, default=8)
    parser.add_argument("--probe-cases", type=int, default=1)
    parser.add_argument("--sft-run-id", default="")
    parser.add_argument("--dataset-run-id", default="")
    parser.add_argument("--warmup-run-id", default="")
    parser.add_argument("--comparison-run-id", default="")
    parser.add_argument("--readiness-run-id", default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--replay", metavar="RUN_DIRECTORY")
    args = parser.parse_args()
    if args.replay and args.stage not in set(range(3, 17)):
        parser.error("--replay requires --stage 3 through 16")
    if args.stage == 16:
        from dataclasses import asdict
        from pathlib import Path
        from .stage16 import FormGRPOConfig, source_receipts, plan, replay_saved_run
        if args.replay:
            print(f'Verified form GRPO bundle ({replay_saved_run(Path(args.replay))} rounds or episodes)')
        else:
            config = FormGRPOConfig(seed=args.seed, sft_run_id=args.sft_run_id,
                comparison_run_id=args.comparison_run_id, readiness_run_id=args.readiness_run_id)
            print(json.dumps({'config': asdict(config), 'receipts': source_receipts(config, Path(args.output_dir)),
                              'plan': plan(config)}, indent=2))
        return
    if args.stage == 15:
        from dataclasses import asdict
        from pathlib import Path
        from .stage15 import ExplorationConfig, source_receipts, study_plan, replay_saved_run
        if args.replay:
            print(f'Verified {replay_saved_run(Path(args.replay))} training-only exploration episodes')
        else:
            config = ExplorationConfig(seed=args.seed, sft_run_id=args.sft_run_id, warmup_run_id=args.warmup_run_id)
            print(json.dumps({'config': asdict(config), 'sources': source_receipts(config, Path(args.output_dir)),
                              'plan': study_plan(config)}, indent=2))
        return
    if args.stage==14:
        from dataclasses import asdict
        from pathlib import Path
        from .stage14 import FormSFTConfig, source_receipts, replay_training, replay_evaluation, schedule
        if args.replay:
            directory=Path(args.replay);kind=json.loads((directory/'manifest.json').read_text()).get('kind')
            if kind=='form-sft-checkpoint': print(f'Verified {replay_training(directory)} cumulative SFT updates')
            else: print(f'Verified {replay_evaluation(directory)} development episodes')
        else:
            config=FormSFTConfig(seed=args.seed,epochs=args.epochs,dataset_run_id=args.dataset_run_id,sft_run_id=args.sft_run_id)
            receipt=source_receipts(config,Path(args.output_dir))
            print(json.dumps({'config':asdict(config),'receipts':receipt,
                              'total_updates':len(schedule(config,receipt['dataset']['targets'])),
                              'note':'Configuration only. No final-test instances are opened.'},indent=2))
        return
    if args.stage == 13:
        from dataclasses import asdict
        from pathlib import Path
        from .stage13 import FormDatasetConfig, pools, replay_saved_run
        if args.replay:
            print(f"Verified {replay_saved_run(Path(args.replay))} form demonstrations; no training performed")
        else:
            config=FormDatasetConfig(seed=args.seed)
            print(json.dumps({'config':asdict(config),'pools':pools(config),
                              'note':'Configuration only. Dev/final instances remain unopened.'},indent=2))
        return
    if args.stage == 12:
        from dataclasses import asdict
        from pathlib import Path
        from .stage10 import latest_local_sft
        from .stage11 import source_receipt
        from .stage12 import BrowserGRPOConfig, build_cases, pilot_config, replay_saved_run
        if args.replay:
            print(f"Verified {replay_saved_run(Path(args.replay))} browser GRPO/evaluation episodes")
        else:
            source_id=args.sft_run_id or latest_local_sft(Path(args.output_dir))
            config=BrowserGRPOConfig(seed=args.seed,episodes=args.episodes,sft_run_id=source_id)
            print(json.dumps({"config":asdict(config),"source":source_receipt(pilot_config(config),Path(args.output_dir)),
                              "cases":build_cases(config),"note":"Configuration only; gradient checks and updates run remotely."},indent=2))
        return
    if args.stage == 11:
        from dataclasses import asdict
        from pathlib import Path
        from .stage10 import latest_local_sft
        from .stage11 import BrowserDifficultyConfig, build_cases, reserved_eval, replay_saved_run, source_receipt
        if args.replay:
            print(f"Verified {replay_saved_run(Path(args.replay))} harder-task training episodes")
        else:
            source_id = args.sft_run_id or latest_local_sft(Path(args.output_dir))
            config = BrowserDifficultyConfig(seed=args.seed, episodes=args.episodes, sft_run_id=source_id)
            print(json.dumps({"config":asdict(config),"source":source_receipt(config,Path(args.output_dir)),
                              "training_cases":build_cases(config),"reserved_eval":reserved_eval(config),
                              "optimizer_steps":0,"note":"Configuration checks only; real browsers and GPU run remotely."},indent=2))
        return
    if args.stage == 10:
        from dataclasses import asdict
        from pathlib import Path
        from .stage10 import BrowserReadinessConfig, build_cases, latest_local_sft, replay_saved_run, source_receipt
        if args.replay:
            print(f"Verified {replay_saved_run(Path(args.replay))} browser evaluation/probe episodes")
        else:
            source_id = args.sft_run_id or latest_local_sft(Path(args.output_dir))
            config = BrowserReadinessConfig(seed=args.seed, episodes=args.episodes, sft_run_id=source_id,
                                            fresh_cases=args.fresh_cases, probe_cases=args.probe_cases)
            case_list, pools = build_cases(config)
            print(json.dumps({"config": asdict(config), "source": source_receipt(config, Path(args.output_dir)),
                              "cases": case_list, "pools": pools, "optimizer_steps": 0,
                              "note": "Configuration/source checks only; browsers and models run remotely."}, indent=2))
        return
    if args.stage == 9:
        from dataclasses import asdict
        from pathlib import Path
        from .stage9 import BrowserSFTConfig, latest_local_source, replay_saved_run, validate_source
        if args.replay:
            print(f"Verified {replay_saved_run(Path(args.replay))} browser SFT trajectories and assistant token masks")
        else:
            config = BrowserSFTConfig(seed=args.seed, episodes=args.episodes)
            source = latest_local_source(Path(args.output_dir), config)
            receipt = validate_source(config, Path(args.output_dir)/source)
            print(json.dumps({"config": asdict(config), "cpu_source": receipt,
                              "note": "Source and split checks only; model training runs remotely."}, indent=2))
        return
    if args.stage == 8:
        from dataclasses import asdict
        from .browser import BrowserConfig, cases
        if args.replay:
            from pathlib import Path
            from .stage8 import replay_saved_run
            print(f"Verified {replay_saved_run(Path(args.replay))} saved browser trajectories (no live browser re-execution)")
        else:
            config = BrowserConfig(seed=args.seed, episodes=args.episodes)
            print(json.dumps({"config": asdict(config), "cases": cases(config),
                              "note": "Configuration only. Browser and model execution run remotely."}, indent=2))
        return
    if args.stage == 7:
        from .stage7 import dataset_row, replay_saved_run
        from .gridworld import make_layouts, render_svg, shortest_path
        from .trajectories import move_call
        from .trl_environment import GridToolEnvironment, dispatch_response
        if args.replay:
            from pathlib import Path
            print(f"Verified {replay_saved_run(Path(args.replay))} TRL/evaluation trajectories")
        else:
            episodes = []
            env = GridToolEnvironment(renderer=render_svg, assets={}, episodes=episodes,
                                      image_owners={}, next_key=lambda: "cpu-example", seed=args.seed)
            layout = make_layouts("train", 2)[1]
            env.reset(**dataset_row(layout))
            while not (env._env.terminated or env._env.truncated):
                text = move_call(shortest_path(layout, env._env.position)[0])
                env._set_decision(text, {"generated_ids": None, "token_source": "solver_text"})
                env.move(**dispatch_response(text)["tool_calls"][0]["function"]["arguments"])
            print(json.dumps({"return": env.get_reward(), "steps": episodes[0]["steps"],
                              "observations": len(episodes[0]["observations"]),
                              "replayed": True, "note": "CPU wrapper exercise only; native TRL checks run remotely."}, indent=2))
        return
    if args.stage == 6:
        from .reporting import wilson_interval
        if args.replay:
            from pathlib import Path
            from .stage6 import replay_saved_run
            print(f"Verified {replay_saved_run(Path(args.replay))} trajectories and their paired evaluation")
        else:
            print(json.dumps({"successes": 1, "boards": 4, "success_rate": .25,
                              "wilson_95": wilson_interval(1, 4),
                              "checkpoint_contents": ["adapter", "Adam state", "parameter order", "round", "RNG", "history"]}, indent=2))
        return
    if args.stage == 5:
        if args.replay:
            from pathlib import Path
            from .stage5 import replay_saved_run

            print(f"Verified {replay_saved_run(Path(args.replay))} trajectories and their GRPO groups")
        else:
            from .grpo import synthetic_checks

            print(json.dumps(synthetic_checks(), indent=2))
        return
    if args.stage == 4:
        from pathlib import Path
        import platform
        from .stage4 import Stage4Config, build_demonstrations, replay_saved_run, save_report
        from .workspace import new_run_id, run_path, source_hashes

        if args.replay:
            print(f"Verified {replay_saved_run(Path(args.replay))} complete multi-turn trajectories")
            return
        try:
            config = Stage4Config(seed=args.seed, episodes=args.episodes, train_layouts=args.train_layouts)
        except ValueError as exc:
            parser.error(str(exc))
        assets = {}
        data = build_demonstrations(config, assets)
        directory = run_path(Path(args.output_dir), new_run_id(stage=4))
        environment = {"mode": "cpu", "python": platform.python_version(), "gpu": None,
                       "images": "SVG inspection only; GPU job renders PNG observations"}
        save_report(directory, config, data, environment, source_hashes(Path(__file__).resolve().parents[2]), assets)
        count = replay_saved_run(directory)
        print(json.dumps({"run_directory": str(directory), "replayed_demonstrations": count,
                          "action_examples": len(data["examples"]), "metrics": data["metrics"]}, indent=2))
        return
    if args.stage == 3:
        from pathlib import Path
        import platform
        from .stage3 import Stage3Config, evaluate, replay_saved_run, save_report
        from .workspace import new_run_id, run_path, source_hashes

        if args.replay:
            count = replay_saved_run(Path(args.replay))
            print(f"Verified {count} episodes by replaying their dynamics")
            return
        try:
            config = Stage3Config(seed=args.seed, episodes=args.episodes)
        except ValueError as exc:
            parser.error(str(exc))
        data = evaluate(config)
        directory = run_path(Path(args.output_dir), new_run_id(stage=3))
        environment = {"mode": "cpu", "python": platform.python_version(), "gpu": None}
        save_report(directory, config, data, environment, source_hashes(Path(__file__).resolve().parents[2]))
        print(json.dumps({"run_directory": str(directory), "metrics": data["metrics"],
                          "verification": data["verification"]}, indent=2))
        return
    try:
        if args.stage == 1:
            config = config_dict(args.seed)
        else:
            from dataclasses import asdict
            from .vision import Stage2Config

            config = asdict(Stage2Config(seed=args.seed))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(config, indent=2))
