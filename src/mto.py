<<<<<<< HEAD
import os
import pprint
import time
import threading
import torch as th
from tqdm.auto import tqdm
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import copy
import json
import shutil

from learners.multi_task import REGISTRY as le_REGISTRY
from runners.multi_task import REGISTRY as r_REGISTRY
from controllers.multi_task import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.offline_buffer import OfflineBuffer
from components.transforms import OneHot

import numpy as np


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    results_save_dir = args.results_save_dir

    if args.use_tensorboard and not args.evaluate:
        # only log tensorboard when in training mode
        # though we are always in training mode when we reach here
        tb_exp_direc = os.path.join(results_save_dir, "tb_logs")
        logger.setup_tb(tb_exp_direc)

    # set model save dir
    args.save_dir = os.path.join(results_save_dir, "models")

    # write config file
    config_str = json.dumps(vars(args), indent=4)
    with open(os.path.join(results_save_dir, "config.json"), "w") as f:
        f.write(config_str)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(main_args, logger, task2runner):
    n_test_runs = max(1, main_args.test_nepisode // main_args.batch_size_run)
    with th.no_grad():
        for task in main_args.test_tasks:
            for _ in range(n_test_runs):
                task2runner[task].run(test_mode=True)

            if main_args.save_replay:
                task2runner[task].save_replay()

            task2runner[task].close_env()

    logger.log_stat("episode", 0, 0)
    logger.print_recent_stats()


def init_tasks(task_list, main_args, logger):
    task2args, task2runner, task2buffer = {}, {}, {}
    task2scheme, task2groups, task2preprocess = {}, {}, {}

    for task in task_list:
        # define task_args
        task_args = copy.deepcopy(main_args)
        if task_args.env == "sc2":
            task_args.env_args["map_name"] = task
        elif task_args.env == "gymma":
            task_args.env_args["N"] = task
        task2args[task] = task_args

        task_runner = r_REGISTRY[main_args.runner](
            args=task_args, logger=logger, task=task
        )
        task2runner[task] = task_runner

        # Set up schemes and groups here
        env_info = task_runner.get_env_info()
        for k, v in env_info.items():
            setattr(task_args, k, v)

        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            # "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        # For individual rewards in gymmai reward is of shape (1, n_agents)
        if main_args.env == "sc2":
            scheme["reward"] = {"vshape": (1,)}
        elif main_args.common_reward:
            scheme["reward"] = {"vshape": (1,)}
        else:
            scheme["reward"] = {"vshape": (main_args.n_agents,)}
        groups = {"agents": task_args.n_agents}
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=task_args.n_actions)])
        }

        task2buffer[task] = ReplayBuffer(
            scheme,
            groups,
            1,
            env_info["episode_limit"] + 1,
            preprocess=preprocess,
            device="cpu" if task_args.buffer_cpu_only else task_args.device,
        )

        # store task information
        task2scheme[task], task2groups[task], task2preprocess[task] = (
            scheme,
            groups,
            preprocess,
        )

    return (
        task2args,
        task2runner,
        task2buffer,
        task2scheme,
        task2groups,
        task2preprocess,
    )



# ======================== HISSD_RESUME_V3 ========================

def _hissd_resume_signature(main_args):
    keys = (
        "task",
        "seed",
        "agent",
        "mac",
        "mixer",
        "optim_type",
        "lr",
        "critic_lr",
        "gamma",
        "beta",
        "td_weight",
        "coef_conservative",
        "coef_dist",
        "coef_kl",
        "epsilon",
        "batch_size",
        "weight_decay",
        "grad_norm_clip",
        "target_update_interval",
        "double_q",
        "entity_embed_dim",
        "attn_embed_dim",
        "skill_dim",
        "common_skill_adapter_enabled",
        "common_skill_adapter_hidden_dim",
        "mixing_embed_dim",
        "hypernet_layers",
        "hypernet_embed",
        "rnn_hidden_dim",
        "c_step",
        "ssl_type",
        "ssl_time_window",
        "double_neg",
        "head",
        "depth",
        "num_stack_frames",
        "vq_skill",
        "code_dim",
        "offline_data_size",
        "offline_data_shuffle",
        "offline_data_name",
    )

    sig = {
        key: getattr(main_args, key, None)
        for key in keys
    }

    # Task order itself changes because np.random.shuffle() is in-place.
    # Signature checks task membership; exact current ordering is saved
    # separately as training state.
    sig["train_task_set"] = tuple(
        sorted(getattr(main_args, "train_tasks", []))
    )

    qualities = getattr(
        main_args,
        "train_tasks_data_quality",
        {},
    )

    sig["train_tasks_data_quality"] = tuple(
        sorted(qualities.items())
    )

    return sig


def _hissd_capture_rng():
    import random

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": th.get_rng_state(),
    }

    if th.cuda.is_available():
        state["cuda"] = th.cuda.get_rng_state_all()

    return state


def _hissd_restore_rng(state):
    import random

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    # torch.set_rng_state() requires a CPU ByteTensor.
    # resume.th itself may be loaded with map_location="cuda",
    # so explicitly move RNG states back to CPU.
    torch_rng = state["torch"]
    if not isinstance(torch_rng, th.Tensor):
        torch_rng = th.as_tensor(torch_rng)
    torch_rng = torch_rng.detach().cpu().to(dtype=th.uint8)
    th.set_rng_state(torch_rng)

    if th.cuda.is_available() and "cuda" in state:
        cuda_rng = []
        for rng_state in state["cuda"]:
            if not isinstance(rng_state, th.Tensor):
                rng_state = th.as_tensor(rng_state)
            cuda_rng.append(
                rng_state.detach().cpu().to(dtype=th.uint8)
            )
        th.cuda.set_rng_state_all(cuda_rng)


def _hissd_save_resume_state(
    save_path,
    learner,
    main_args,
    t_env,
    episode,
    train_tasks,
    last_test_T,
    last_log_T,
):
    resume_path = os.path.join(
        save_path,
        "resume.th",
    )

    tmp_path = resume_path + ".tmp"

    payload = {
        "version": 3,

        # Outer training loop
        "t_env": int(t_env),
        "episode": int(episode),
        "model_save_time": int(t_env),
        "last_test_T": int(last_test_T),
        "last_log_T": int(last_log_T),

        # np.random.shuffle mutates this exact list.
        "train_tasks": list(train_tasks),

        # Learner counters
        "training_steps": int(
            getattr(learner, "training_steps", 0)
        ),
        "pretrain_steps": int(
            getattr(learner, "pretrain_steps", 0)
        ),
        "last_target_update_episode": int(
            getattr(
                learner,
                "last_target_update_episode",
                0,
            )
        ),

        # HiSSD's MoCo cross-task negative cache
        "last_batch": getattr(
            learner,
            "last_batch",
            {},
        ),
        "last_task": getattr(
            learner,
            "last_task",
            "",
        ),

        # Learner logging scheduler
        "task2train_info": getattr(
            learner,
            "task2train_info",
            {},
        ),

        # Both optimizer states
        "optimiser": learner.optimiser.state_dict(),
        "pre_optimiser": (
            learner.pre_optimiser.state_dict()
        ),

        # Original HiSSD save_models() does not preserve
        # the exact lagged target network.
        "target_agent": (
            learner.target_mac.agent.state_dict()
        ),

        "target_mixer": (
            learner.target_mixer.state_dict()
            if getattr(
                learner,
                "target_mixer",
                None,
            ) is not None
            else None
        ),

        # Exact stochastic stream continuation
        "rng": _hissd_capture_rng(),

        # Protect against e.g. AdamW/3e-4 -> Adam/1e-4 resume
        "config_signature": (
            _hissd_resume_signature(main_args)
        ),
    }

    th.save(payload, tmp_path)
    os.replace(tmp_path, resume_path)


def _hissd_load_resume_state(
    resume_path,
    learner,
    main_args,
):
    payload = th.load(
        resume_path,
        map_location=getattr(
            main_args,
            "device",
            "cpu",
        ),
        weights_only=False,
    )

    if int(payload.get("version", -1)) != 3:
        raise RuntimeError(
            "Unsupported HiSSD resume version: "
            "{}".format(payload.get("version"))
        )

    saved_sig = payload.get(
        "config_signature",
        {},
    )

    current_sig = _hissd_resume_signature(
        main_args
    )

    mismatches = []

    for key, saved_value in saved_sig.items():
        current_value = current_sig.get(key)

        if saved_value != current_value:
            mismatches.append(
                "{}: checkpoint={!r}, current={!r}".format(
                    key,
                    saved_value,
                    current_value,
                )
            )

    if mismatches:
        raise RuntimeError(
            "HiSSD resume config mismatch:\n  "
            + "\n  ".join(mismatches)
        )

    # Current agent/mixer and optimiser were already loaded
    # by the repository's existing learner.load_models().
    # Replace optimizer state with our exact saved state.
    learner.optimiser.load_state_dict(
        payload["optimiser"]
    )

    learner.pre_optimiser.load_state_dict(
        payload["pre_optimiser"]
    )

    # Restore lagged target network exactly.
    learner.target_mac.agent.load_state_dict(
        payload["target_agent"]
    )

    if (
        payload.get("target_mixer") is not None
        and getattr(
            learner,
            "target_mixer",
            None,
        ) is not None
    ):
        learner.target_mixer.load_state_dict(
            payload["target_mixer"]
        )

    learner.training_steps = int(
        payload["training_steps"]
    )

    learner.pretrain_steps = int(
        payload["pretrain_steps"]
    )

    learner.last_target_update_episode = int(
        payload["last_target_update_episode"]
    )

    learner.last_batch = payload.get(
        "last_batch",
        {},
    )

    learner.last_task = payload.get(
        "last_task",
        "",
    )

    learner.task2train_info = payload.get(
        "task2train_info",
        learner.task2train_info,
    )

    # Fresh train() normally sets learner.device when
    # training_steps == 0. Resume skips that branch.
    if learner.last_batch:
        first_batch = next(
            iter(learner.last_batch.values())
        )
        learner.device = first_batch.device
    else:
        learner.device = getattr(
            main_args,
            "device",
            None,
        )

    return payload

# ====================== /HISSD_RESUME_V3 =========================


def train_sequential(
    train_tasks,
    main_args,
    logger,
    learner,
    task2args,
    task2runner,
    task2offlinedata,
    t_start=0,
    pretrain=False,
    test_task2offlinedata=None,
):
    ########## start training ##########
    resume_state = getattr(
        main_args,
        "_resume_state",
        None,
    )

    if resume_state is not None:
        t_start = int(resume_state["t_env"])

    t_env = t_start

    episode = (
        int(resume_state["episode"])
        if resume_state is not None
        else 0
    )

    t_max = (
        main_args.t_max
        if not pretrain
        else main_args.pretrain_steps
    )

    model_save_time = (
        int(resume_state["model_save_time"])
        if resume_state is not None
        else 0
    )

    last_test_T = (
        int(resume_state["last_test_T"])
        if resume_state is not None
        else 0
    )

    last_log_T = (
        int(resume_state["last_log_T"])
        if resume_state is not None
        else 0
    )
    start_time = time.time()
    last_time = start_time
    test_time_total = 0
    test_start_time = 0

    # get some common information
    batch_size_train = main_args.batch_size
    batch_size_run = main_args.batch_size_run

    # do test before training
    n_test_runs = max(1, main_args.test_nepisode // batch_size_run)
    if main_args.evaluate:
        n_test_runs = 0
    if main_args.debug:
        n_test_runs = 0

    # HISSD_FAST_EVAL_V1 -- bounded, per-map temporary parallel evaluation.
    # This intentionally does NOT change batch_size_run for training.
    _fast_eval_workers = int(os.environ.get("HISSD_EVAL_WORKERS", "1"))
    if not 1 <= _fast_eval_workers <= 10:
        raise ValueError("HISSD_EVAL_WORKERS must be between 1 and 10")
    _fast_eval_retries = int(os.environ.get("HISSD_EVAL_RETRIES", "2"))
    if not 0 <= _fast_eval_retries <= 5:
        raise ValueError("HISSD_EVAL_RETRIES must be between 0 and 5")

    def _hissd_fast_run_eval_task(task, original_n_runs):
        if original_n_runs <= 0:
            return
        _eval_t0 = time.perf_counter()
        _num_episodes = original_n_runs * batch_size_run
        _worker_count = _fast_eval_workers
        if _num_episodes % _worker_count:
            raise ValueError(
                f"Cannot evaluate {_num_episodes} episodes with "
                f"{_worker_count} workers without changing episode count"
            )

        if _worker_count == 1:
            for _ in range(original_n_runs):
                task2runner[task].run(test_mode=True, pretrain=pretrain)
        else:
            original_runner = task2runner[task]
            if not hasattr(original_runner, "_hissd_fast_eval_setup"):
                raise RuntimeError("Original runner setup for parallel eval is missing")
            _scheme, _groups, _preprocess, _mac = original_runner._hissd_fast_eval_setup
            _parallel_args = copy.deepcopy(task2args[task])
            _parallel_args.batch_size_run = _worker_count
            _parallel_args.test_nepisode = _num_episodes
            _parallel_args.parallel_env_timeout_seconds = float(
                os.environ.get("HISSD_ENV_TIMEOUT_SECONDS", "120")
            )
            for _attempt in range(_fast_eval_retries + 1):
                _parallel_runner = None
                try:
                    _parallel_runner = r_REGISTRY["mt_parallel"](
                        args=_parallel_args, logger=logger, task=task
                    )
                    _parallel_runner.setup(
                        scheme=_scheme,
                        groups=_groups,
                        preprocess=_preprocess,
                        mac=_mac,
                    )
                    _parallel_runner.t_env = t_env
                    for _ in range(_num_episodes // _worker_count):
                        _parallel_runner.run(
                            test_mode=True, pretrain_phase=pretrain
                        )
                    break
                except Exception as _eval_error:
                    if _attempt >= _fast_eval_retries:
                        raise
                    logger.console_logger.warning(
                        "HISSD_EVAL_RETRY task=%s steps=%s attempt=%s/%s error=%s",
                        task,
                        t_env,
                        _attempt + 1,
                        _fast_eval_retries,
                        _eval_error,
                    )
                    time.sleep(1.0)
                finally:
                    if _parallel_runner is not None:
                        try:
                            _parallel_runner.close_env()
                        finally:
                            # Reap only this temporary runner's own worker children.
                            for _p in _parallel_runner.ps:
                                _p.join(timeout=3.0)
                                if _p.is_alive():
                                    _p.terminate()
                                    _p.join(timeout=3.0)
                                if _p.is_alive():
                                    _p.kill()
                                    _p.join(timeout=3.0)
        _eval_elapsed = time.perf_counter() - _eval_t0
        logger.console_logger.info(
            "HISSD_EVAL_TIMING task=%s steps=%s episodes=%s workers=%s seconds=%.3f"
            % (task, t_env, _num_episodes, _worker_count, _eval_elapsed)
        )

    initial_n_test_runs = (
        0
        if resume_state is not None
        else n_test_runs
    )

    pbar = tqdm(
        total=t_max,
        initial=min(t_env, t_max),
        desc="HiSSD training",
        unit="step",
        dynamic_ncols=True,
        mininterval=1.0,
        disable=not getattr(main_args, "progress_bar", True),
    )

    if not getattr(main_args, "smoke_skip_evaluation", False):
        test_start_time = time.time()

        with th.no_grad():
            for task in main_args.test_tasks:
                task2runner[task].t_env = t_env
                _hissd_fast_run_eval_task(task, initial_n_test_runs)

            # test_pretrain for pretrained tasks
            if pretrain and test_task2offlinedata is not None:
                for task, data_buffer in test_task2offlinedata.items():
                    episode_sample = data_buffer.sample(batch_size_train * 3)

                    if episode_sample.device != task2args[task].device:
                        episode_sample.to(task2args[task].device)

                    if hasattr(learner, "test_pretrain"):
                        learner.test_pretrain(episode_sample, t_env, episode, task)
                    else:
                        raise ValueError(
                            "Do test_pretrain with a learner that does not have a `test_pretrain` method!"
                        )

        test_time_total += time.time() - test_start_time

    if resume_state is not None:
        _hissd_restore_rng(
            resume_state["rng"]
        )

        logger.console_logger.info(
            "HiSSD resume RNG restored at "
            "t_env={} training_steps={}".format(
                t_env,
                learner.training_steps,
            )
        )

    # HISSD_TQDM_SUMMARY_V1
    _hissd_source_tasks = list(main_args.train_tasks)
    _hissd_source_set = set(_hissd_source_tasks)
    _hissd_target_tasks = [
        task for task in main_args.test_tasks
        if task not in _hissd_source_set
    ]

    # V3_TQDM_METRICS_PATCH_20261015
    # Keep all fields across training and evaluation postfix refreshes.
    _v3_postfix = {"task": "-"}

    while t_env < t_max:
        # shuffle tasks
        np.random.shuffle(train_tasks)
        # train each task
        for task in train_tasks:

            episode_sample = task2offlinedata[task].sample(batch_size_train)

            if episode_sample.device != task2args[task].device:
                episode_sample.to(task2args[task].device)

            if pretrain:
                if hasattr(learner, "pretrain"):
                    terminated = learner.pretrain(episode_sample, t_env, episode, task)
                else:
                    raise ValueError(
                        "Do pretraining with a learner that does not have a `pretrain` method!"
                    )
            else:
                terminated = learner.train(episode_sample, t_env, episode, task)

            if terminated is not None and terminated:
                break

            t_env += 1
            episode += batch_size_run
            if pbar.n < pbar.total:
                pbar.update(1)
            _v3_postfix["task"] = task
            _v3_cache = getattr(learner, "v3_latest_metrics", None)
            if _v3_cache is not None:
                _v3_postfix.update({
                    "d/c": "%.3f" % _v3_cache["delta_ratio"],
                })
            pbar.set_postfix(_v3_postfix, refresh=False)
            if t_env >= t_max:
                break

        learner.update(pretrain=pretrain)

        if terminated is not None and terminated:
            logger.console_logger.info(
                f"Terminate training by the learner at t_env = {t_env}. Finish training."
            )
            break

        # Execute test runs once in a while & final evaluation
        if (
            not getattr(main_args, "smoke_skip_evaluation", False)
            and (
                (t_env - last_test_T) / main_args.test_interval >= 1
                or t_env >= t_max
            )
        ):
            test_start_time = time.time()

            with th.no_grad():
                for task in main_args.test_tasks:
                    task2runner[task].t_env = t_env
                    _hissd_fast_run_eval_task(task, n_test_runs)

                # test_pretrain for pretrained tasks
                if pretrain and test_task2offlinedata is not None:
                    for task, data_buffer in test_task2offlinedata.items():
                        episode_sample = data_buffer.sample(batch_size_train * 10)

                        if episode_sample.device != task2args[task].device:
                            episode_sample.to(task2args[task].device)

                        if hasattr(learner, "test_pretrain"):
                            learner.test_pretrain(episode_sample, t_env, episode, task)
                        else:
                            raise ValueError(
                                "Do test_pretrain with a learner that does not have a `test_pretrain` method!"
                            )

            test_time_total += time.time() - test_start_time

            def _hissd_latest_win(_task):
                _key = f"{_task}/test_battle_won_mean"
                _rows = logger.stats.get(_key, [])
                if not _rows:
                    return None
                return float(_rows[-1][1])

            _hissd_src = [
                _hissd_latest_win(task)
                for task in _hissd_source_tasks
            ]
            _hissd_tgt = [
                _hissd_latest_win(task)
                for task in _hissd_target_tasks
            ]

            _hissd_src = [
                value for value in _hissd_src
                if value is not None
            ]
            _hissd_tgt = [
                value for value in _hissd_tgt
                if value is not None
            ]

            _hissd_src_mean = (
                sum(_hissd_src) / len(_hissd_src)
                if _hissd_src else None
            )
            _hissd_tgt_mean = (
                sum(_hissd_tgt) / len(_hissd_tgt)
                if _hissd_tgt else None
            )

            if (
                _hissd_src_mean is not None
                and _hissd_tgt_mean is not None
            ):
                logger.console_logger.info(
                    "source={:.1f}% target={:.1f}%".format(
                        100.0 * _hissd_src_mean,
                        100.0 * _hissd_tgt_mean,
                    )
                )

            progress_metrics = {}
            if _hissd_src_mean is not None:
                progress_metrics["source"] = (
                    f"{100.0 * _hissd_src_mean:.1f}%"
                )
            if _hissd_tgt_mean is not None:
                progress_metrics["target"] = (
                    f"{100.0 * _hissd_tgt_mean:.1f}%"
                )
            if progress_metrics:
                _v3_postfix.update(progress_metrics)
                pbar.set_postfix(_v3_postfix, refresh=True)

            # Progress/ETA are displayed by tqdm.
            last_time = time.time()
            last_test_T = t_env

        if main_args.save_model and (
            t_env - model_save_time >= main_args.save_model_interval
            or model_save_time == 0
            or t_env >= t_max
        ):
            if pretrain:
                save_path = os.path.join(main_args.pretrain_save_dir, str(t_env))
            else:
                save_path = os.path.join(main_args.save_dir, str(t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            learner.save_models(save_path)

            _hissd_save_resume_state(
                save_path,
                learner,
                main_args,
                t_env,
                episode,
                train_tasks,
                last_test_T,
                last_log_T,
            )

            model_save_time = t_env

        if (t_env - last_log_T) >= main_args.log_interval:
            last_log_T = t_env
            logger.log_stat("episode", episode, t_env)
            # Concise console mode: stats remain recorded by Sacred.

    pbar.close()


def run_sequential(args, logger):
    # Init runner so we can get env info
    args.n_tasks = len(args.train_tasks)
    # define main_args
    main_args = copy.deepcopy(args)

    if getattr(main_args, "pretrain", False):
        all_tasks = list(set(args.train_tasks + args.test_tasks + args.pretrain_tasks))
    else:
        all_tasks = list(set(args.train_tasks + args.test_tasks))

    task2args, task2runner, task2buffer, task2scheme, task2groups, task2preprocess = (
        init_tasks(all_tasks, main_args, logger)
    )
    task2buffer_scheme = {task: task2buffer[task].scheme for task in all_tasks}

    # define mac
    mac = mac_REGISTRY[main_args.mac](
        train_tasks=all_tasks,
        task2scheme=task2buffer_scheme,
        task2args=task2args,
        main_args=main_args,
    )

    for task in main_args.test_tasks:
        task2runner[task].setup(
            scheme=task2scheme[task],
            groups=task2groups[task],
            preprocess=task2preprocess[task],
            mac=mac,
        )
        task2runner[task]._hissd_fast_eval_setup = (
            task2scheme[task],
            task2groups[task],
            task2preprocess[task],
            mac,
        )

    # define learner
    learner = le_REGISTRY[main_args.learner](mac, logger, main_args)

    if main_args.use_cuda:
        learner.cuda()

    stage2_init_checkpoint = str(
        getattr(main_args, "stage2_init_checkpoint", "")
    )
    if stage2_init_checkpoint:
        if main_args.checkpoint_path:
            raise RuntimeError(
                "stage2_init_checkpoint and checkpoint_path are mutually exclusive"
            )
        if not hasattr(learner, "initialize_from_official"):
            raise RuntimeError(
                "stage2_init_checkpoint requires the capability/team adapter learner"
            )
        learner.initialize_from_official(stage2_init_checkpoint)
    if hasattr(learner, "arm_freeze_validation"):
        learner.arm_freeze_validation()

    if main_args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(main_args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(
                    main_args.checkpoint_path
                )
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(main_args.checkpoint_path):
            full_name = os.path.join(main_args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if main_args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(
                timesteps, key=lambda x: abs(x - main_args.load_step)
            )

        model_path = os.path.join(main_args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)

        if not (
            main_args.evaluate
            or main_args.save_replay
        ):
            if getattr(
                main_args,
                "pretrain",
                False,
            ):
                raise RuntimeError(
                    "HISSD_RESUME_V3 currently supports "
                    "normal HiSSD training only "
                    "(pretrain=False)."
                )

            resume_path = os.path.join(
                model_path,
                "resume.th",
            )

            if not os.path.isfile(resume_path):
                raise RuntimeError(
                    "Training resume requested but "
                    "resume.th is missing from {}".format(
                        model_path
                    )
                )

            resume_state = (
                _hissd_load_resume_state(
                    resume_path,
                    learner,
                    main_args,
                )
            )

            # Restore exact list ordering because
            # np.random.shuffle() mutates it in place.
            main_args.train_tasks[:] = list(
                resume_state["train_tasks"]
            )

            setattr(
                main_args,
                "_resume_state",
                resume_state,
            )

            logger.console_logger.info(
                "Exact HiSSD resume state loaded "
                "from {} at t_env={} "
                "training_steps={}".format(
                    model_path,
                    resume_state["t_env"],
                    resume_state["training_steps"],
                )
            )

        if main_args.evaluate or main_args.save_replay:
            evaluate_sequential(main_args, logger, task2runner)
            return

    if (
        getattr(main_args, "pretrain", True)
        and getattr(main_args, "agent") != "mt_odis_ns"
    ):
        # initialize training data for each task
        task2offlinedata = {}
        for task in main_args.pretrain_tasks:
            # create offline data buffer
            task2offlinedata[task] = OfflineBuffer(
                task,
                main_args.pretrain_tasks_data_quality[task],
                data_folder=main_args.offline_data_name,
                offline_data_size=args.offline_data_size,
                random_sample=args.offline_data_shuffle,
            )

        test_task2offlinedata = None
        # add test data if learner has `test_pretrain` function
        if hasattr(learner, "test_pretrain") and hasattr(
            main_args, "test_tasks_data_quality"
        ):
            test_task2offlinedata = {}
            for task in main_args.test_tasks_data_quality.keys():
                test_task2offlinedata[task] = OfflineBuffer(
                    task,
                    main_args.test_tasks_data_quality[task],
                    data_folder=main_args.offline_data_name,
                    offline_data_size=args.offline_data_size,
                    random_sample=args.offline_data_shuffle,
                )

        logger.console_logger.info(
            "Beginning pre-training with {} timesteps for each task".format(
                main_args.pretrain_steps
            )
        )
        train_sequential(
            main_args.pretrain_tasks,
            main_args,
            logger,
            learner,
            task2args,
            task2runner,
            task2offlinedata,
            pretrain=True,
            test_task2offlinedata=test_task2offlinedata,
        )
        logger.console_logger.info(f"Finished pretraining")
        test_task2offlinedata = None  # free memory

        save_path = os.path.join(
            main_args.pretrain_save_dir, str(main_args.pretrain_steps)
        )
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving models to {}".format(save_path))
        learner.save_models(save_path)

    # elif hasattr(main_args, "pretrain") and getattr(main_args, "agent") != 'mt_model':
    #     # load models from pretrained model directory
    #     load_path = os.path.join(main_args.pretrain_save_dir, str(main_args.pretrain_steps))
    #     learner.load_models(load_path)
    #     logger.console_logger.info("Load pretrained models from {}".format(load_path))

    # initialize training data for each task
    task2offlinedata = {}
    for task in main_args.train_tasks:
        # create offline data buffer
        task2offlinedata[task] = OfflineBuffer(
            task,
            main_args.train_tasks_data_quality[task],
            data_folder=main_args.offline_data_name,
            offline_data_size=args.offline_data_size,
            random_sample=args.offline_data_shuffle,
        )

    logger.console_logger.info(
        "Beginning multi-task offline training with {} timesteps for each task".format(
            main_args.t_max
        )
    )
    train_sequential(
        main_args.train_tasks,
        main_args,
        logger,
        learner,
        task2args,
        task2runner,
        task2offlinedata,
    )

    # save the final model
    if main_args.save_model:
        save_path = os.path.join(main_args.save_dir, str(main_args.t_max))
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving final models to {}".format(save_path))
        learner.save_models(save_path)

    for task in args.test_tasks:
        task2runner[task].close_env()
    logger.console_logger.info(f"Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
=======
import os
import pprint
import time
import threading
import torch as th
from tqdm.auto import tqdm
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import copy
import json
import shutil

from learners.multi_task import REGISTRY as le_REGISTRY
from runners.multi_task import REGISTRY as r_REGISTRY
from controllers.multi_task import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.offline_buffer import OfflineBuffer
from components.transforms import OneHot

import numpy as np


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    results_save_dir = args.results_save_dir

    if args.use_tensorboard and not args.evaluate:
        # only log tensorboard when in training mode
        # though we are always in training mode when we reach here
        tb_exp_direc = os.path.join(results_save_dir, "tb_logs")
        logger.setup_tb(tb_exp_direc)

    # set model save dir
    args.save_dir = os.path.join(results_save_dir, "models")

    # write config file
    config_str = json.dumps(vars(args), indent=4)
    with open(os.path.join(results_save_dir, "config.json"), "w") as f:
        f.write(config_str)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(main_args, logger, task2runner):
    n_episodes = main_args.test_nepisode
    worker_count = int(os.environ.get("HISSD_EVAL_WORKERS", "1"))
    retry_count = int(os.environ.get("HISSD_EVAL_RETRIES", "2"))
    if not 1 <= worker_count <= 10:
        raise ValueError("HISSD_EVAL_WORKERS must be between 1 and 10")
    if not 0 <= retry_count <= 5:
        raise ValueError("HISSD_EVAL_RETRIES must be between 0 and 5")
    if n_episodes % worker_count:
        raise ValueError(
            "test_nepisode must be divisible by HISSD_EVAL_WORKERS"
        )

    eval_started = time.time()
    total_tasks = len(main_args.test_tasks)
    with th.no_grad():
        for task_index, task in enumerate(main_args.test_tasks, start=1):
            task_started = time.time()
            original_runner = task2runner[task]

            if worker_count == 1:
                for _ in range(n_episodes):
                    original_runner.run(test_mode=True)
            else:
                scheme, groups, preprocess, mac = (
                    original_runner._hissd_fast_eval_setup
                )
                parallel_args = copy.deepcopy(original_runner.args)
                parallel_args.batch_size_run = worker_count
                parallel_args.test_nepisode = n_episodes
                parallel_args.parallel_env_timeout_seconds = float(
                    os.environ.get("HISSD_ENV_TIMEOUT_SECONDS", "120")
                )

                for attempt in range(retry_count + 1):
                    parallel_runner = None
                    try:
                        parallel_runner = r_REGISTRY["mt_parallel"](
                            args=parallel_args,
                            logger=logger,
                            task=task,
                        )
                        parallel_runner.setup(
                            scheme=scheme,
                            groups=groups,
                            preprocess=preprocess,
                            mac=mac,
                        )
                        for _ in range(n_episodes // worker_count):
                            parallel_runner.run(test_mode=True)
                        break
                    except Exception as eval_error:
                        if attempt >= retry_count:
                            raise
                        logger.console_logger.warning(
                            "HISSD_EVAL_RETRY task=%s attempt=%s/%s error=%s",
                            task,
                            attempt + 1,
                            retry_count,
                            eval_error,
                        )
                        time.sleep(1.0)
                    finally:
                        if parallel_runner is not None:
                            try:
                                parallel_runner.close_env()
                            finally:
                                for process in parallel_runner.ps:
                                    process.join(timeout=3.0)
                                    if process.is_alive():
                                        process.terminate()
                                        process.join(timeout=3.0)
                                    if process.is_alive():
                                        process.kill()
                                        process.join(timeout=3.0)

            if main_args.save_replay:
                original_runner.save_replay()

            original_runner.close_env()

            win_key = f"{task}/test_battle_won_mean"
            win_rows = logger.stats.get(win_key, [])
            win_rate = float(win_rows[-1][1]) if win_rows else float("nan")
            elapsed = time.time() - eval_started
            mean_task_seconds = elapsed / task_index
            eta_seconds = mean_task_seconds * (total_tasks - task_index)
            logger.console_logger.info(
                "EVAL_PROGRESS completed=%s/%s map=%s episodes=%s "
                "win_rate=%.1f%% map_seconds=%.1f elapsed_seconds=%.1f "
                "eta_seconds=%.1f",
                task_index,
                total_tasks,
                task,
                n_episodes,
                win_rate * 100.0,
                time.time() - task_started,
                elapsed,
                eta_seconds,
            )

    logger.log_stat("episode", 0, 0)
    logger.print_recent_stats()


def init_tasks(task_list, main_args, logger):
    task2args, task2runner, task2buffer = {}, {}, {}
    task2scheme, task2groups, task2preprocess = {}, {}, {}

    for task in task_list:
        # define task_args
        task_args = copy.deepcopy(main_args)
        if task_args.env == "sc2":
            task_args.env_args["map_name"] = task
        elif task_args.env == "gymma":
            task_args.env_args["N"] = task
        task2args[task] = task_args

        task_runner = r_REGISTRY[main_args.runner](
            args=task_args, logger=logger, task=task
        )
        task2runner[task] = task_runner

        # Set up schemes and groups here
        env_info = task_runner.get_env_info()
        for k, v in env_info.items():
            setattr(task_args, k, v)

        # Default/Base scheme
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            # "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        # For individual rewards in gymmai reward is of shape (1, n_agents)
        if main_args.env == "sc2":
            scheme["reward"] = {"vshape": (1,)}
        elif main_args.common_reward:
            scheme["reward"] = {"vshape": (1,)}
        else:
            scheme["reward"] = {"vshape": (main_args.n_agents,)}
        groups = {"agents": task_args.n_agents}
        preprocess = {
            "actions": ("actions_onehot", [OneHot(out_dim=task_args.n_actions)])
        }

        task2buffer[task] = ReplayBuffer(
            scheme,
            groups,
            1,
            env_info["episode_limit"] + 1,
            preprocess=preprocess,
            device="cpu" if task_args.buffer_cpu_only else task_args.device,
        )

        # store task information
        task2scheme[task], task2groups[task], task2preprocess[task] = (
            scheme,
            groups,
            preprocess,
        )

    return (
        task2args,
        task2runner,
        task2buffer,
        task2scheme,
        task2groups,
        task2preprocess,
    )



# ======================== HISSD_RESUME_V3 ========================

def _hissd_resume_signature(main_args):
    keys = (
        "task",
        "seed",
        "agent",
        "mac",
        "mixer",
        "optim_type",
        "lr",
        "critic_lr",
        "gamma",
        "beta",
        "td_weight",
        "coef_conservative",
        "coef_dist",
        "coef_kl",
        "epsilon",
        "batch_size",
        "weight_decay",
        "grad_norm_clip",
        "target_update_interval",
        "double_q",
        "entity_embed_dim",
        "attn_embed_dim",
        "skill_dim",
        "common_skill_adapter_enabled",
        "common_skill_adapter_hidden_dim",
        "mixing_embed_dim",
        "hypernet_layers",
        "hypernet_embed",
        "rnn_hidden_dim",
        "c_step",
        "ssl_type",
        "ssl_time_window",
        "double_neg",
        "head",
        "depth",
        "num_stack_frames",
        "vq_skill",
        "code_dim",
        "offline_data_size",
        "offline_data_shuffle",
        "offline_data_name",
    )

    sig = {
        key: getattr(main_args, key, None)
        for key in keys
    }

    # Task order itself changes because np.random.shuffle() is in-place.
    # Signature checks task membership; exact current ordering is saved
    # separately as training state.
    sig["train_task_set"] = tuple(
        sorted(getattr(main_args, "train_tasks", []))
    )

    qualities = getattr(
        main_args,
        "train_tasks_data_quality",
        {},
    )

    sig["train_tasks_data_quality"] = tuple(
        sorted(qualities.items())
    )

    return sig


def _hissd_capture_rng():
    import random

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": th.get_rng_state(),
    }

    if th.cuda.is_available():
        state["cuda"] = th.cuda.get_rng_state_all()

    return state


def _hissd_restore_rng(state):
    import random

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    # torch.set_rng_state() requires a CPU ByteTensor.
    # resume.th itself may be loaded with map_location="cuda",
    # so explicitly move RNG states back to CPU.
    torch_rng = state["torch"]
    if not isinstance(torch_rng, th.Tensor):
        torch_rng = th.as_tensor(torch_rng)
    torch_rng = torch_rng.detach().cpu().to(dtype=th.uint8)
    th.set_rng_state(torch_rng)

    if th.cuda.is_available() and "cuda" in state:
        cuda_rng = []
        for rng_state in state["cuda"]:
            if not isinstance(rng_state, th.Tensor):
                rng_state = th.as_tensor(rng_state)
            cuda_rng.append(
                rng_state.detach().cpu().to(dtype=th.uint8)
            )
        th.cuda.set_rng_state_all(cuda_rng)


def _hissd_save_resume_state(
    save_path,
    learner,
    main_args,
    t_env,
    episode,
    train_tasks,
    last_test_T,
    last_log_T,
):
    resume_path = os.path.join(
        save_path,
        "resume.th",
    )

    tmp_path = resume_path + ".tmp"

    payload = {
        "version": 3,

        # Outer training loop
        "t_env": int(t_env),
        "episode": int(episode),
        "model_save_time": int(t_env),
        "last_test_T": int(last_test_T),
        "last_log_T": int(last_log_T),

        # np.random.shuffle mutates this exact list.
        "train_tasks": list(train_tasks),

        # Learner counters
        "training_steps": int(
            getattr(learner, "training_steps", 0)
        ),
        "pretrain_steps": int(
            getattr(learner, "pretrain_steps", 0)
        ),
        "last_target_update_episode": int(
            getattr(
                learner,
                "last_target_update_episode",
                0,
            )
        ),

        # HiSSD's MoCo cross-task negative cache
        "last_batch": getattr(
            learner,
            "last_batch",
            {},
        ),
        "last_task": getattr(
            learner,
            "last_task",
            "",
        ),

        # Learner logging scheduler
        "task2train_info": getattr(
            learner,
            "task2train_info",
            {},
        ),

        # Both optimizer states
        "optimiser": learner.optimiser.state_dict(),
        "pre_optimiser": (
            learner.pre_optimiser.state_dict()
        ),

        # Original HiSSD save_models() does not preserve
        # the exact lagged target network.
        "target_agent": (
            learner.target_mac.agent.state_dict()
        ),

        "target_mixer": (
            learner.target_mixer.state_dict()
            if getattr(
                learner,
                "target_mixer",
                None,
            ) is not None
            else None
        ),

        # Exact stochastic stream continuation
        "rng": _hissd_capture_rng(),

        # Protect against e.g. AdamW/3e-4 -> Adam/1e-4 resume
        "config_signature": (
            _hissd_resume_signature(main_args)
        ),
    }

    th.save(payload, tmp_path)
    os.replace(tmp_path, resume_path)


def _hissd_load_resume_state(
    resume_path,
    learner,
    main_args,
):
    payload = th.load(
        resume_path,
        map_location=getattr(
            main_args,
            "device",
            "cpu",
        ),
        weights_only=False,
    )

    if int(payload.get("version", -1)) != 3:
        raise RuntimeError(
            "Unsupported HiSSD resume version: "
            "{}".format(payload.get("version"))
        )

    saved_sig = payload.get(
        "config_signature",
        {},
    )

    current_sig = _hissd_resume_signature(
        main_args
    )

    mismatches = []

    for key, saved_value in saved_sig.items():
        current_value = current_sig.get(key)

        if saved_value != current_value:
            mismatches.append(
                "{}: checkpoint={!r}, current={!r}".format(
                    key,
                    saved_value,
                    current_value,
                )
            )

    if mismatches:
        raise RuntimeError(
            "HiSSD resume config mismatch:\n  "
            + "\n  ".join(mismatches)
        )

    # Current agent/mixer and optimiser were already loaded
    # by the repository's existing learner.load_models().
    # Replace optimizer state with our exact saved state.
    learner.optimiser.load_state_dict(
        payload["optimiser"]
    )

    learner.pre_optimiser.load_state_dict(
        payload["pre_optimiser"]
    )

    # Restore lagged target network exactly.
    learner.target_mac.agent.load_state_dict(
        payload["target_agent"]
    )

    if (
        payload.get("target_mixer") is not None
        and getattr(
            learner,
            "target_mixer",
            None,
        ) is not None
    ):
        learner.target_mixer.load_state_dict(
            payload["target_mixer"]
        )

    learner.training_steps = int(
        payload["training_steps"]
    )

    learner.pretrain_steps = int(
        payload["pretrain_steps"]
    )

    learner.last_target_update_episode = int(
        payload["last_target_update_episode"]
    )

    learner.last_batch = payload.get(
        "last_batch",
        {},
    )

    learner.last_task = payload.get(
        "last_task",
        "",
    )

    learner.task2train_info = payload.get(
        "task2train_info",
        learner.task2train_info,
    )

    # Fresh train() normally sets learner.device when
    # training_steps == 0. Resume skips that branch.
    if learner.last_batch:
        first_batch = next(
            iter(learner.last_batch.values())
        )
        learner.device = first_batch.device
    else:
        learner.device = getattr(
            main_args,
            "device",
            None,
        )

    return payload

# ====================== /HISSD_RESUME_V3 =========================


def train_sequential(
    train_tasks,
    main_args,
    logger,
    learner,
    task2args,
    task2runner,
    task2offlinedata,
    t_start=0,
    pretrain=False,
    test_task2offlinedata=None,
):
    ########## start training ##########
    resume_state = getattr(
        main_args,
        "_resume_state",
        None,
    )

    if resume_state is not None:
        t_start = int(resume_state["t_env"])

    t_env = t_start

    episode = (
        int(resume_state["episode"])
        if resume_state is not None
        else 0
    )

    t_max = (
        main_args.t_max
        if not pretrain
        else main_args.pretrain_steps
    )

    model_save_time = (
        int(resume_state["model_save_time"])
        if resume_state is not None
        else 0
    )

    last_test_T = (
        int(resume_state["last_test_T"])
        if resume_state is not None
        else 0
    )

    last_log_T = (
        int(resume_state["last_log_T"])
        if resume_state is not None
        else 0
    )
    start_time = time.time()
    last_time = start_time
    test_time_total = 0
    test_start_time = 0

    # get some common information
    batch_size_train = main_args.batch_size
    batch_size_run = main_args.batch_size_run

    # do test before training
    n_test_runs = max(1, main_args.test_nepisode // batch_size_run)
    if main_args.evaluate:
        n_test_runs = 0
    if main_args.debug:
        n_test_runs = 0

    # HISSD_FAST_EVAL_V1 -- bounded, per-map temporary parallel evaluation.
    # This intentionally does NOT change batch_size_run for training.
    _fast_eval_workers = int(os.environ.get("HISSD_EVAL_WORKERS", "1"))
    if not 1 <= _fast_eval_workers <= 10:
        raise ValueError("HISSD_EVAL_WORKERS must be between 1 and 10")
    _fast_eval_retries = int(os.environ.get("HISSD_EVAL_RETRIES", "2"))
    if not 0 <= _fast_eval_retries <= 5:
        raise ValueError("HISSD_EVAL_RETRIES must be between 0 and 5")

    def _hissd_fast_run_eval_task(task, original_n_runs):
        if original_n_runs <= 0:
            return
        _eval_t0 = time.perf_counter()
        _num_episodes = original_n_runs * batch_size_run
        _worker_count = _fast_eval_workers
        if _num_episodes % _worker_count:
            raise ValueError(
                f"Cannot evaluate {_num_episodes} episodes with "
                f"{_worker_count} workers without changing episode count"
            )

        if _worker_count == 1:
            for _ in range(original_n_runs):
                task2runner[task].run(test_mode=True, pretrain=pretrain)
        else:
            original_runner = task2runner[task]
            if not hasattr(original_runner, "_hissd_fast_eval_setup"):
                raise RuntimeError("Original runner setup for parallel eval is missing")
            _scheme, _groups, _preprocess, _mac = original_runner._hissd_fast_eval_setup
            _parallel_args = copy.deepcopy(task2args[task])
            _parallel_args.batch_size_run = _worker_count
            _parallel_args.test_nepisode = _num_episodes
            _parallel_args.parallel_env_timeout_seconds = float(
                os.environ.get("HISSD_ENV_TIMEOUT_SECONDS", "120")
            )
            for _attempt in range(_fast_eval_retries + 1):
                _parallel_runner = None
                try:
                    _parallel_runner = r_REGISTRY["mt_parallel"](
                        args=_parallel_args, logger=logger, task=task
                    )
                    _parallel_runner.setup(
                        scheme=_scheme,
                        groups=_groups,
                        preprocess=_preprocess,
                        mac=_mac,
                    )
                    _parallel_runner.t_env = t_env
                    for _ in range(_num_episodes // _worker_count):
                        _parallel_runner.run(
                            test_mode=True, pretrain_phase=pretrain
                        )
                    break
                except Exception as _eval_error:
                    if _attempt >= _fast_eval_retries:
                        raise
                    logger.console_logger.warning(
                        "HISSD_EVAL_RETRY task=%s steps=%s attempt=%s/%s error=%s",
                        task,
                        t_env,
                        _attempt + 1,
                        _fast_eval_retries,
                        _eval_error,
                    )
                    time.sleep(1.0)
                finally:
                    if _parallel_runner is not None:
                        try:
                            _parallel_runner.close_env()
                        finally:
                            # Reap only this temporary runner's own worker children.
                            for _p in _parallel_runner.ps:
                                _p.join(timeout=3.0)
                                if _p.is_alive():
                                    _p.terminate()
                                    _p.join(timeout=3.0)
                                if _p.is_alive():
                                    _p.kill()
                                    _p.join(timeout=3.0)
        _eval_elapsed = time.perf_counter() - _eval_t0
        logger.console_logger.info(
            "HISSD_EVAL_TIMING task=%s steps=%s episodes=%s workers=%s seconds=%.3f"
            % (task, t_env, _num_episodes, _worker_count, _eval_elapsed)
        )

    initial_n_test_runs = (
        0
        if resume_state is not None
        else n_test_runs
    )

    pbar = tqdm(
        total=t_max,
        initial=min(t_env, t_max),
        desc="HiSSD training",
        unit="step",
        dynamic_ncols=True,
        mininterval=1.0,
        disable=not getattr(main_args, "progress_bar", True),
    )

    if not getattr(main_args, "smoke_skip_evaluation", False):
        test_start_time = time.time()

        with th.no_grad():
            for task in main_args.test_tasks:
                task2runner[task].t_env = t_env
                _hissd_fast_run_eval_task(task, initial_n_test_runs)

            # test_pretrain for pretrained tasks
            if pretrain and test_task2offlinedata is not None:
                for task, data_buffer in test_task2offlinedata.items():
                    episode_sample = data_buffer.sample(batch_size_train * 3)

                    if episode_sample.device != task2args[task].device:
                        episode_sample.to(task2args[task].device)

                    if hasattr(learner, "test_pretrain"):
                        learner.test_pretrain(episode_sample, t_env, episode, task)
                    else:
                        raise ValueError(
                            "Do test_pretrain with a learner that does not have a `test_pretrain` method!"
                        )

        test_time_total += time.time() - test_start_time

    if resume_state is not None:
        _hissd_restore_rng(
            resume_state["rng"]
        )

        logger.console_logger.info(
            "HiSSD resume RNG restored at "
            "t_env={} training_steps={}".format(
                t_env,
                learner.training_steps,
            )
        )

    # HISSD_TQDM_SUMMARY_V1
    _hissd_source_tasks = list(main_args.train_tasks)
    _hissd_source_set = set(_hissd_source_tasks)
    _hissd_target_tasks = [
        task for task in main_args.test_tasks
        if task not in _hissd_source_set
    ]

    # V3_TQDM_METRICS_PATCH_20261015
    # Keep all fields across training and evaluation postfix refreshes.
    _v3_postfix = {"task": "-"}

    while t_env < t_max:
        # shuffle tasks
        np.random.shuffle(train_tasks)
        # train each task
        for task in train_tasks:

            episode_sample = task2offlinedata[task].sample(batch_size_train)

            if episode_sample.device != task2args[task].device:
                episode_sample.to(task2args[task].device)

            if pretrain:
                if hasattr(learner, "pretrain"):
                    terminated = learner.pretrain(episode_sample, t_env, episode, task)
                else:
                    raise ValueError(
                        "Do pretraining with a learner that does not have a `pretrain` method!"
                    )
            else:
                terminated = learner.train(episode_sample, t_env, episode, task)

            if terminated is not None and terminated:
                break

            t_env += 1
            episode += batch_size_run
            if pbar.n < pbar.total:
                pbar.update(1)
            _v3_postfix["task"] = task
            _v3_cache = getattr(learner, "v3_latest_metrics", None)
            if _v3_cache is not None:
                _v3_postfix.update({
                    "d/c": "%.3f" % _v3_cache["delta_ratio"],
                })
            pbar.set_postfix(_v3_postfix, refresh=False)
            if t_env >= t_max:
                break

        learner.update(pretrain=pretrain)

        if terminated is not None and terminated:
            logger.console_logger.info(
                f"Terminate training by the learner at t_env = {t_env}. Finish training."
            )
            break

        # Execute test runs once in a while & final evaluation
        if (
            not getattr(main_args, "smoke_skip_evaluation", False)
            and (
                (t_env - last_test_T) / main_args.test_interval >= 1
                or t_env >= t_max
            )
        ):
            test_start_time = time.time()

            with th.no_grad():
                for task in main_args.test_tasks:
                    task2runner[task].t_env = t_env
                    _hissd_fast_run_eval_task(task, n_test_runs)

                # test_pretrain for pretrained tasks
                if pretrain and test_task2offlinedata is not None:
                    for task, data_buffer in test_task2offlinedata.items():
                        episode_sample = data_buffer.sample(batch_size_train * 10)

                        if episode_sample.device != task2args[task].device:
                            episode_sample.to(task2args[task].device)

                        if hasattr(learner, "test_pretrain"):
                            learner.test_pretrain(episode_sample, t_env, episode, task)
                        else:
                            raise ValueError(
                                "Do test_pretrain with a learner that does not have a `test_pretrain` method!"
                            )

            test_time_total += time.time() - test_start_time

            def _hissd_latest_win(_task):
                _key = f"{_task}/test_battle_won_mean"
                _rows = logger.stats.get(_key, [])
                if not _rows:
                    return None
                return float(_rows[-1][1])

            _hissd_src = [
                _hissd_latest_win(task)
                for task in _hissd_source_tasks
            ]
            _hissd_tgt = [
                _hissd_latest_win(task)
                for task in _hissd_target_tasks
            ]

            _hissd_src = [
                value for value in _hissd_src
                if value is not None
            ]
            _hissd_tgt = [
                value for value in _hissd_tgt
                if value is not None
            ]

            _hissd_src_mean = (
                sum(_hissd_src) / len(_hissd_src)
                if _hissd_src else None
            )
            _hissd_tgt_mean = (
                sum(_hissd_tgt) / len(_hissd_tgt)
                if _hissd_tgt else None
            )

            if (
                _hissd_src_mean is not None
                and _hissd_tgt_mean is not None
            ):
                logger.console_logger.info(
                    "source={:.1f}% target={:.1f}%".format(
                        100.0 * _hissd_src_mean,
                        100.0 * _hissd_tgt_mean,
                    )
                )

            progress_metrics = {}
            if _hissd_src_mean is not None:
                progress_metrics["source"] = (
                    f"{100.0 * _hissd_src_mean:.1f}%"
                )
            if _hissd_tgt_mean is not None:
                progress_metrics["target"] = (
                    f"{100.0 * _hissd_tgt_mean:.1f}%"
                )
            if progress_metrics:
                _v3_postfix.update(progress_metrics)
                pbar.set_postfix(_v3_postfix, refresh=True)

            # Progress/ETA are displayed by tqdm.
            last_time = time.time()
            last_test_T = t_env

        if main_args.save_model and (
            t_env - model_save_time >= main_args.save_model_interval
            or model_save_time == 0
            or t_env >= t_max
        ):
            if pretrain:
                save_path = os.path.join(main_args.pretrain_save_dir, str(t_env))
            else:
                save_path = os.path.join(main_args.save_dir, str(t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            learner.save_models(save_path)

            _hissd_save_resume_state(
                save_path,
                learner,
                main_args,
                t_env,
                episode,
                train_tasks,
                last_test_T,
                last_log_T,
            )

            model_save_time = t_env

        if (t_env - last_log_T) >= main_args.log_interval:
            last_log_T = t_env
            logger.log_stat("episode", episode, t_env)
            # Concise console mode: stats remain recorded by Sacred.

    pbar.close()


def run_sequential(args, logger):
    # Init runner so we can get env info
    args.n_tasks = len(args.train_tasks)
    # define main_args
    main_args = copy.deepcopy(args)

    if getattr(main_args, "pretrain", False):
        all_tasks = list(set(args.train_tasks + args.test_tasks + args.pretrain_tasks))
    else:
        all_tasks = list(set(args.train_tasks + args.test_tasks))

    task2args, task2runner, task2buffer, task2scheme, task2groups, task2preprocess = (
        init_tasks(all_tasks, main_args, logger)
    )
    task2buffer_scheme = {task: task2buffer[task].scheme for task in all_tasks}

    # define mac
    mac = mac_REGISTRY[main_args.mac](
        train_tasks=all_tasks,
        task2scheme=task2buffer_scheme,
        task2args=task2args,
        main_args=main_args,
    )

    for task in main_args.test_tasks:
        task2runner[task].setup(
            scheme=task2scheme[task],
            groups=task2groups[task],
            preprocess=task2preprocess[task],
            mac=mac,
        )
        task2runner[task]._hissd_fast_eval_setup = (
            task2scheme[task],
            task2groups[task],
            task2preprocess[task],
            mac,
        )

    # define learner
    learner = le_REGISTRY[main_args.learner](mac, logger, main_args)

    if main_args.use_cuda:
        learner.cuda()

    stage2_init_checkpoint = str(
        getattr(main_args, "stage2_init_checkpoint", "")
    )
    if stage2_init_checkpoint:
        if main_args.checkpoint_path:
            raise RuntimeError(
                "stage2_init_checkpoint and checkpoint_path are mutually exclusive"
            )
        if not hasattr(learner, "initialize_from_official"):
            raise RuntimeError(
                "stage2_init_checkpoint requires the capability/team adapter learner"
            )
        learner.initialize_from_official(stage2_init_checkpoint)
    if hasattr(learner, "arm_freeze_validation"):
        learner.arm_freeze_validation()

    if main_args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(main_args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(
                    main_args.checkpoint_path
                )
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(main_args.checkpoint_path):
            full_name = os.path.join(main_args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if main_args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(
                timesteps, key=lambda x: abs(x - main_args.load_step)
            )

        model_path = os.path.join(main_args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)

        if not (
            main_args.evaluate
            or main_args.save_replay
        ):
            if getattr(
                main_args,
                "pretrain",
                False,
            ):
                raise RuntimeError(
                    "HISSD_RESUME_V3 currently supports "
                    "normal HiSSD training only "
                    "(pretrain=False)."
                )

            resume_path = os.path.join(
                model_path,
                "resume.th",
            )

            if not os.path.isfile(resume_path):
                raise RuntimeError(
                    "Training resume requested but "
                    "resume.th is missing from {}".format(
                        model_path
                    )
                )

            resume_state = (
                _hissd_load_resume_state(
                    resume_path,
                    learner,
                    main_args,
                )
            )

            # Restore exact list ordering because
            # np.random.shuffle() mutates it in place.
            main_args.train_tasks[:] = list(
                resume_state["train_tasks"]
            )

            setattr(
                main_args,
                "_resume_state",
                resume_state,
            )

            logger.console_logger.info(
                "Exact HiSSD resume state loaded "
                "from {} at t_env={} "
                "training_steps={}".format(
                    model_path,
                    resume_state["t_env"],
                    resume_state["training_steps"],
                )
            )

        if main_args.evaluate or main_args.save_replay:
            evaluate_sequential(main_args, logger, task2runner)
            return

    if (
        getattr(main_args, "pretrain", True)
        and getattr(main_args, "agent") != "mt_odis_ns"
    ):
        # initialize training data for each task
        task2offlinedata = {}
        for task in main_args.pretrain_tasks:
            # create offline data buffer
            task2offlinedata[task] = OfflineBuffer(
                task,
                main_args.pretrain_tasks_data_quality[task],
                data_folder=main_args.offline_data_name,
                offline_data_size=args.offline_data_size,
                random_sample=args.offline_data_shuffle,
            )

        test_task2offlinedata = None
        # add test data if learner has `test_pretrain` function
        if hasattr(learner, "test_pretrain") and hasattr(
            main_args, "test_tasks_data_quality"
        ):
            test_task2offlinedata = {}
            for task in main_args.test_tasks_data_quality.keys():
                test_task2offlinedata[task] = OfflineBuffer(
                    task,
                    main_args.test_tasks_data_quality[task],
                    data_folder=main_args.offline_data_name,
                    offline_data_size=args.offline_data_size,
                    random_sample=args.offline_data_shuffle,
                )

        logger.console_logger.info(
            "Beginning pre-training with {} timesteps for each task".format(
                main_args.pretrain_steps
            )
        )
        train_sequential(
            main_args.pretrain_tasks,
            main_args,
            logger,
            learner,
            task2args,
            task2runner,
            task2offlinedata,
            pretrain=True,
            test_task2offlinedata=test_task2offlinedata,
        )
        logger.console_logger.info(f"Finished pretraining")
        test_task2offlinedata = None  # free memory

        save_path = os.path.join(
            main_args.pretrain_save_dir, str(main_args.pretrain_steps)
        )
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving models to {}".format(save_path))
        learner.save_models(save_path)

    # elif hasattr(main_args, "pretrain") and getattr(main_args, "agent") != 'mt_model':
    #     # load models from pretrained model directory
    #     load_path = os.path.join(main_args.pretrain_save_dir, str(main_args.pretrain_steps))
    #     learner.load_models(load_path)
    #     logger.console_logger.info("Load pretrained models from {}".format(load_path))

    # initialize training data for each task
    task2offlinedata = {}
    for task in main_args.train_tasks:
        # create offline data buffer
        task2offlinedata[task] = OfflineBuffer(
            task,
            main_args.train_tasks_data_quality[task],
            data_folder=main_args.offline_data_name,
            offline_data_size=args.offline_data_size,
            random_sample=args.offline_data_shuffle,
        )

    logger.console_logger.info(
        "Beginning multi-task offline training with {} timesteps for each task".format(
            main_args.t_max
        )
    )
    train_sequential(
        main_args.train_tasks,
        main_args,
        logger,
        learner,
        task2args,
        task2runner,
        task2offlinedata,
    )

    # save the final model
    if main_args.save_model:
        save_path = os.path.join(main_args.save_dir, str(main_args.t_max))
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving final models to {}".format(save_path))
        learner.save_models(save_path)

    for task in args.test_tasks:
        task2runner[task].close_env()
    logger.console_logger.info(f"Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
>>>>>>> origin/main
