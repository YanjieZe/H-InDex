import logging
import numpy as np
from mjrl.utils.gym_env import GymEnv
try:
    from hand_imitation.env.environments.mug_place_object_env import MugPlaceObjectEnv
    from hand_imitation.env.environments.mug_pour_water_env import WaterPouringEnv
    from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate
except:
    pass
from mjrl.utils import tensor_utils
logging.disable(logging.CRITICAL)
import multiprocessing as mp
import time as timer
logging.disable(logging.CRITICAL)
import gc
import torch


dexmv_envs = ['pour-v0', 'place_inside-v0', 'relocate-mug-v0',  'relocate-foam_brick-v0', 
              'relocate-large_clamp-v0', 'relocate-mustard_bottle-v0', 'relocate-potted_meat_can-v0',
              'relocate-potted_meat_can-v0', 'relocate-sugar_box-v0','relocate-tomato_soup_can-v0',]

# Single core rollout to sample trajectories
# =======================================================
def do_rollout(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed=None,
        env_kwargs=None,
        device_id=0,
        arena_id=0,
):
    """
    :param num_traj:    number of trajectories (int)
    :param env:         environment (env class, str with env_name, or factory function)
    :param policy:      policy to use for action selection
    :param eval_mode:   use evaluation mode for action computation (bool)
    :param horizon:     max horizon length for rollout (<= env.horizon)
    :param base_seed:   base seed for rollouts (int)
    :param env_kwargs:  dictionary with parameters, will be passed to env generator
    :return:
    """

    # get the correct env behavior
    if type(env) == str:
        if env == 'pour-v0':
            e = WaterPouringEnv(has_renderer=False, tank_size=(0.15, 0.15, 0.12),arena_id=arena_id)
        elif env == 'place_inside-v0':
            e = MugPlaceObjectEnv(has_renderer=False, object_scale=0.8, mug_scale=1.7,arena_id=arena_id)
        elif 'relocate' in env and env != 'relocate-v0':
            env_name, object_name, _ = env.split('-')
            if object_name is None:
                raise ValueError("For relocate task, object name is needed.")
            friction = (1, 0.5, 0.01)
            e = YCBRelocate(has_renderer=False, object_name=object_name, friction=friction, object_scale=0.8,
                            solref="-6000 -300", randomness_scale=0.25,arena_id=arena_id)
        else:
            e = GymEnv(env)

        if env_kwargs and 'rrl_kwargs' in env_kwargs:
            from rrl.multicam import RRL, RRL_dexmv
            if env in dexmv_envs:
                env = RRL_dexmv(e, env_id=env, **env_kwargs['rrl_kwargs'], device_id=device_id)
            else:
                env = RRL(e, **env_kwargs['rrl_kwargs'], device_id=device_id)
    elif isinstance(env, GymEnv):
        env = env
    elif callable(env):
        env = env(**env_kwargs)
    else:
        print("Unsupported environment format")
        raise AttributeError

    if base_seed is not None:
        env.set_seed(base_seed)
        np.random.seed(base_seed)
    else:
        np.random.seed()
    horizon = min(horizon, env.horizon)
    paths = []

    observations_img=[]  # we save one episode of images for visualization
    for ep in range(num_traj):
        # seeding
        if base_seed is not None:
            seed = base_seed + ep
            env.set_seed(seed)
            np.random.seed(seed)

        observations=[]
            
        actions=[]
        rewards=[]
        agent_infos = []
        env_infos = []
        success = 0.

        o, img_o = env.reset()
        done = False
        t = 0

        while t < horizon and done != True:
            a, agent_info = policy.get_action(o)
            if eval_mode:
                a = agent_info['evaluation']
            env_info_base = env.get_env_infos()
            next_o, next_img_o, r, done, env_info_step = env.step(a)
            # below is important to ensure correct env_infos for the timestep
            env_info = env_info_step if env_info_base == {} else env_info_base
            observations.append(o)
            if ep==0:
                observations_img.append(img_o)
            actions.append(a)
            rewards.append(r)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            o = next_o
            img_o = next_img_o
            t += 1

            if 'success' in env_info_step.keys(): # dexmv
                success += float(env_info_step['success'])
            else:
                success += float(env_info_step['goal_achieved'])

        if env.env_id in dexmv_envs:
            if 'relocate' in env.env_id:
                threshold = 0.1
                success = 0.
                for single_env_info in env_infos:
                    success += (single_env_info['goal_measurement'] <= threshold)
            elif 'pour' in env.env_id:
                threshold = 0.3
                success = 0.
                for single_env_info in env_infos:
                    success += (single_env_info['goal_measurement'] >= threshold)
            elif 'place' in env.env_id:
                threshold = 0.1
                success = 0.
                for single_env_info in env_infos:
                    success += (single_env_info['goal_measurement'] <= threshold)
            else:
                raise NotImplementedError(f'Success measure not implemented for this env: {env.env_id}')
            is_success = float(success >= 10)
        elif env.env_id in ['relocate-v0', 'hammer-v0', 'door-v0']:
            is_success = float(success > 25.)
        elif env.env_id in ['pen-v0' ]:
            is_success = float(success > 20.)
        else:
            raise NotImplementedError(f'Success measure not implemented for this env: {env.env_id}')
        
        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done,
            is_success=is_success,
        )
        paths.append(path)

    del(env)
    gc.collect()
    # we save one episode of images for visualization
    if isinstance(observations_img[0], np.ndarray):
        paths[0]['observations_img'] = torch.from_numpy(np.stack(observations_img))
    elif isinstance(observations_img[0], torch.Tensor):
        paths[0]['observations_img'] = torch.stack(observations_img)
    else:
        raise NotImplementedError(f'Observations type not implemented: {type(paths[0]["observations"])}')
    return paths


def sample_paths(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        num_cpu = 1,
        max_process_time=3000,
        max_timeouts=4,
        suppress_print=False,
        env_kwargs=None,
        arena_id=0,
        ):

    num_cpu = 1 if num_cpu is None else num_cpu
    num_cpu = mp.cpu_count() if num_cpu == 'max' else num_cpu
    assert type(num_cpu) == int
    if num_cpu == 1:
        input_dict = dict(num_traj=num_traj, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon, base_seed=base_seed,
                          env_kwargs=env_kwargs)
        # dont invoke multiprocessing if not necessary
        return do_rollout(**input_dict)

    # do multiprocessing otherwise
    paths_per_cpu = int(np.ceil(num_traj/num_cpu))
    input_dict_list= []
    for i in range(num_cpu):
        input_dict = dict(num_traj=paths_per_cpu, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon,
                          base_seed=base_seed + i * paths_per_cpu,
                          env_kwargs=env_kwargs, device_id=i, arena_id=arena_id)
        input_dict_list.append(input_dict)
    if suppress_print is False:
        start_time = timer.time()
        print("####### Gathering Samples #######")

    results = _try_multiprocess(do_rollout, input_dict_list,
                                num_cpu, max_process_time, max_timeouts)
    paths = []
    # result is a paths type and results is list of paths
    for result in results:
        for path in result:
            paths.append(path)  

    # check paths have visual observations for visualization
    assert paths[0].get('observations_img') is not None
    if suppress_print is False:
        print("======= Samples Gathered  ======= | >>>> Time taken = %f " %(timer.time()-start_time) )

    return paths


def sample_data_batch(
        num_samples,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        num_cpu = 1,
        paths_per_call = 1,
        env_kwargs=None,
        ):

    num_cpu = 1 if num_cpu is None else num_cpu
    num_cpu = mp.cpu_count() if num_cpu == 'max' else num_cpu
    assert type(num_cpu) == int

    start_time = timer.time()
    print("####### Gathering Samples #######")
    sampled_so_far = 0
    paths_so_far = 0
    paths = []
    base_seed = 123 if base_seed is None else base_seed
    while sampled_so_far < num_samples:
        base_seed = base_seed + 12345
        new_paths = sample_paths(paths_per_call * num_cpu, env, policy,
                                 eval_mode, horizon, base_seed, num_cpu,
                                 suppress_print=True, env_kwargs=env_kwargs)
        for path in new_paths:
            paths.append(path)
        paths_so_far += len(new_paths)
        new_samples = np.sum([len(p['rewards']) for p in new_paths])
        sampled_so_far += new_samples
    print("======= Samples Gathered  ======= | >>>> Time taken = %f " % (timer.time() - start_time))
    print("................................. | >>>> # samples = %i # trajectories = %i " % (
    sampled_so_far, paths_so_far))
    return paths


def _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts):
    
    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=None)
    parallel_runs = [pool.apply_async(func, kwds=input_dict) for input_dict in input_dict_list]
    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()
        return _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts-1)

    pool.close()
    pool.terminate()
    pool.join()  
    return results
