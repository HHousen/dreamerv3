import crafter
from gym import spaces
import abstraction.pycolab_env as pycolab_env
from pycolab.examples import fluvial_natation


class FluvialNatationEnv(pycolab_env.PyColabEnv):
    """Fluvial natation game.
    Reference:
        https://github.com/deepmind/pycolab/blob/master/pycolab/examples/fluvial_natation.py
    """

    def __init__(self,
                 max_iterations=10,
                 default_reward=-1.):
        super(FluvialNatationEnv, self).__init__(
            max_iterations=max_iterations,
            default_reward=default_reward,
            action_space=spaces.Discrete(2 + 1),
            resize_scale=8)

    def make_game(self):
        return fluvial_natation.make_game()



def main():

    import warnings
    import dreamerv3
    import abstraction.box_world2 as bw
    from dreamerv3 import embodied

    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    # See configs.yaml for all options.
    from dreamerv3 import agent as agt
    # config = embodied.Config(dreamerv3.configs["defaults"])
    config = embodied.Config(agt.Agent.configs['defaults'])
    config = config.update(agt.Agent.configs["xlarge"])
    config = config.update(
        {
            "logdir": "dreamer-boxworld-logs2/",
            "run.train_ratio": 512,
            #   'run.log_every': 30,  # Seconds
            "batch_size": 16,
            #   'jax.prealloc': False,
            "encoder.mlp_keys": "$^",
            "decoder.mlp_keys": "$^",
            "encoder.cnn_keys": "image",
            "decoder.cnn_keys": "image",
            # 'jax.platform': 'cpu',
        }
    )
    config = embodied.Flags(config).parse()
    print(config)

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            #   embodied.logger.TensorBoardOutput(logdir),
            embodied.logger.WandBOutput("useless", logdir, config),
            # embodied.logger.MLFlowOutput(logdir.name),
        ],
    )

    from embodied.envs import from_gym
    from embodied.envs.crafter import Crafter

    # env = bw.BoxWorldEnv(
    #     max_num_steps=160,
    #     grid_size=14,
    #     solution_length=(1,),
    #     num_forward=(0,),
    #     num_backward=(0,),
    #     branch_length=1,
    # )
    env = FluvialNatationEnv(max_iterations=250)
    # env = crafter.Env(size=(64, 64), reward=True, seed=42)
    env = from_gym.FromGym(env, obs_key="image")  # Or obs_key='vector'.
    # env = Crafter("reward")
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / "replay"
    )
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length
    )
    embodied.run.train(agent, env, replay, logger, args)
    # embodied.run.eval_only(agent, env, logger, args)


if __name__ == "__main__":
    main()
