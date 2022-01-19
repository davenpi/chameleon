import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from chameleon import Chameleon
from tqdm.auto import tqdm


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


parser = argparse.ArgumentParser()
parser.add_argument(
    "-tt",
    "--total_timesteps",
    type=int,
    default=int(1e3),
    help="Total timesteps argument for algorithm to learn on",
)
parser.add_argument(
    "-tp",
    "--target_position",
    type=float,
    help="Target position to reach to",
    default=0.18,
)
parser.add_argument("-E", type=float, help="Young's Modulus of tongue", default=1.0)
parser.add_argument(
    "-s", "--save", type=bool, help="Save model after training", default=False
)
args = parser.parse_args()

target_pos = args.target_position
timesteps = args.total_timesteps
E = args.E
env = Chameleon(E=E, target_pos=target_pos)
wrapped = Monitor(env, filename="logger_file")
model = PPO(
    "MlpPolicy",
    wrapped,
    gamma=1,
    learning_rate=1e-4,
    verbose=0,
)
with ProgressBarManager(timesteps) as callback:
    model.learn(total_timesteps=timesteps, callback=callback)

if args.save:
    model.save(f"{timesteps}_steps_ppo")
