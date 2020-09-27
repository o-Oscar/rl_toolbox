
from models.actor import SimpleActor, MixtureOfExpert
from models.critic import Critic

from environments.cartpole import CartPoleEnv

experts = ["results/prim_swing/models/{}_{}", "results/prim_stab/models/{}_{}"]
result = "results/mixture/models/{}_{}"

env = CartPoleEnv()
mix = MixtureOfExpert(env)
for i, name in enumerate(experts):
	primitive = SimpleActor(env)
	mix.load_specific_primitive(name, i)

mix.save_primitive(result)
mix.save_influence(result)

critic = Critic(env)
critic.model.save_weights(result.format("critic", "model"), overwrite=True)