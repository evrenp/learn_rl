from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, **kwargs):
        super(RandomAgent, self).__init__(**kwargs)

    def select_action_at_t(self):
        return self.action_space.sample()
