import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from multiagent.charge_environment import MultiPileEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent import Agent
from common.replay_buffer import Buffer


class Runner:
    def __init__(self, args, env: MultiPileEnv):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    # def run(self):
    #     returns = []
    #     for time_step in tqdm(range(self.args.time_steps)):
    #         # reset the environment
    #         if time_step % self.episode_limit == 0:
    #             s = self.env.reset()
    #         u = []
    #         actions = []
    #         with torch.no_grad():
    #             for agent_id, agent in enumerate(self.agents):
    #                 action = agent.select_action(s[agent_id], self.noise, self.epsilon)
    #                 u.append(action)
    #                 actions.append(action)
    #         for i in range(self.args.n_agents, self.args.n_players):
    #             actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
    #         s_next, r, done, info = self.env.step(actions)
    #         self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
    #         s = s_next
    #         if self.buffer.current_size >= self.args.batch_size:
    #             transitions = self.buffer.sample(self.args.batch_size)
    #             for agent in self.agents:
    #                 other_agents = self.agents.copy()
    #                 other_agents.remove(agent)
    #                 agent.learn(transitions, other_agents)
    #         if time_step > 0 and time_step % self.args.evaluate_rate == 0:
    #             returns.append(self.evaluate())
    #             plt.figure()
    #             plt.plot(range(len(returns)), returns)
    #             plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
    #             plt.ylabel('average returns')
    #             plt.savefig(self.save_path + '/plt.png', format='png')
    #         self.noise = max(0.05, self.noise - 0.0000005)
    #         self.epsilon = max(0.05, self.epsilon - 0.0000005)
    #         np.save(self.save_path + '/returns.pkl', returns)

    def charge_run(self):
        # np.random.seed(0)
        returns = []
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
            u = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
            self.env.world_step(u)
            s_next, r, done, info = self.env.step(u)
            self.buffer.store_episode(s, u, r, s_next)
            s = s_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.charge_evaluate())
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)

            np.save(self.save_path + '/returns.pkl', returns)

    # def evaluate(self):
    #     returns = []
    #     for episode in range(self.args.evaluate_episodes):
    #         # reset the environment
    #         s = self.env.reset()
    #         rewards = 0
    #         for time_step in range(self.args.evaluate_episode_len):
    #             self.env.render()
    #             actions = []
    #             with torch.no_grad():
    #                 for agent_id, agent in enumerate(self.agents):
    #                     action = agent.select_action(s[agent_id], 0, 0)
    #                     actions.append(action)
    #             for i in range(self.args.n_agents, self.args.n_players):
    #                 actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
    #             s_next, r, done, info = self.env.step(actions)
    #             rewards += r[0]
    #             s = s_next
    #         returns.append(rewards)
    #         print('Returns is', rewards)
    #     return sum(returns) / self.args.evaluate_episodes

    def charge_evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            writer_station = SummaryWriter()
            writer_info = SummaryWriter()
            writer_piles = SummaryWriter()
            miss_info = []
            get_info = []
            bill_info = []
            save_time_info = []
            code_info = []
            coming_evs = 0
            for time_step in range(self.args.evaluate_episode_len):
                actions = []

                # writer_station.add_scalars('station_record', {'working_num': self.env.world.working_num,
                #                                               'waiting_num': self.env.world.wait.still_wait,
                #                                               'coming_evs':
                #                                                   len(self.env.world.wait.wait_EVs) - coming_evs
                #                                               }, time_step)
                # for i, pile in enumerate(self.env.piles):
                #     if pile.connected:
                #         writer_piles[i].add_scalars('pile_%d' % (i + 1), {
                #             'ev_%d_state' % pile.state.code: pile.state.cur_b,
                #             'ev_%d_target' % pile.state.code: pile.state.tar_b
                #         }, time_step)
                coming_evs = len(self.env.world.wait.wait_EVs)
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)

                # for i, pile in enumerate(self.env.piles):
                #     if pile.connected:
                #         writer_piles.add_scalars('piles', {
                #             'pile_%d_cur_b' % i: pile.state.cur_b,
                #             'pile_%d_tar_b' % i: pile.state.tar_b
                #         }, time_step)
                self.env.world_step(actions)
                # for i, pile in enumerate(self.env.piles):
                #     if pile.connected:
                #         writer_piles.add_scalars('piles', {
                #             'pile_%d_action' % i: pile.real_action,
                #             'pile_%d_cur_b_new' % i: pile.state.cur_b
                #         }, time_step)
                s_next, r, done, info = self.env.step(actions)

                miss_info += info['energy_miss']
                bill_info += info['ev_payment']
                save_time_info += info['ev_save_time']
                get_info += info['energy_get']
                code_info += info['ev_code']
                rewards += r[0]
                s = s_next

                # writer_station.add_scalars('station_record', {'rewards': rewards,
                #                                               'extra_deal': self.env.world.extra_deal,
                #                                               'es_action': self.env.world.es.real_action,
                #                                               'es_capacity': self.env.es.state.cur_c,
                #                                               'piles_power_sum': self.env.world.piles_power_sum,
                #                                               'pv_power': self.env.world.pv.power[
                #                                                   (self.env.world.cur_t - 1) % len(
                #                                                       self.env.world.pv.power)]
                #                                               }, time_step)

            # for i in range(len(miss_info)):
            #     writer_info.add_scalars('left_ev', {'get': get_info[i],
            #                                         'bill': bill_info[i]}, i)

            writer_station.close()
            writer_info.close()
            writer_piles.close()

            returns.append(rewards)
            print(np.mean(miss_info).round(1))
            print(np.mean(save_time_info).round(2))
            print(self.env.world.es.state.cur_c)
        return sum(returns) / self.args.evaluate_episodes
