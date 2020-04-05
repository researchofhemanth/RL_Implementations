import gym
import tensorflow as tf
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument("n_eps",type=int)
args = parser.parse_args()
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
env = gym.make("CartPole-v0")

def plot_results(X,Y):
    plt.plot(X,Y)
    plt.show()

def mlp(x,hidden_size=[128,256,128],activation="tanh"):
    for size in hidden_size:
        x = tf.layers.dense(x,units=size,activation=activation)
    return x

obs_ph = tf.placeholder(shape=(None,env.observation_space.shape[0]),dtype=tf.float32)
obs_ph_next = tf.placeholder(shape=(None,env.observation_space.shape[0]),dtype=tf.float32)
old_log_policy = tf.placeholder(shape=(None,),dtype=tf.float32)
#Actor
x = mlp(obs_ph)
logits = tf.layers.dense(x,units=env.action_space.n,activation=None)
actor = tf.squeeze(tf.multinomial(logits,1),axis=1)

#Critic
x1 = mlp(obs_ph)
logits_critic = tf.layers.dense(x1,units=1,activation=None)
#Critic Next
x2 = mlp(obs_ph_next)
logits_critic_next = tf.layers.dense(x2,units=1,activation=None)

n_episodes=args.n_eps
clip_range= 0.2
batch_size=5000
weights_ph = tf.placeholder(shape=(None,),dtype=tf.float32)
act_ph = tf.placeholder(shape=(None,),dtype=tf.int32)

r = weights_ph
td_error = (r+(0.9*logits_critic_next)) - logits_critic
v_loss = tf.reduce_sum(tf.square(td_error))
v_train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(v_loss)

act_one_hot = tf.one_hot(act_ph,env.action_space.n)
log_policy = tf.reduce_sum(act_one_hot*tf.nn.log_softmax(logits),axis=1)
ratio = tf.exp(log_policy - old_log_policy)
clipped_ratio = tf.clip_by_value(ratio,1-clip_range,1+clip_range)
loss = - tf.reduce_mean(tf.minimum(tf.stop_gradient(td_error)*clipped_ratio,tf.stop_gradient(td_error)*ratio))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

class Agent:
    def __init__(self,n_episodes):
        self.n_episodes = n_episodes
    def train(self):
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        X=[h for h in range(self.n_episodes)]
        Y=[]
        for i in range(self.n_episodes):
            obs = env.reset()
            batch_obs,batch_obs_next,batch_act,batch_ret,batch_weights,batch_lens,batch_old_log_policy=[],[],[],[],[],[],[]
            ep_rews = []
            done = False
            while True:
                batch_obs.append(obs.copy())
                act = sess.run(actor,{obs_ph:obs.reshape(1,-1)})[0]
                batch_old_log_policy.append(sess.run(log_policy,{obs_ph:obs.reshape(1,-1),act_ph:np.array([act])})[0])
                obs_next,rew,done,_ = env.step(act)
                batch_obs_next.append(obs_next)
                obs = obs_next
                ep_rews.append(rew)
                batch_act.append(act)
                if done:
                    ep_ret,ep_len= sum(ep_rews),len(ep_rews)
                    batch_weights+=[ep_ret]*ep_len
                    batch_ret.append(ep_ret)
                    batch_lens.append(ep_len)
                    obs,ep_rews,done = env.reset(),[],False
                    if len(batch_obs)>batch_size:break
             
            batch_loss,_ = sess.run([loss,train_op],feed_dict={obs_ph:np.array(batch_obs),obs_ph_next:np.array(batch_obs_next),act_ph:np.array(batch_act),weights_ph:np.array(batch_weights),old_log_policy:np.array(batch_old_log_policy)})
            print(f'episode {i}: ')
            print(f'loss: {batch_loss} reward:{np.mean(batch_ret)} length: {np.mean(batch_lens)}')
            Y.append(np.mean(batch_lens))
        plot_results(X,Y)

if __name__ == "__main__":
    agent = Agent(n_episodes)
    agent.train()

