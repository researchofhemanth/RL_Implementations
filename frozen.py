import tensorflow as tf
import gym
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
env = gym.make("FrozenLake-v0")
def mlp(x,hidden_size=[32],activation="tanh"):
    for size in hidden_size:
        x = tf.layers.dense(x,units=size,activation=activation)
    return x
obs_ph= tf.placeholder(shape=(None,env.observation_space.n),dtype=tf.float32)

x= mlp(obs_ph,[32])
logits = tf.layers.dense(x,units=env.action_space.n,activation=None)
action = tf.squeeze(tf.multinomial(logits,1),axis=1)

n_episodes=500

weight_ph = tf.placeholder(shape=(None,),dtype=tf.float32)
act_ph = tf.placeholder(shape=(None,),dtype=tf.int32)

act_one_hot = tf.one_hot(act_ph,env.action_space.n)
log_prob = tf.reduce_sum(act_one_hot * tf.nn.log_softmax(logits) ,axis=1)
loss = -tf.reduce_mean(weight_ph*log_prob)
train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

batch_size=5000
for i in range(n_episodes):
    obs = env.reset()
    batch_obs,batch_act,batch_ret,batch_weights,batch_lens = [],[],[],[],[]
    ep_rews=[]
    done=False
    while True:
        obse = [0]*env.observation_space.n
        obse[obs] = 1
        obse=np.array(obse)
        batch_obs.append(obse)
        act = sess.run(action,feed_dict={obs_ph: obse.reshape(1,-1)})[0]
        obs,rew,done,_ = env.step(act)
        ep_rews.append(rew)
        batch_act.append(act)
        if i%(n_episodes-1)==0:env.render()
        if done:
            ep_ret,ep_len = sum(ep_rews),len(ep_rews)
            batch_weights += [ep_ret]*ep_len
            batch_ret.append(ep_ret)
            batch_lens.append(ep_len)
            #if len(batch_obs)>batch_size:break
            obs,ep_rews,done = env.reset(),[],False
            if len(batch_obs)>batch_size:break
    batch_loss,_ = sess.run([loss,train_op],feed_dict={obs_ph:np.array(batch_obs),act_ph: np.array(batch_act),weight_ph: np.array(batch_weights)})
    print(f'episode {i}: ')
    print(f'loss: {batch_loss}  reward:{np.mean(batch_ret)} length: {np.mean(batch_lens)}')
