import tensorflow as tf
import gym
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
env = gym.make("HalfCheetah-v2")
def mlp(x,hidden_size=(32,32),activation="tanh"):
    for size in hidden_size:
        x = tf.layers.dense(x,units=size,activation=activation)
    return x
obs_ph= tf.placeholder(shape=(None,env.observation_space.shape[0]),dtype=tf.float32)

x= mlp(obs_ph,(32,32))
logits = tf.layers.dense(x,units=env.action_space.shape[0],activation=None)
action = tf.squeeze(tf.multinomial(logits,1),axis=1)

n_episodes=10

weight_ph = tf.placeholder(shape=(None,),dtype=tf.float32)
act_ph = tf.placeholder(shape=(None,),dtype=tf.int32)

#Baseline
v=mlp(obs_ph,[32,32])
v_logits = tf.layers.dense(v,units=1,activation=None)
weights = weight_ph-v_logits
v_loss = tf.reduce_mean(tf.squared_difference(weight_ph,v_logits))
v_train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(v_loss)


act_one_hot = tf.one_hot(act_ph,env.action_space.shape[0])
log_prob = tf.reduce_sum(act_one_hot * tf.nn.log_softmax(logits) ,axis=1)
loss = -tf.reduce_mean(tf.stop_gradient(weights)*log_prob)
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
        batch_obs.append(obs.copy())
        act = sess.run(action,feed_dict={obs_ph: obs.reshape(1,-1)})[0]
        obs,rew,done,_ = env.step(act)
        ep_rews.append(rew)
        batch_act.append(act)
        if i==n_episodes-1:env.render()
        if done:
            #if i==n_episodes-1:env.render()
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
