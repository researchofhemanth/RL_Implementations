import tensorflow as tf
import gym
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
env = gym.make("CartPole-v0")
def plot_results(X,Y):
    plt.plot(X,Y)
    plt.show()

def mlp(x,hidden_size,activation="tanh"):
    for size in hidden_size:
        x = tf.layers.dense(x,units=size,activation=activation)
    return x
obs_ph= tf.placeholder(shape=(None,env.observation_space.shape[0]),dtype=tf.float32)
obs_ph_next= tf.placeholder(shape=(None,env.observation_space.shape[0]),dtype=tf.float32)
#Actor
x= mlp(obs_ph,[64,128])
logits = tf.layers.dense(x,units=env.action_space.n,activation=None)
actor = tf.squeeze(tf.multinomial(logits,1),axis=1)
#Critic
x2 = mlp(obs_ph,[64,128])
logits_critic = tf.layers.dense(x2,units=1,activation=None)
#Critic_Next
x1= mlp(obs_ph_next,[64,128])
logits1_critic_n = tf.layers.dense(x1,units=1,activation=None)

n_episodes=200

weight_ph = tf.placeholder(shape=(None,),dtype=tf.float32)
act_ph = tf.placeholder(shape=(None,),dtype=tf.int32)

#s,r,done,_ = env.step(
#print(critic_next)
r=weight_ph
td_error = (r+(0.9 *logits1_critic_n))-logits_critic
v_loss = tf.reduce_mean(tf.square(td_error))
print(td_error,v_loss)
v_train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(v_loss)


act_one_hot = tf.one_hot(act_ph,env.action_space.n)
log_prob = tf.reduce_sum(act_one_hot * tf.nn.log_softmax(logits) ,axis=1)
loss = -tf.reduce_mean(tf.stop_gradient(td_error)*log_prob)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

batch_size=5000
X=[k for k in range(n_episodes)]
Y=[]
for i in range(n_episodes):
    obs = env.reset()
    batch_obs,batch_obs_next,batch_act,batch_ret,batch_weights,batch_lens = [],[],[],[],[],[]
    ep_rews=[]
    done=False
    while True:
        batch_obs.append(obs.copy())
        act = sess.run(actor,feed_dict={obs_ph: obs.reshape(1,-1)})[0]
        obs_next,rew,done,_ = env.step(act)
        batch_obs_next.append(obs_next)
        obs = obs_next
        ep_rews.append(rew)
        batch_act.append(act)
        if i==n_episodes-1:env.render()
        if done:
            if i==n_episodes-1:env.render()
            ep_ret,ep_len = sum(ep_rews),len(ep_rews)
            batch_weights += [ep_ret]*ep_len
            batch_ret.append(ep_ret)
            batch_lens.append(ep_len)
            #if len(batch_obs)>batch_size:break
            obs,ep_rews,done = env.reset(),[],False
            if len(batch_obs)>batch_size:break
    batch_loss,_ = sess.run([loss,train_op],feed_dict={obs_ph:np.array(batch_obs),obs_ph_next:np.array(batch_obs_next),act_ph: np.array(batch_act),weight_ph: np.array(batch_weights)})
    print(f'episode {i}: ')
    print(f'loss: {batch_loss}  reward:{np.mean(batch_ret)} length: {np.mean(batch_lens)}')
    Y.append(np.mean(batch_lens))
plot_results(X,Y)
