import tensorflow as tf
import numpy as np

# 準備資料
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# W 指的是係數，斜率介於 -1 至 1 之間
# b 指的是截距，從 0 開始逼近任意數字
# y 指的是預測值
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# 我們的目標是要讓 loss（MSE）最小化
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)
train = optimizer.minimize(loss)

# 初始化
init = tf.global_variables_initializer()

# 將神經網絡圖畫出來
sess = tf.Session()
sess.run(init)

# 將迴歸線的係數與截距模擬出來
# 每跑 20 次把當時的係數與截距印出來
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
        
# 關閉 Session
sess.close()
