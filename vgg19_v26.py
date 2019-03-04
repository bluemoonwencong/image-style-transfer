# -*- coding: utf-8 -*-

import tensorflow as tf

class vgg19:
    '''
    
    '''
    def __init__(self, trainable):
        self.trainable = False
        self.is_training = True
        self.is_reuse = tf.AUTO_REUSE
        # tf.AUTO_REUSE
        self.keep_prob = tf.placeholder(shape=None, dtype=tf.float32, name='keep_prob_placeholder')
        self.VARIABLE_list = []
        self.TENSOR_list = []
        # [filter_height, filter_width, filter_depth, filter_number]
        self.filter_v1 = [3,3,-1 ,64]
        self.filter_v2 = [3,3,-1 ,128]
        self.filter_v3 = [3,3,-1 ,256]
        self.filter_v4 = [3,3,-1 ,512]
        self.filter_v5 = [3,3,-1 ,512]

    def activation_summary(self, x):
        '''
        :param x: A Tensor
        :return: Add histogram summary and scalar summary of the sparsity of the tensor
        '''
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def new_variables(self, shape, name='xx'):
        '''
        tf.Variable dones't provide variable reuse, variable reuse is necessary to implement RNN and Recursive NN
        https://www.tensorflow.org/guide/variables
        https://zhuanlan.zhihu.com/p/37922147
        '''
        # output = tf.Variable(tf.truncated_normal(shape, stddev=0.05), name=name)
        output = tf.get_variable(name, shape=shape, trainable=self.trainable, initializer=tf.random_normal_initializer())
        # output = tf.constant(value=0.0, name=name, shape=shape)
        self.VARIABLE_list.append(output)
        return output

    def new_fc_layer(self, input_layer, num_outputs):
        '''
        :param input_layer: N_sample*n_feature
        :num_outputs: pass
        '''
        num_inputs = input_layer.get_shape()[1:].num_elements()
        weights    = self.new_variables(shape=[num_inputs, num_outputs])
        biases     = self.new_variables(shape=[num_outputs], name = 'fc_bias')
        output     = tf.matmul(input_layer, weights) + biases
        self.TENSOR_list.append(output)
        return output

    def new_conv_relu_layer(self, input_layer, filter_shape, stride=1):
        '''
        :param filter_shape: [filter_height, filter_width, filter_depth, filter_number]
        '''
        filter_shape[2] = input_layer.get_shape().as_list()[-1]
        # msg = 'NHWC, the C in input_layer should be equal to filter_depth=filter_shape[2]'
        # assert input_layer.get_shape().as_list()[-1] == filter_shape[2], msg
        filter_weights = self.new_variables(shape=filter_shape, name='weight')
        output         = tf.nn.conv2d(input=input_layer, filter=filter_weights, strides=[1, stride, stride, 1], padding='SAME')
        output         = output + self.new_variables(shape=[filter_shape[-1]], name='bias')
        output         = tf.nn.relu(output)
        self.TENSOR_list.append(output)
        return output


    def inference(self, input_batch_tensor, output_class):
        '''
        maybe I should see this official implementation, the checkpoint can be downloaded at
        https://github.com/tensorflow/models/tree/master/research/slim
        and https://zhuanlan.zhihu.com/p/34879333 help me a lot too
        :param input_batch_tensor: NHWC 4D tensor
        :param n_res_blocks: num_residual_blocks
        :param self.is_reuse: To build train graph, self.is_reuse=False. To build validation graph and share weights with train graph, resue=True
        :return: last layer in the network. Not softmax-ed
        
        '''
        
        input_batch_tensor_C = input_batch_tensor.get_shape().as_list()[-1]
        with tf.variable_scope('conv1_%d'%0, reuse=self.is_reuse):
            conv_zero = self.new_conv_relu_layer(input_batch_tensor, filter_shape=self.filter_v1)
        with tf.variable_scope('conv1_%d'%1, reuse=self.is_reuse):
            conv_zero = self.new_conv_relu_layer(conv_zero, filter_shape=self.filter_v1)
        with tf.variable_scope('conv1_pool', reuse=self.is_reuse):
            conv_zero = tf.nn.avg_pool(conv_zero, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            
        for i in range(2):
            with tf.variable_scope('conv2_%d'%i, reuse=self.is_reuse):
                conv_zero = self.new_conv_relu_layer(conv_zero, filter_shape=self.filter_v2)
        with tf.variable_scope('conv2_pool', reuse=self.is_reuse):
            conv_zero = tf.nn.avg_pool(conv_zero, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        for i in range(4):
            with tf.variable_scope('conv3_%d'%i, reuse=self.is_reuse):
                conv_zero = self.new_conv_relu_layer(conv_zero, filter_shape=self.filter_v3)
        with tf.variable_scope('conv3_pool', reuse=self.is_reuse):
            conv_zero = tf.nn.avg_pool(conv_zero, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        for i in range(4):
            with tf.variable_scope('conv4_%d'%i, reuse=self.is_reuse):
                conv_zero = self.new_conv_relu_layer(conv_zero, filter_shape=self.filter_v4)
        with tf.variable_scope('conv4_pool', reuse=self.is_reuse):
            conv_zero = tf.nn.avg_pool(conv_zero, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        for i in range(4):
            with tf.variable_scope('conv5_%d'%i, reuse=self.is_reuse):
                conv_zero = self.new_conv_relu_layer(conv_zero, filter_shape=self.filter_v5)
        with tf.variable_scope('conv5_pool', reuse=self.is_reuse):
            conv_zero = tf.nn.avg_pool(conv_zero, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 全连接层是不需要的，浪费空间
        # with tf.variable_scope('fc1', reuse=self.is_reuse):
        #     global_pool = tf.reduce_mean(conv_zero, [1, 2])
        #     # num_features = conv_zero.shape[1:4].num_elements()
        #     # layer_flat   = tf.reshape(conv_zero, [-1, num_features])
        #     # print('num_features: ', num_features)
            
        #     output       = self.new_fc_layer(global_pool, num_outputs=4096)
        #     output       = tf.nn.relu(output)
        #     output       = tf.nn.dropout(output, keep_prob=self.keep_prob)
        
        # with tf.variable_scope('fc2', reuse=self.is_reuse):
        #     output       = self.new_fc_layer(output, num_outputs=4096)
        #     output       = tf.nn.relu(output)
        #     output       = tf.nn.dropout(output, keep_prob=self.keep_prob)
        
        # with tf.variable_scope('fc3', reuse=self.is_reuse):
        #     output_logits= self.new_fc_layer(output, num_outputs=output_class)

        # return output_logits


# def test_graph():
#     '''
#     Run this function to look at the graph structure on tensorboard. A fast way!
#     :param train_dir:
#     '''
#     input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
#     result = inference(input_batch_tensor=input_tensor, n_res_blocks=2, output_class=output_class)
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     output = sess.run(result)
#     writer = tf.summary.FileWriter('./my_graph', sess.graph)
#     return output

# tensorboard --logdir='./my_graph'



'''
在tensorflow中，为了 节约变量存储空间 ，我们常常需要通过共享 变量作用域(variable_scope) 来实现 共享变量 。
大家比较常用也比较笨的一种方法是，在重复使用（即 非第一次使用）时，设置 reuse=True 来 再次调用 该共享变量作用域（variable_scope）。
但是这种方法太繁琐了。有种更简洁 的方法来一次性对variable_scope进行reuse，现将代码模板总结如下： 
使用 tf.Variable_scope(…, reuse=tf.AUTO_REUSE)

https://www.tensorflow.org/api_docs/python/tf/variable_scope
https://stackoverflow.com/questions/38545362/tensorflow-variable-scope-reuse-if-variable-exists/38895052

def _func(in_put, in_channel, out_channel):
    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):    ### 改动部分 ###
        weights = tf.get_variable(name="weights", shape=[2, 2, in_channel, out_channel],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        convolution = tf.nn.conv2d(input=in_put, filter=weights, strides=[1, 1, 1, 1], padding="SAME")
    print(weights, convolution)
    return convolution, weights

def _main():
    with tf.Graph().as_default():
        input_x = tf.placeholder(dtype=tf.float32, shape=[1, 4, 4, 1])

        for _ in range(5):
            output, ww = _func(input_x, 1, 1)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                import numpy as np
                _output = sess.run([output], feed_dict={input_x: np.arange(16).reshape(1, 4, 4, 1)})
                
                print(list(np.array(_output).flat))
_main()


'''


'''
### 填坑记录 *****
# Since AdamOptimizer has it's own variables, we should define the initilizer init after opt, not before.
# 断点恢复
# sess = tf.Session()
# saver = tf.train.Saver()
# saver.restore(sess, "./checkpoint/mymodel.ckpt")



# tf.nn.softmax(logits.shape=(N,out_class) )必须设置 axis=1
# tf.argmax(input.shape=(N,out_class) )必须设置 axis=1
# with tf.Session() as sess:
#     a = np.array([[1,9,3,4.0],
#                   [5,6,8,7.0]])
#     cache1 = tf.nn.softmax(logits=a)
#     print(sess.run(cache1))
#     cache2 = tf.argmax(input=cache1,axis=1)
#     print(cache2)
#     print(sess.run(cache2))



# _check_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_logits, labels=tf.one_hot(indices=y, depth=output_class))
# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_logits, labels=y)
# lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name ])
# cost = tf.reduce_mean(cross_entropy) + 0.0003*lossL2
# optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# %time input_x, input_y = data.batch_x_y_random_read_train()
# feed_dict_train = {x: input_x, y: input_y, keep_prob:1.0}

# check
# tf.nn.softmax_cross_entropy_with_logits_v2
# tf.nn.sparse_softmax_cross_entropy_with_logits
# 是否一致。结果发现，完全一致。
# 注意，检查这步，需要设置 keep_prob = 1
# print(sess.run(_check_cross_entropy, feed_dict=feed_dict_train))
# print(sess.run(cross_entropy, feed_dict=feed_dict_train))
# print(sess.run(_check_cross_entropy, feed_dict=feed_dict_train) - sess.run(cross_entropy, feed_dict=feed_dict_train))



### 踩坑记录 ***
# 去掉 new_whale 之后，要考虑 one-hot 内部实现啊！
# new_whale 原本编码为0，这会使得 batch 的标签 y 不是从 0 开始的
# 当然可以在 tf.nn.softmax_cross_entropy_with_logits_v2 里传入 y 的时候
# 用 tf.one_hot(indices=y, depth=output_class+1) 来解决，
# 可是 tf.nn.sparse_softmax_cross_entropy_with_logits 没有 depth 参数
# sess.run(tf.one_hot(indices=[1,2,3,4,5], depth=6))
# cache_p1 = sess.run(y_pred_prob, feed_dict=feed_dict_train)[0]
# np.log(cache_p1)




# # 单元测试，原来是 bn 层导致输出结果和 batch 数量有关
# t1 = new_batch_normalization_layer(x)
# t2 = new_conv_layer(x, filter_shape=[3,3,3,20], stride=1)
# t3 = new_conv_bn_relu_layer(x, filter_shape=[3,3,3,20], stride=1)
# t4 = new_bn_relu_conv_layer(x, filter_shape=[3,3,3,20], stride=1)
# t5 = new_residual_block(x,output_channel=6)
# sess.run(tf.global_variables_initializer())
# t__ = t5
# feed_dict_1 = {x: input_x_[0:50], keep_prob: 1.0}
# feed_dict_2 = {x: input_x_[0:5], keep_prob: 1.0}
# print(sess.run(t__, feed_dict=feed_dict_1)[0,0,0])
# print(sess.run(t__, feed_dict=feed_dict_2)[0,0,0])


'''


