# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from vgg19_v26 import vgg19
import os
from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt
from tqdm import tqdm


img_H,img_W,img_C = 500, 800, 3
vgg19net = vgg19(trainable=False)
img = tf.get_variable(name='wanted', shape=(1,img_H,img_W,img_C))
beta_placeholder = tf.placeholder(shape=None, dtype=tf.float32)
lr_placeholder   = tf.placeholder(shape=None, dtype=tf.float32)
vgg19net.inference(img, 1000)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


def read_pretrained_weight():
    for i in tqdm(range(32)):
        sess.run(vgg19net.VARIABLE_list[i].assign(np.load('./weight/' + f'keras_{i}.npy')))

# only run once
print('read fixed weights from pre-trained model:')
read_pretrained_weight()



# define some help functions
def check_see():
    plt.figure(figsize=(20,10))
    for i in range(12):
        plt.subplot(3,4,i+1)
        see = sess.run(vgg19net.TENSOR_list[i])[0,:,:,32]
        plt.imshow((see-see.min())/(see.max()-see.min()), cmap='gray')
    plt.show()


# see network structrue
print('vgg19net.TENSOR_list:')
for i,o in enumerate(vgg19net.TENSOR_list):
    print('{0:>2}'.format(i),':',o)

# for i,v in enumerate(tf.global_variables()):
#     print('trainable:{0} {1:>2}'.format(v.trainable,i),':',v)

# for i,v in enumerate(tf.trainable_variables()):
#     print('trainable:{0} {1:>2}'.format(v.trainable,i),':',v)

# for i,v in enumerate(vgg19net.VARIABLE_list):
#     print('trainable:{0} {1:>2}'.format(v.trainable,i),':',v)

# _ = sess.run(img.assign(load_img('content1.jpg')))
# check_see()

# _ = sess.run(img.assign(load_img('style5.jpg')))
# check_see()




def load_img(path):
    '''
    读取jpg/png/bmp格式图片文件，返回一个(1,H,W,3)的np.array
    这三种格式像素点值域为[0,255]
    '''
    if path.split('.')[-1] == 'npy':
        return np.load(path)
    img    = Image.open(path).resize((img_W,img_H))
    img_np = np.zeros((1,img_H,img_W,img_C),dtype=np.float32)
    img_np[0,:,:,:] = np.array(img)
    # img_np = img_np/255.0*2.0
    # img_np = img_np - 1.0
    return img_np - 128.0

def save_img(path, value):
    '''
    默认网络能自己学到输入数据的值域
    默认网络能学到"亮度"、"对比度"、"饱和度"...
    '''
    value = value[0] + 128.0
    # 强制把结果线性放缩到 [-1,1]，然后线性放缩到[0,255]以保存为图片
    # value = 2*(value - value.min())/(value.max() - value.min()) - 1.0
    # value = (value + 1.0)/2.0*255.0
    value = np.clip(value,0,255).astype(np.uint8)
    value = Image.fromarray((value))
    value = ImageEnhance.Color(value).enhance(1.0)
    value = ImageEnhance.Contrast(value).enhance(1.0)
    value = ImageEnhance.Brightness(value).enhance(1.0)
    value.save(path)

def loss_content_func(sess, content_tensor_list):
    '''
    param:adjust_parameter: is used for balance the content loss value and the style loss value
    '''
    adjust_parameter = 1.00
    def loss_single_conv(fixed_content, tensor):
        num_C  = fixed_content.shape[3]
        num_HW = fixed_content.shape[1]*fixed_content.shape[2]
        return (1/(2*num_C*num_HW))*tf.reduce_sum(tf.pow(fixed_content-tensor,2))
    return sum([adjust_parameter*w*loss_single_conv(sess.run(tensor), tensor) for w, tensor in content_tensor_list])

def loss_style_func(sess, style_tensor_list):
    def gram_matrix(F, num_HW, num_C):
        F_ik = tf.reshape(F, (num_HW, num_C))
        return tf.matmul(tf.transpose(F_ik), F_ik)
    def loss_single_conv(fixed_content, tensor):
        num_C  = fixed_content.shape[3]
        num_HW = fixed_content.shape[1]*fixed_content.shape[2]
        fixed_content_gram = gram_matrix(fixed_content, num_HW, num_C)
        tensor_gram = gram_matrix(tensor, num_HW, num_C)
        return (1/(4*num_C*num_C*num_HW*num_HW))*tf.reduce_sum(tf.pow(fixed_content_gram-tensor_gram,2))
    return sum([w*loss_single_conv(sess.run(tensor), tensor) for w, tensor in style_tensor_list])



def set_content_tensor_list(_select, _weight_list=[0.1, 0.2, 0.3, 0.2, 0.1]):
    '''
    虽然理论上越靠后越抽象而且感受视野也越大
    如果把重点放在前面的层，会更接近content的内容
    如果把重点放在后面的层，会更接近content的内容
    _select = [0,1,2,3,4]
    _select = [11,12,13,14,15]
    '''
    global content_tensor_list
    content_tensor_list = [[vgg19net.TENSOR_list[i]] for i in _select]
    content_tensor_list = np.hstack((np.array(_weight_list).reshape(-1,1), content_tensor_list))
    print('the content_tensor_list is set as follow:')
    for w, t in content_tensor_list:
        print(w, t)

def set_style_tensor_list(_select, _weight_list=[0.1, 0.2, 0.3, 0.2, 0.1]):
    '''
    虽然理论上越靠后越抽象而且感受视野也越大
    如果把重点放在前面的层，会更接近style宏观层面内容，比如色块分布
    如果把重点放在后面的层，会更接近style宏观层面内容，比如纹理分布
    _select = [0,1,2,3,4]
    _select = [11,12,13,14,15]
    '''
    global style_tensor_list
    style_tensor_list = [[vgg19net.TENSOR_list[i]] for i in _select]
    style_tensor_list = np.hstack((np.array(_weight_list).reshape(-1,1), style_tensor_list))
    print('the style_tensor_list is set as follow:')
    for w, t in style_tensor_list:
        print(w, t)


def set_content_style_loss(content_img, style_img, dir_label):
    global loss_content, loss_style, loss
    global optimizer_step, savedir
    # content_img, style_img = content, style

    _ = sess.run(img.assign(load_img(content_img)))
    loss_content = loss_content_func(sess, content_tensor_list)
    _ = sess.run(img.assign(load_img(style_img)))
    loss_style = loss_style_func(sess, style_tensor_list)
    loss = beta_placeholder*loss_content + (1-beta_placeholder)*loss_style
    
    optimizer = tf.train.AdamOptimizer(lr_placeholder)
    optimizer_step = optimizer.minimize(loss)
    sess.run(tf.variables_initializer(optimizer.variables()))
    
    _ = content_img.split('.')[0]+'-'+style_img.split('.')[0]
    savedir = './output/' + dir_label + _ + '/'
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    print('loss_content and loss_style have been set by flie ' + content_img + ' and ' + style_img)


# total_iter_num = 0
def train_loop(img_names, beta, learning_rate, loop_num):
    global total_iter_num
    
    input_img_n, output_img_n = img_names
    
    sess.run(img.assign(load_img(input_img_n)))
    for i in range(total_iter_num, total_iter_num+loop_num):
        sess.run(optimizer_step, feed_dict={beta_placeholder:beta, lr_placeholder:learning_rate})
        total_iter_num = total_iter_num + 1
        if i % 100 == 99 or i == 0:
            output = sess.run(img)
            save_img(path=savedir+output_img_n+f'_iter_{i}.jpg', value=output)
            if i % 1000 == 999:
                np.save(savedir+'_'+output_img_n+f'_{i}.npy', output)
            loss_content_v, loss_style_v, loss_v = sess.run([loss_content, loss_style, loss], feed_dict={beta_placeholder:beta})
            msg = 'i:{0:5}, loss_content:{1}, loss_style:{2}, loss:{3}'.format(i, loss_content_v, loss_style_v, loss_v)
            print(msg)
            with open(savedir+'_'+output_img_n+'_recording.txt','a') as flie:
                flie.write(msg + '\n')


# ----- training -------------------------
def train(content_img, style_img, dir_label, beta_list=None):
    global total_iter_num
    set_content_style_loss(content_img, style_img, dir_label)
    noise = 0.5
    input_img = load_img(content_img)
    input_img = (1-noise)*input_img + noise*np.random.uniform(input_img.min(),input_img.max(),size=input_img.shape)
    save_img(savedir+'_input_img.jpg', input_img)

    if beta_list == None:
        beta_list = 0.0003, 0.0021, 0.0091, 0.0401, 0.1201, 0.3601, 0.6001, 0.8801, 0.9601, 0.9961
    
    for beta_value in beta_list:
        total_iter_num = 0
        output_img_n = 'beta' + str(beta_value)
        print('beta:', beta_value)
        
        img_names = (savedir+'_input_img.jpg', output_img_n)
        train_loop(img_names, beta=beta_value, learning_rate=2.1, loop_num=1000*2)
        
        img_names = (savedir+'_'+output_img_n+f'_{total_iter_num-1}.npy', output_img_n)
        train_loop(img_names, beta=beta_value, learning_rate=0.8, loop_num=1000*2)
        
        img_names = (savedir+'_'+output_img_n+f'_{total_iter_num-1}.npy', output_img_n)
        train_loop(img_names, beta=beta_value, learning_rate=0.4, loop_num=1000*1)

        # total_iter_num = 0
        # img_names = (savedir+'_'+output_img_n+f'_{}.npy', output_img_n)
        # train_loop(img_names, beta=beta_value, learning_rate=0.001, loop_num=1000*6)


print('start training:')

# 快速风格转换真的能在10s内有好的效果吗？
# https://zhuanlan.zhihu.com/p/23651687


# vgg19net.TENSOR_list:
#  0 : Tensor("conv1_0/Relu:0", shape=(1, 500, 800, 64), dtype=float32)
#  1 : Tensor("conv1_1/Relu:0", shape=(1, 500, 800, 64), dtype=float32)
#  2 : Tensor("conv2_0/Relu:0", shape=(1, 250, 400, 128), dtype=float32)
#  3 : Tensor("conv2_1/Relu:0", shape=(1, 250, 400, 128), dtype=float32)
#  4 : Tensor("conv3_0/Relu:0", shape=(1, 125, 200, 256), dtype=float32)
#  5 : Tensor("conv3_1/Relu:0", shape=(1, 125, 200, 256), dtype=float32)
#  6 : Tensor("conv3_2/Relu:0", shape=(1, 125, 200, 256), dtype=float32)
#  7 : Tensor("conv3_3/Relu:0", shape=(1, 125, 200, 256), dtype=float32)
#  8 : Tensor("conv4_0/Relu:0", shape=(1, 63, 100, 512), dtype=float32)
#  9 : Tensor("conv4_1/Relu:0", shape=(1, 63, 100, 512), dtype=float32)
# 10 : Tensor("conv4_2/Relu:0", shape=(1, 63, 100, 512), dtype=float32)
# 11 : Tensor("conv4_3/Relu:0", shape=(1, 63, 100, 512), dtype=float32)
# 12 : Tensor("conv5_0/Relu:0", shape=(1, 32, 50, 512), dtype=float32)
# 13 : Tensor("conv5_1/Relu:0", shape=(1, 32, 50, 512), dtype=float32)
# 14 : Tensor("conv5_2/Relu:0", shape=(1, 32, 50, 512), dtype=float32)
# 15 : Tensor("conv5_3/Relu:0", shape=(1, 32, 50, 512), dtype=float32)

# set_content_tensor_list(_select=[2, 3, 4,  5,  6], _weight_list=[0.16, 0.22, 0.31, 0.22, 0.16])
# set_style_tensor_list(  _select=[6, 8, 9, 10, 12], _weight_list=[0.16, 0.22, 0.31, 0.22, 0.16])

# 等待从 4000 继续训练
# train('content1.jpg', 'style1_starry.jpg', dir_label='test1--')
# train('content1.jpg', 'style2_udnie.jpg', dir_label='test1--')
# train('content1.jpg', 'style3_muse.jpg', dir_label='test1--')
# train('content1.jpg', 'style4_mosaic.jpg', dir_label='test1--')
# train('content1.jpg', 'style5_cubist.jpg', dir_label='test1--')

# train('content2.jpg', 'style1_starry.jpg', dir_label='test1--')
# train('content2.jpg', 'style2_udnie.jpg', dir_label='test1--')
# train('content2.jpg', 'style3_muse.jpg', dir_label='test1--')
# train('content2.jpg', 'style4_mosaic.jpg', dir_label='test1--')
# train('content2.jpg', 'style5_cubist.jpg', dir_label='test1--')

# train('content0.jpg', 'style1_starry.jpg', dir_label='test1--')
# train('content0.jpg', 'style2_udnie.jpg', dir_label='test1--')
# train('content0.jpg', 'style3_muse.jpg', dir_label='test1--')
# train('content0.jpg', 'style4_mosaic.jpg', dir_label='test1--')
# train('content0.jpg', 'style5_cubist.jpg', dir_label='test1--')



# set_content_tensor_list(_select=[2, 3,  4,  5,  6], _weight_list=[0.16, 0.22, 0.31, 0.22, 0.16])
# set_style_tensor_list(  _select=[10, 10, 12, 14, 15], _weight_list=[0.16, 0.22, 0.31, 0.22, 0.16])

# train('content1.jpg', 'style1_starry.jpg', dir_label='test2--')
# train('content1.jpg', 'style2_udnie.jpg', dir_label='test2--')
# train('content1.jpg', 'style3_muse.jpg', dir_label='test2--')
# train('content1.jpg', 'style4_mosaic.jpg', dir_label='test2--')
# train('content1.jpg', 'style5_cubist.jpg', dir_label='test2--')

# train('content2.jpg', 'style1_starry.jpg', dir_label='test2--')
# train('content2.jpg', 'style2_udnie.jpg', dir_label='test2--')
# train('content2.jpg', 'style3_muse.jpg', dir_label='test2--')
# train('content2.jpg', 'style4_mosaic.jpg', dir_label='test2--', beta_list=[0.3601, 0.6001, 0.8801, 0.9601, 0.9961])
# train('content2.jpg', 'style5_cubist.jpg', dir_label='test2--')

# train('content0.jpg', 'style1_starry.jpg', dir_label='test2--', beta_list=[0.3601, 0.6001, 0.8801, 0.9601, 0.9961])
# train('content0.jpg', 'style2_udnie.jpg', dir_label='test2--')
# train('content0.jpg', 'style3_muse.jpg', dir_label='test2--')
# train('content0.jpg', 'style4_mosaic.jpg', dir_label='test2--')
# train('content0.jpg', 'style5_cubist.jpg', dir_label='test2--', beta_list=[0.0401, 0.1201, 0.3601, 0.6001, 0.8801, 0.9601, 0.9961])



# set_content_tensor_list(_select=[5, 8, 11, 13, 14], _weight_list=[0.16, 0.22, 0.31, 0.22, 0.16])
# set_style_tensor_list(  _select=[8, 10, 12, 14, 15], _weight_list=[0.16, 0.22, 0.31, 0.22, 0.16])

# train('content1.jpg', 'style1_starry.jpg', dir_label='test3--')
# train('content1.jpg', 'style2_udnie.jpg', dir_label='test3--', beta_list=[0.1201, 0.3601, 0.6001, 0.8801, 0.9601, 0.9961])
# train('content1.jpg', 'style3_muse.jpg', dir_label='test3--', beta_list=[0.6001, 0.8801, 0.9601, 0.9961])
# train('content1.jpg', 'style4_mosaic.jpg', dir_label='test3--')
# train('content1.jpg', 'style5_cubist.jpg', dir_label='test3--')

# train('content2.jpg', 'style1_starry.jpg', dir_label='test3--')
# train('content2.jpg', 'style2_udnie.jpg', dir_label='test3--')
# train('content2.jpg', 'style3_muse.jpg', dir_label='test3--', beta_list=[0.0021, 0.0091, 0.0401, 0.1201, 0.3601, 0.6001, 0.8801, 0.9601, 0.9961])
# train('content2.jpg', 'style4_mosaic.jpg', dir_label='test3--')
# train('content2.jpg', 'style5_cubist.jpg', dir_label='test3--')

# train('content0.jpg', 'style1_starry.jpg', dir_label='test3--', beta_list=[0.6001, 0.8801, 0.9601, 0.9961])
# train('content0.jpg', 'style2_udnie.jpg', dir_label='test3--')
# train('content0.jpg', 'style3_muse.jpg', dir_label='test3--', beta_list=[0.9961])
# train('content0.jpg', 'style4_mosaic.jpg', dir_label='test3--', beta_list=[0.0401, 0.1201, 0.3601, 0.6001, 0.8801, 0.9601, 0.9961])
# train('content0.jpg', 'style5_cubist.jpg', dir_label='test3--')



set_content_tensor_list(_select=[3, 6, 8, 11, 14], _weight_list=[0.31, 0.31, 0.31, 0.31, 0.31])
set_style_tensor_list(  _select=[3, 6, 8, 11, 14], _weight_list=[0.31, 0.31, 0.31, 0.31, 0.31])

# train('content1.jpg', 'style1_starry.jpg', dir_label='test4--')
# train('content1.jpg', 'style2_udnie.jpg', dir_label='test4--')
# train('content1.jpg', 'style3_muse.jpg', dir_label='test4--')
# train('content1.jpg', 'style4_mosaic.jpg', dir_label='test4--')
# train('content1.jpg', 'style5_cubist.jpg', dir_label='test4--', beta_list=[0.0401, 0.1201, 0.3601, 0.6001, 0.8801, 0.9601, 0.9961])

# train('content2.jpg', 'style1_starry.jpg', dir_label='test4--')
# train('content2.jpg', 'style2_udnie.jpg', dir_label='test4--')
train('content2.jpg', 'style3_muse.jpg', dir_label='test4--', beta_list=[0.3601, 0.6001, 0.8801, 0.9601, 0.9961])
train('content2.jpg', 'style4_mosaic.jpg', dir_label='test4--')
train('content2.jpg', 'style5_cubist.jpg', dir_label='test4--')

train('content0.jpg', 'style1_starry.jpg', dir_label='test4--')
train('content0.jpg', 'style2_udnie.jpg', dir_label='test4--')
train('content0.jpg', 'style3_muse.jpg', dir_label='test4--')
train('content0.jpg', 'style4_mosaic.jpg', dir_label='test4--')
train('content0.jpg', 'style5_cubist.jpg', dir_label='test4--')

