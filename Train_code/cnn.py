#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 MECH Project
# Brian Jin, Rajinder Singh-Moon, Rayal Raj Prasad
# TensorFlow CNN Code for AlexNet and NvidNet

import tensorflow as tf
import numpy as np
import time
from imageGen import ImageGenerator


##############----------- DEFINE FUNDAMENTAL LAYER ELEMENTS ------------------##############
class conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, kernel_shape, rand_seed, index=0,strid=1):
        """
        :param input_x: The input of the conv layer. Should be a 4D array like (batch_num, img_len, img_len, channel_num)
        :param in_channel: The 4-th demension (channel number) of input matrix. For example, in_channel=3 means the input contains 3 channels.
        :param out_channel: The 4-th demension (channel number) of output matrix. For example, out_channel=5 means the output contains 5 channels (feature maps).
        :param kernel_shape: the shape of the kernel. For example, kernal_shape = 3 means you have a 3*3 kernel.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param index: The index of the layer. It is used for naming only.
        """
       # assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

        with tf.variable_scope('conv_layer_%d' % index):
            with tf.name_scope('conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                weight = tf.get_variable(name='conv_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='conv_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            # strides [1, x_movement, y_movement, 1]
            conv_out = tf.nn.conv2d(input_x, weight, strides=[1, strid, strid, 1], padding="SAME")
            cell_out = tf.nn.relu(conv_out + bias)

            self.cell_out = cell_out

            tf.summary.histogram('conv_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('conv_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out


class max_pooling_layer(object):
    def __init__(self, input_x, k_size, padding="SAME"):
        """
        :param input_x: The input of the pooling layer.
        :param k_size: The kernel size you want to behave pooling action.
        :param padding: The padding setting. Read documents of tf.nn.max_pool for more information.
        """
        with tf.variable_scope('max_pooling'):
            # strides [1, k_size, k_size, 1]
            pooling_shape = [1, k_size, k_size, 1]
            cell_out = tf.nn.max_pool(input_x, strides=pooling_shape,
                                      ksize=pooling_shape, padding=padding)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out

class max_pooling_layer_alexnet(object):
    def __init__(self, input_x, padding="SAME"):
        """
        :param input_x: The input of the pooling layer.
        :param k_size: set to 2 The kernel size you want to behave pooling action.
        :param padding: The padding setting. Read documents of tf.nn.max_pool for more information.
        """
        with tf.variable_scope('max_pooling'):
            pooling_shape = [1, 2, 2, 1] # Hard Coded for AlexNet
            k_shape = [1, 3, 3, 1] # Hard Coded for AlexNet
            cell_out = tf.nn.max_pool(input_x, strides=pooling_shape,
                                      ksize=k_shape, padding=padding)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class norm_layer(object):
    def __init__(self, input_x):
        """
        :param input_x: The input that needed for normalization.
        """
        with tf.variable_scope('batch_norm'):
            mean, variance = tf.nn.moments(input_x, axes=[0], keep_dims=True)
            cell_out = tf.nn.batch_normalization(input_x,
                                                 mean,
                                                 variance,
                                                 offset=1e-12,
                                                 scale=0.99,
                                                 variance_epsilon=1e-6,
                                                 name=None)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out

class fc_layer(object):
    def __init__(self, input_x, in_size, out_size, rand_seed, keep_prob=1.0, activation_function=None, index=0):
        """
        :param input_x: The input of the FC layer. It should be a flatten vector.
        :param in_size: The length of input vector.
        :param out_size: The length of output vector.
        :param rand_seed: An integer that presents the random seed used to generate the initial parameter value.
        :param keep_prob: The probability of dropout. Default set by 1.0 (no drop-out applied)
        :param activation_function: The activation function for the output. Default set to None.
        :param index: The index of the layer. It is used for naming only.

        """
        with tf.variable_scope('fc_layer_%d' % index):
            with tf.name_scope('fc_kernel'):
                w_shape = [in_size, out_size]
                weight = tf.get_variable(name='fc_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.weight = weight
            
            with tf.variable_scope('fc_kernel'):
                b_shape = [out_size]
                bias = tf.get_variable(name='fc_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            cell_out = tf.add(tf.matmul(input_x, weight), bias)
            if activation_function is not None:
                cell_out = activation_function(cell_out)
                
            self.cell_out = cell_out
            tf.summary.histogram('fc_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('fc_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out
    
def squared_error(output, input_y):
    with tf.name_scope('squared_error'):        
        se = tf.reduce_mean(tf.square(input_y-output))
    return se

def train_step_Adam(loss, learning_rate=1e-4):
    with tf.name_scope('train_step'):
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return step

def train_step_Mom(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss)
    return step

def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        error_num = tf.reduce_mean(tf.square(output - input_y), name='error_num')
        tf.summary.scalar('Net_error_num', error_num)
    return error_num


##############--------------  DEFINE CNN's ------------------##############
# NvidNET
def NvidNet(input_x, input_y,
          img_len=32, channel_num=3, output_size=1,
          conv_featmap=[3, 64], fc_units=[1024, 1024],
          conv_kernel_size=[11, 11], pooling_size=[2, 2],
          l2_norm=0.01, seed=235, keep_prob=1.0, is_training=True):
    """
        NvidNet: Adapted from  End to End Learning for Self-Driving Cars, Bojarski et al
        Layer, convolutional, stride, kernel dimensions  hard coded in
    """

    # Normalize Images    
    normImgs = (input_x/255.)
    
    # Convolution+Relu 1
    conv_layer_1 = conv_layer(input_x=normImgs,
                              in_channel=3,
                              out_channel=24,
                              kernel_shape=5,
                              rand_seed=seed,index=1,strid=2)        
    
    # Convolution+Relu 2 
    conv_layer_2 = conv_layer(input_x=conv_layer_1.output(),
                              in_channel=24,
                              out_channel=36,
                              kernel_shape=5,
                              rand_seed=seed,
                             index=2,strid=2)    

    # Convolution+Relu 3
    conv_layer_3 = conv_layer(input_x=conv_layer_2.output(),
                              in_channel=36,
                              out_channel=48,
                              kernel_shape=5,
                              rand_seed=seed,
                             index=3,strid=2)
  
    # Convolution+Relu 4
    conv_layer_4 = conv_layer(input_x=conv_layer_3.output(),
                              in_channel=48,
                              out_channel=64,
                              kernel_shape=3,
                              rand_seed=seed,
                             index=4)
    # Convolution+Relu 5
    conv_layer_5 = conv_layer(input_x=conv_layer_4.output(),
                              in_channel=64,
                              out_channel=64,
                              kernel_shape=3,
                              rand_seed=seed,
                             index=5)
    
    # Flatten
    pool_shape = conv_layer_5.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(conv_layer_5.output(), shape=[-1, img_vector_length])
    
    # Fully-Connected 6  + ReLU 6
    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=1164,
                          rand_seed=seed,                          
                          activation_function=tf.nn.relu,
                          index=0)
    
    fc_layer_0_drop = tf.nn.dropout(fc_layer_0.output(),keep_prob)
    
    # Fully-Connected 6  + ReLU 6
    fc_layer_1 = fc_layer(input_x=fc_layer_0_drop,
                          in_size=1164,
                          out_size=100,
                          rand_seed=seed,                          
                          activation_function=tf.nn.relu,
                          index=1)
    
    fc_layer_1_drop = tf.nn.dropout(fc_layer_1.output(),keep_prob)    
    
    # Fully-Connected 7  + ReLU 7
    fc_layer_2 = fc_layer(input_x=fc_layer_1_drop,
                          in_size=100,
                          out_size=50,
                          rand_seed=seed,                          
                          activation_function=tf.nn.relu,
                          index=2)
    fc_layer_2_drop = tf.nn.dropout(fc_layer_2.output(),keep_prob)    

        
    # Fully-Connected 7  + ReLU 7
    fc_layer_3 = fc_layer(input_x=fc_layer_2_drop,
                          in_size=50,
                          out_size=10,
                          rand_seed=seed,                          
                          activation_function=tf.nn.relu,
                          index=3)
    fc_layer_3_drop = tf.nn.dropout(fc_layer_3.output(),keep_prob)       
        
    # Fully-Connected 8  + ReLU 8
    fc_layer_4 = fc_layer(input_x=fc_layer_3_drop,
                          in_size=10,
                          out_size=output_size,
                          rand_seed=seed,
                          keep_prob=keep_prob,
                          activation_function=tf.tanh,
                          index=4)
            
    # saving the parameters for l2_norm loss
    conv_w = [conv_layer_1.weight, conv_layer_2.weight, conv_layer_3.weight, conv_layer_4.weight, conv_layer_5.weight]
    conv_b = [conv_layer_1.bias, conv_layer_2.bias, conv_layer_3.bias, conv_layer_4.bias, conv_layer_5.bias]    
    fc_w = [fc_layer_0.weight, fc_layer_1.weight, fc_layer_2.weight, fc_layer_3.weight, fc_layer_4.weight]
#    fc_b = [fc_layer_1.bias, fc_layer_2.bias,fc_layer_3.bias,fc_layer_4.bias]

    # loss
    with tf.name_scope("loss"):
        # Compute L2 Norm of Trainable Params for Regularization
        tv = tf.trainable_variables()
        l2_loss = tf.reduce_sum([tf.nn.l2_loss(w) for w in tv])                

        # Compute Squared Error Loss
        squared_error_loss = tf.reduce_mean(
            tf.square(input_y - fc_layer_4.output()),
            name='squared_error')        
  
        #loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')
        loss = tf.add(squared_error_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('NvidNet_loss', loss)

    return fc_layer_4.output(), loss



# ALEXNET
def AlexNet(input_x, input_y,
          img_len=32, channel_num=3, output_size=1,
          conv_featmap=[3, 64], fc_units=[1024, 1024],
          conv_kernel_size=[11, 11], pooling_size=[2, 2],
          l2_norm=0.01, seed=235, keep_prob=1.0):
    """
        AlexNet: adapted from as described in 2012 Alex Krizhevsky
    """
    
    # Convolution 1 (normalize image before)
    conv_layer_1 = conv_layer(input_x=(input_x-127.0)/255.0,
                              in_channel=3,
                              out_channel=64,
                              kernel_shape=11,
                              rand_seed=seed,index=1,strid=4)        
 
    # Max Pooling 1 
    pooling_layer_1 = max_pooling_layer_alexnet(input_x=conv_layer_1.output(),padding="VALID")    
        
    # Convolution 2 layer 
    conv_layer_2 = conv_layer(input_x=pooling_layer_1.output(),
                              in_channel=64,
                              out_channel=192,
                              kernel_shape=5,
                              rand_seed=seed,
                             index=2)    
    
    # Max Pooling 2
    pooling_layer_2 = max_pooling_layer_alexnet(input_x=conv_layer_2.output(),
                                        padding="VALID")    
    
    # Convolution 3
    conv_layer_3 = conv_layer(input_x=pooling_layer_2.output(),
                              in_channel=192,
                              out_channel=384,
                              kernel_shape=3,
                              rand_seed=seed,
                             index=3)
    # Convolution 4
    conv_layer_4 = conv_layer(input_x=conv_layer_3.output(),
                              in_channel=384,
                              out_channel=256,
                              kernel_shape=3,
                              rand_seed=seed,
                             index=4)
    # Convolution 5
    conv_layer_5 = conv_layer(input_x=conv_layer_4.output(),
                              in_channel=256,
                              out_channel=256,
                              kernel_shape=3,
                              rand_seed=seed,
                             index=5)
    # Max Pooling 5
    pooling_layer_5 = max_pooling_layer_alexnet(input_x=conv_layer_5.output(),
                                        padding="VALID")
    
    # Flatten
    pool_shape = pooling_layer_5.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(pooling_layer_5.output(), shape=[-1, img_vector_length])
    
    # Fully-Connected 6  + ReLU 6
    fc_layer_1 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=4096,
                          rand_seed=seed,                          
                          activation_function=tf.nn.relu,
                          index=1)

    # Dropout 6    
    fc_layer_1_drop = tf.nn.dropout(fc_layer_1.output(),keep_prob)
        
    # Fully-Connected 7  + ReLU 7
    fc_layer_2 = fc_layer(input_x=fc_layer_1_drop,
                          in_size=4096,
                          out_size=1000,
                          rand_seed=seed,                          
                          activation_function=tf.nn.relu,
                          index=2)

    # Dropout 7   
    fc_layer_2_drop = tf.nn.dropout(fc_layer_2.output(),keep_prob)
    
    # Fully-Connected 7  + ReLU 7
    fc_layer_3 = fc_layer(input_x=fc_layer_2_drop,
                          in_size=1000,
                          out_size=10,
                          rand_seed=seed,                          
                          activation_function=tf.nn.relu,
                          index=3)

    # Dropout 7   
    fc_layer_3_drop = tf.nn.dropout(fc_layer_3.output(),keep_prob)    
        
    # Fully-Connected 8  + ReLU 8
    fc_layer_4 = fc_layer(input_x=fc_layer_3_drop,
                          in_size=10,
                          out_size=output_size,
                          rand_seed=seed,
                          keep_prob=keep_prob,
                          activation_function=None,
                          index=4)
            
    # loss
    with tf.name_scope("loss"):
        tv = tf.trainable_variables()
        l2_loss = tf.reduce_sum([tf.nn.l2_loss(w) for w in tv])           

        # Compute Squared Error Loss
        squared_error_loss = tf.reduce_mean(
            tf.square(input_y - fc_layer_4.output()),
            name='squared_error')        

        loss = tf.add(squared_error_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('AlexNet_loss', loss)

    return fc_layer_4.output(), loss




def ModNVidNet(input_x, input_y,
          img_len=32, channel_num=3, output_size=10,
          conv_featmap=[6, 16], fc_units=[84],
          conv_kernel_size=[5, 5], pooling_size=[2, 2],
          l2_norm=0.01, seed=235, keep_prob=1.0):
    """
        Modified/ Simplified NvidiaNet CNN Architecture

    """

    conv_layer_0 = conv_layer(input_x=(input_x-127.)/255.,
                              in_channel=3,
                              out_channel=24,
                              kernel_shape=5,
                              rand_seed=seed)
    
    pooling_layer_0 = max_pooling_layer(input_x=conv_layer_0.output(),
                                        k_size=pooling_size[0],
                                        padding="VALID")

    # Implement Batch Normalization     
    bn_layer_0 = norm_layer(input_x=pooling_layer_0.output())

    
    # second conv layer 
    conv_layer_1 = conv_layer(input_x=bn_layer_0.output(),
                              in_channel=24,
                              out_channel=64,
                              kernel_shape=3,
                              rand_seed=seed,
                             index=1)
    pooling_layer_1 = max_pooling_layer(input_x=conv_layer_1.output(),
                                        k_size=2,
                                        padding="VALID")

    bn_layer_1 = norm_layer(input_x=pooling_layer_1.output())
    
    # flatten
    pool_shape = bn_layer_1.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(bn_layer_1.output(), shape=[-1, img_vector_length])

    
    # fc layer
    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=256,
                          rand_seed=seed,                          
                          activation_function=tf.nn.relu,
                          index=0)

    # Dropout    
    fc_layer_0_drop = tf.nn.dropout(fc_layer_0.output(),keep_prob)
    
    fc_layer_1 = fc_layer(input_x=fc_layer_0_drop,
                          in_size=256,
                          out_size=1,
                          rand_seed=seed,
                          keep_prob=keep_prob,
                          activation_function=None,
                          index=1)
            
    # saving the parameters for l2_norm loss
    conv_w = [conv_layer_0.weight, conv_layer_1.weight]
    fc_w = [fc_layer_0.weight, fc_layer_1.weight]

    # loss
    with tf.name_scope("loss"):
        tv = tf.trainable_variables()
        l2_loss = tf.reduce_sum([tf.nn.l2_loss(w) for w in tv])  
        
        squared_error_loss = tf.reduce_mean(
            tf.square(input_y - fc_layer_1.output()),
            name='squared_error')  
    
        loss = tf.add(squared_error_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('Mod_Nvid_loss', loss)

    return fc_layer_1.output(), loss

##############--------------  DEFINE CNN TRAINING FUNCTIONS ------------------##############
# Modified training function for the AlexNet model
def my_training_AlexNet(X_train, y_train, X_val, y_val,
             conv_featmap=[6],
             fc_units=[84],
             conv_kernel_size=[5],
             pooling_size=[2],
             l2_norm=0.01,
             seed=235,
             learning_rate=1e-2,
             epoch=20,
             batch_size=245,
             verbose=False,
             pre_trained_model=None,
             keep_prob=1.0, im_len=64, ofn='alexnet_training.csv', ):
    print("Building my AlexNet. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))
    print("keep_prob={}".format(keep_prob))
    print("Output filename is ={}".format(ofn))
        
    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, X_train.shape[1], X_train.shape[2], 3], dtype=tf.float32,name="xs")
        ys = tf.placeholder(shape=[None,], dtype=tf.float32,name="ys") # changed
        kp = tf.placeholder(tf.float32,name="kp")
        
    output, loss = AlexNet(xs, ys, 
                         img_len=im_len,
                         channel_num=3,
                         output_size=1,
                         conv_featmap=conv_featmap,
                         fc_units=fc_units,
                         conv_kernel_size=conv_kernel_size,
                         pooling_size=pooling_size,
                         l2_norm=l2_norm,
                         seed=seed,
                         keep_prob=kp)
      
    iters = int(X_train.shape[0] / batch_size)


    step = train_step_Adam(loss,learning_rate=1e-4)
    eve = evaluate(output, ys)

    iter_total = 0
    best_loss = 10000 # initialize loss
    cur_model_name = 'alexnet_{}'.format(int(time.time()))

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass
        # Output filename open
        ofl = open(ofn+".{}".format(time.time())+".csv", 'w')
        t_start = time.time()                
        for epc in range(epoch):
            tmptime = time.time() - t_start
            print("epoch {} ".format(epc + 1))
            im_gen = ImageGenerator(X_train,y_train,img_size=im_len)
            training_images = im_gen.next_batch_gen(batch_size)
            iter = 0
            
            for training_batch_x,training_batch_y in training_images:
                iter_total += 1
                iter += 1
                if(iter>iters):
                    break
                    
                training_images_gen = ImageGenerator(training_batch_x,training_batch_y,img_size=im_len)
                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_images_gen.x, ys: training_images_gen.y, kp: keep_prob})

                if iter_total % 10 == 0:
                    # do validation

                    tr_loss, merge_result = sess.run([loss, merge], feed_dict={xs: training_batch_x, ys: training_batch_y, kp: 1.0})
                    valid_loss = sess.run([loss], feed_dict={xs: X_val, ys: y_val, kp: 1.0})

                    if verbose:
                        print('{}/{} loss: {}; val loss : {}, tr loss : {}'.format(
                            batch_size * (iter + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_loss, tr_loss))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    
                    if valid_loss[0] < best_loss:
                        print('Best validation loss! iteration:{} loss: {}'.format(iter_total, valid_loss))
                        best_loss = valid_loss
                        saver.save(sess, 'model/{}'.format(cur_model_name))
                    
                    # Print Training anf Validation Results
                    ofl.write(str(iter_total))             
                    ofl.write("\t")                    
                    ofl.write(str(time.time()-t_start))             
                    ofl.write("\t")                    
                    ofl.write(str(tr_loss))
                    ofl.write("\t")
                    ofl.write(str(valid_loss))
                    ofl.write("\t")
                    ofl.write(str(learning_rate))
                    ofl.write("\t")
                    ofl.write(str(batch_size))                    
                    ofl.write("\n")
        ofl.write(str(best_loss))                   
    print("Training ends. The best valid loss is {}. Model named {}.".format(best_loss, cur_model_name))            

    
# Modified training function for the AlexNet model
def my_training_NvidNet(X_train, y_train, X_val, y_val, 
             conv_featmap=[6],
             fc_units=[84],
             conv_kernel_size=[5],
             pooling_size=[2],
             l2_norm=0.01,
             seed=235,
             learning_rate=1e-2,
             epoch=20,
             batch_size=245,
             verbose=False,
             pre_trained_model=None,
             keep_prob=1.0, im_len=64, ofn='nvidnet_training.csv'):
    print("Building my NvidNet. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))
    print("keep_prob={}".format(keep_prob))
    print("Output filename is ={}".format(ofn))
        
    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, X_train.shape[1], X_train.shape[2], 3], dtype=tf.float32,name="xs")
        ys = tf.placeholder(shape=[None,], dtype=tf.float32,name="ys") # changed
        kp = tf.placeholder(tf.float32,name="kp")
        
    output, loss = NvidNet(xs, ys,
                         img_len=im_len,
                         channel_num=3,
                         output_size=1,
                         conv_featmap=conv_featmap,
                         fc_units=fc_units,
                         conv_kernel_size=conv_kernel_size,
                         pooling_size=pooling_size,
                         l2_norm=l2_norm,
                         seed=seed,
                         keep_prob=kp)
      
    iters = int(X_train.shape[0] / batch_size)


    step = train_step_Adam(loss,learning_rate=1e-4)
    eve = evaluate(output, ys)

    iter_total = 0
    best_loss = 10000 # initialize loss
    cur_model_name = 'nvidnet_{}'.format(int(time.time()))

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass
        # Output filename open
        ofl = open(ofn+".{}".format(time.time())+".csv", 'w')
        t_start = time.time()    
        
        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))
            im_gen = ImageGenerator(X_train,y_train,img_size=im_len)
            training_images = im_gen.next_batch_gen(batch_size)
            iter = 0
            
            for training_batch_x,training_batch_y in training_images:
                iter_total += 1
                iter += 1
                if(iter>iters):
                    break
                    
                training_images_gen = ImageGenerator(training_batch_x,training_batch_y,img_size=im_len)
                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_images_gen.x, ys: training_images_gen.y, kp: keep_prob})
              

                if iter_total % 10 == 0:
                    # do validation
                    tr_loss = sess.run([loss], feed_dict={xs: training_images_gen.x, ys: training_images_gen.y, kp: 1.0})  
                    valid_loss, merge_result = sess.run([loss, merge], feed_dict={xs: X_val, ys: y_val, kp: 1.0})

                    if verbose:
                        print('{}/{} loss: {}; val loss : {}, tr loss : {}'.format(
                            batch_size * (iter + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_loss, tr_loss))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                  
                    if valid_loss < best_loss:
                        print('Best validation loss! iteration:{} loss: {}'.format(iter_total, valid_loss))
                        best_loss = valid_loss
                        saver.save(sess, 'model/{}'.format(cur_model_name))
                    
                    # Print Training anf Validation Results
                    ofl.write(str(iter_total))             
                    ofl.write("\t")                    
                    ofl.write(str(time.time()-t_start))             
                    ofl.write("\t")                    
                    ofl.write(str(tr_loss))
                    ofl.write("\t")
                    ofl.write(str(valid_loss))
                    ofl.write("\t")
                    ofl.write(str(learning_rate))
                    ofl.write("\t")
                    ofl.write(str(batch_size))                    
                    ofl.write("\n")
        ofl.write(str(best_loss))                     
    print("Training ends. The best valid loss is {}. Model named {}.".format(best_loss, cur_model_name))            

# Modified Nvidia Model
def my_training_Mod_Nvid(X_train, y_train, X_val, y_val,
             conv_featmap=[6],
             fc_units=[84],
             conv_kernel_size=[5],
             pooling_size=[2],
             l2_norm=0.01,
             seed=235,
             learning_rate=1e-2,
             epoch=20,
             batch_size=245,
             verbose=False,
             pre_trained_model=None,
             keep_prob=1.0, im_len=64,ofn = 'mod_nvid_'):
    print("Building my Mod_Nvid. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))
    print("keep_prob={}".format(keep_prob))
        
    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, X_train.shape[1], X_train.shape[2], 3], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.float32)
        kp = tf.placeholder(tf.float32)
        
    output, loss = ModNVidNet(xs, ys,
                         img_len=im_len,
                         channel_num=3,
                         output_size=5,
                         conv_featmap=conv_featmap,
                         fc_units=fc_units,
                         conv_kernel_size=conv_kernel_size,
                         pooling_size=pooling_size,
                         l2_norm=l2_norm,
                         seed=seed,
                         keep_prob=kp)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step_Adam(loss)
    eve = evaluate(output, ys)

    iter_total = 0
    best_loss = 100
    cur_model_name = 'mod_nvid_{}'.format(int(time.time()))

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass

        ofl = open(ofn+".{}".format(time.time())+".csv", 'w')
        t_start = time.time()    
        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))

            for itr in range(iters):
                iter_total += 1

                training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]

                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y, kp: keep_prob})

                if iter_total % 100 == 0:
                    # do validation
                    valid_loss, merge_result = sess.run([loss, merge], feed_dict={xs: X_val, ys: y_val, kp: 1.0})
                    tr_loss, merge_result = sess.run([loss, merge], feed_dict={xs: training_batch_x, ys: training_batch_y, kp: 1.0})                 

                    if verbose:
                        print('{}/{} loss: {} validation loss : {}, tr loss: {}'.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_loss, tr_loss))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_loss < best_loss:
                        print('Best validation accuracy! iteration:{} accuracy: {}, tr loss: {}'.format(iter_total, valid_loss, tr_loss))
                        best_loss = valid_loss
                        saver.save(sess, 'model/{}'.format(cur_model_name))
                    # Print Training anf Validation Results
                    ofl.write(str(iter_total))             
                    ofl.write("\t")                    
                    ofl.write(str(time.time()-t_start))             
                    ofl.write("\t")                    
                    ofl.write(str(tr_loss))
                    ofl.write("\t")
                    ofl.write(str(valid_loss))
                    ofl.write("\t")
                    ofl.write(str(learning_rate))
                    ofl.write("\t")
                    ofl.write(str(batch_size))                    
                    ofl.write("\n")
        ofl.write(str(best_loss))                            

    print("Training ends. The best valid loss is {}. Model named {}.".format(best_loss, cur_model_name))          
       
