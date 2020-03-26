# -*-codingï¼šutf8-*- 
import tensorflow as tf 
import cv2
import numpy as np
import os
#import process_20191004 as process
#import matplotlib.pyplot as plt
#import random
#import process_val as process
#import reduce_mask as reduce

model = 's_256'
input_path = r'.\input\exa'
model_path_s =r'.\model\model_%s\ckpt_00020' % model
save_path = r'.\output\exa\%s' % model
white_bk = 1

def cv_img(img_path,img_size=256):
    img_data = cv2.imread(img_path)
    img_data = cv2.resize(img_data,(img_size,img_size))
    img_data = img_data/255
    return img_data

val_face = input_path
val_mask = input_path
img_list = os.listdir(val_face)
img_list.sort()
mask_list = os.listdir(val_mask)
mask_list.sort()
num = 2
people_num = int(len(mask_list)/num)
meta_path = model_path_s+'.meta'
if not os.path.exists(save_path):
    os.mkdir(save_path)
tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default() as g:
    with tf.Session(graph=g) as sess:
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess,model_path_s)
        out_al_nor_reco_sh = [tf.get_default_graph().get_tensor_by_name('Mul_2:0'),
#                              tf.get_default_graph().get_tensor_by_name('path_2/deconde_rec/BiasAdd_12:0'),
#                              tf.get_default_graph().get_tensor_by_name('path_2/add:0'),
                              tf.get_default_graph().get_tensor_by_name('alconv_2/c3k1/BiasAdd:0'),
                              tf.get_default_graph().get_tensor_by_name('norconv_2/nc3k1/BiasAdd:0'),
                              tf.get_default_graph().get_tensor_by_name('transpose_1:0')]
        
        
        for img_i in range((people_num)):
            img_data = cv_img(val_face+'/'+img_list[img_i*num+0])
            mask_data = cv_img(val_mask+'/'+mask_list[img_i*num+1])
            out_all= sess.run(out_al_nor_reco_sh,feed_dict={"gt_face:0":img_data*mask_data[np.newaxis,:],
                                                            'Placeholder:0':False,
                                                            'gt_mask:0':mask_data[np.newaxis,:]})
            img_al = out_all[0].astype("float32")[0]
            cache = img_list[img_i*num][0:-4]

            shading_out = out_all[3][0]
            shading_out = mask_data*shading_out
            shading_out = np.sum(shading_out, 2)/3
            if white_bk:
                cv2.imwrite(save_path+"/"+"%s_input_%s.jpg"% (cache, model),img_data*255*mask_data+ (1-mask_data)*255)

                cv2.imwrite(save_path+"/"+"%s_face_%s.jpg"% (cache, model),out_all[0][0]*255*mask_data+ (1-mask_data)*255)
                cv2.imwrite(save_path+"/"+"%s_albedo_%s.jpg"% (cache, model),out_all[1][0]*255*mask_data+ (1-mask_data)*255)
                cv2.imwrite(save_path+"/"+"%s_normal_%s.jpg"% (cache, model),out_all[2][0]*255*mask_data+ (1-mask_data)*255)
                shading_out = np.expand_dims(shading_out, axis=2)
                shading_out = np.repeat(shading_out, 3, axis=2)
                cv2.imwrite(save_path+"/"+"%s_shading_isa.jpg"%cache,shading_out*0.85*255*mask_data+ (1-mask_data)*255)
            else:
                cv2.imwrite(save_path+"/"+"%s_input_%s.jpg"% (cache, model),img_data*255*mask_data)
                cv2.imwrite(save_path+"/"+"%s_face_%s.jpg"% (cache, model),out_all[0][0]*255*mask_data)
                cv2.imwrite(save_path+"/"+"%s_albedo_%s.jpg"% (cache, model),out_all[1][0]*255*mask_data)
                cv2.imwrite(save_path+"/"+"%s_normal_%s.jpg"% (cache, model),out_all[2][0]*255*mask_data)
                cv2.imwrite(save_path+"/"+"%s_shading_%s.jpg"% (cache, model),shading_out*0.85*255)

'''
gt_face <tf.Tensor 'gt_face:0' shape=(?, 128, 128, 3) dtype=float32>
gt_mask <tf.Tensor 'gt_mask:0' shape=(?, 128, 128, 3) dtype=float32>

output
al_conv  <tf.Tensor 'alconv_2/c3k1/BiasAdd:0' shape=(?, 128, 128, 3) dtype=float32>
nor_conv <tf.Tensor 'norconv_2/nc3k1/BiasAdd:0' shape=(?, 128, 128, 3) dtype=float32>
rec      <tf.Tensor 'Mul_3:0' shape=(?, 128, 128, 3) dtype=float32>
shadow   <tf.Tensor 'path_2/deconde_rec/BiasAdd_10:0' shape=(?, 128, 128, 3) dtype=float32>
img_out  <tf.Tensor 'path_2/add:0' shape=(?, 128, 128, 3) dtype=float32>
'''        
        
        
        
        