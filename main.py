import random,pprint,sys,time,numpy as np,pickle
from keras import backend as K
from keras.optimizers import Adam,RMSprop
from keras.layers import Input
from keras.models import Model
from model import config,data_generators,data_augment
from model import losses,resnet
from keras.utils import generic_utils
from model.process_data import get_data
import model
def data_aug(train_path,num_rois,hf=True,vf=False,
          rot_90=False,num_epoches=2000,
          output_weight_path='weights/{}faster_rcnn.hdf5'.format(time.localtime(time.time()))):
    C = config.Config()
    C.num_rois = int(num_rois)
    C.use_horizontal_flips = hf
    C.use_vertical_flips = vf
    C.rot_90 = rot_90
    C.model_path = output_weight_path
    all_imgs, class_count, class_mapping = get_data(train_path)
    for img_data in all_imgs:
        img_aug,img=data_augment.augment(img_data,C,augment=True)
        if img_aug['filepath']!=img_data['filepath']:
            all_imgs.append(img_aug)
        else:
            pass

def main(train_path,num_rois,hf=True,vf=False,
          rot_90=False,num_epoches=2000,
          output_weight_path='weights/{}faster_rcnn.hdf5'.format(time.localtime(time.time()))):
    C=config.Config()
    C.num_rois=int(num_rois)
    C.use_horizontal_flips=hf
    C.use_vertical_flips=vf
    C.rot_90=rot_90
    C.model_path=output_weight_path
    all_imgs,class_count,class_mapping=get_data(train_path)
    if 'bg' not in class_count:
        class_count['bg']=0
        class_mapping['bg']=len(class_mapping)
    C.class_mapping=class_mapping
    inv_map={v:k for k,v in class_mapping.items()}
    print('classes_count:',class_count)
    print('class_mapping',class_mapping)
    train_imgs=[s for s in all_imgs if s['type']=='train']
    val_imgs=[s for s in all_imgs if s['type']=='val']
    test_imgs=[s for s in all_imgs if s['type']=='test']
    print("the number of train",len(train_imgs))
    print('the number of val',len(val_imgs))
    print('the number of test',len(test_imgs))
    data_gen_train=data_generators.get_anchor_gt(train_imgs,class_count,C,K.image_dim_ordering(),mode='train')
    data_gen_val=data_generators.get_anchor_gt(val_imgs,class_count,C,K.image_dim_ordering(),mode='val')
    input_shape_img=(None,None,3)
    img_input=Input(shape=input_shape_img)
    roi_input=Input(shape=(C.num_rois,4))

    shared_layers=resnet.ResNet50(img_input,trainable=True)

    #RPN built on the base layers
    num_anchors=len(C.anchor_box_scales)*len(C.anchor_box_ratios)
    rpn=resnet.rpn(shared_layers,num_anchors)
    classifier=resnet.classifier(shared_layers,roi_input,C.num_rois,
                                 nb_classes=len(class_count),trainable=True)
    model_rpn=Model(img_input,rpn[:2])
    model_classifier=Model([img_input,roi_input],classifier)
    model_all=Model([img_input,roi_input],rpn[:2]+classifier)

    try:
        print('loading weights')
        model_rpn.load_weights(C.base_net_weight)
        model_classifier.load_weights(C.base_net_weight)
    except Exception as e:
        print(e)
    optimizer=Adam(lr=1e-3)
    optimizer_classifier=Adam(lr=1e-3)
    model_rpn.compile(optimizer=optimizer,loss=[model.losses.rpn_loss_cls(num_anchors), model.losses.rpn_loss_regr(num_anchors)])
    model_classifier.compile(
        optimizer=optimizer_classifier,
        loss=[model.losses.class_loss_cls, model.losses.class_loss_regr(len(class_count)-1)],
        metrics={'dense_class_{}'.format(len(class_count)): 'accuracy'}
    )
    model_all.compile(optimizer='sgd',loss='mae')

    epoch_length=1000
    iter_num=0

    losses=np.zeros((epoch_length,5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time=time.time()
    best_loss=np.Inf

    class_mapping_inv=inv_map
    print('start training')
    for epoch_num in range(num_epoches):
        progbar=generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num + 1, num_epoches))
        while True:
            try:
                if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print(
                        'Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                            mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print(
                            'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                X, Y, img_data = data_gen_train.next()

                loss_rpn = model_rpn.train_on_batch(X, Y)

                P_rpn = model_rpn.predict_on_batch(X)

                R = model.roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True,
                                           overlap_thresh=0.7, max_boxes=300)

                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2 = model.roi_helpers.calc_iou(R, img_data, C, class_mapping)

                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if C.num_rois > 1:
                    if len(pos_samples) < C.num_rois / 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, C.num_rois / 2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                replace=True).tolist()

                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                             [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                iter_num += 1

                progbar.update(iter_num,
                               [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                ('detector_cls', np.mean(losses[:iter_num, 2])),
                                ('detector_regr', np.mean(losses[:iter_num, 0]))])

                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if C.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                            mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    if curr_loss < best_loss:
                        if C.verbose:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                        best_loss = curr_loss
                        model_all.save_weights(C.model_path)

                    break

            except Exception as e:
                print('Exception: {}'.format(e))
                continue

        print('Training complete, exiting.')

if __name__=="__main__":
    main(train_path="point_line",num_rois=32)



