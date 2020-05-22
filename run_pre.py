

""" Generate commands for pre-train phase. """
import os

def run_exp(lr=0.1, gamma=0.2, step_size=20):
    max_epoch = 200
    num_classes=5
    query = 1
    gpu = 1
    way=2
    shot =3
    teshot=1
    base_lr = 0.05
    #pre_init_weights='../logs/pre/COCO_UNet_batchsize8_lr0.1_gamma0.2_step20_maxepoch200/epoch110.pth'
    
    the_command = 'python3 main.py' \
        + ' --pre_max_epoch=' + str(max_epoch) \
        + ' --num_classes=' + str(num_classes) \
        + ' --shot=' + str(shot) \
        + ' --teshot=' + str(teshot) \
        + ' --train_query=' + str(query) \
        + ' --pre_step_size=' + str(step_size) \
        + ' --pre_gamma=' + str(gamma) \
        + ' --gpu=' + str(gpu) \
        + ' --base_lr=' + str(base_lr) \
        + ' --pre_lr=' + str(lr) \
        + ' --phase=pre_train' \
        + ' --way=' + str(way) 
        #+ ' --pre_init_weights=' + pre_init_weights

    os.system(the_command)

run_exp(lr=0.1, gamma=0.2, step_size=20)
