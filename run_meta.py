

""" Generate commands for meta-train phase. """
import os

def run_exp(num_batch=50, shot=3, teshot=1, query=1, lr1=0.0005, lr2=0.005, base_lr=0.01, update_step=20, gamma=0.5):
    max_epoch = 200
    step_size = 20
    way=2 #Backround as a class included. Adjust accordingly.
    gpu=1
       
    the_command = 'python3 main.py' \
        + ' --max_epoch=' + str(max_epoch) \
        + ' --num_batch=' + str(num_batch) \
        + ' --train_query=' + str(query) \
        + ' --meta_lr1=' + str(lr1) \
        + ' --meta_lr2=' + str(lr2) \
        + ' --step_size=' + str(step_size) \
        + ' --gamma=' + str(gamma) \
        + ' --gpu=' + str(gpu) \
        + ' --base_lr=' + str(base_lr) \
        + ' --update_step=' + str(update_step) \
        + ' --way=' + str(way) 

    os.system(the_command + ' --phase=meta_train')
    os.system(the_command + ' --phase=meta_eval')

run_exp(num_batch=50, shot=3, teshot=1, query=1, lr1=0.0005, lr2=0.005, base_lr=0.01, update_step=20, gamma=0.5)
#run_exp(num_batch=100, shot=5, query=15, lr1=0.0001, lr2=0.001, base_lr=0.01, update_step=100, gamma=0.5)
