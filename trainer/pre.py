

""" Trainer for pretrain phase. """
import os.path as osp
import os
import tqdm
import torch
import numpy as np
from utils.metrics import eval_metrics
from torch.utils.data import DataLoader
from models.mtl import MtlLearner
from tensorboardX import SummaryWriter
from dataloader.samplers import CategoriesSampler 
from utils.losses import FocalLoss,CE_DiceLoss,LovaszSoftmax
from utils.misc import Averager, Timer, ensure_path
from dataloader.dataset_loader import DatasetLoader as Dataset
from dataloader.mdataset_loader import mDatasetLoader as mDataset

#torch.cuda.set_device(1)
class PreTrainer(object):

    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = '../logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'pre')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type])
        save_path2 = 'batchsize' + str(args.pre_batch_size) + '_lr' + str(args.pre_lr) + '_gamma' + str(args.pre_gamma) + '_step' + \
            str(args.pre_step_size) + '_maxepoch' + str(args.pre_max_epoch)
        args.save_path = pre_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        # Load pretrain set
        self.trainset = Dataset('train', self.args)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size, shuffle=True, num_workers=8, pin_memory=True)

        # Load pre-val set
        self.valset = mDataset('val', self.args)
        self.val_sampler = CategoriesSampler(self.valset.labeln,self.args.num_batch , self.args.way, self.args.shot + self.args.val_query,self.args.shot)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8, pin_memory=True)


        # Build pretrain model
        self.model = MtlLearner(self.args, mode='train')
        print(self.model)
        
        '''
        if self.args.pre_init_weights is not None:
            self.model_dict = self.model.state_dict()
            pretrained_dict = torch.load(self.args.pre_init_weights)['params']
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.model_dict}
            print(pretrained_dict.keys())
            self.model_dict.update(pretrained_dict)
            self.model.load_state_dict(self.model_dict)   
        '''
        
        self.FL=FocalLoss()
        self.CD=CE_DiceLoss()
        self.LS=LovaszSoftmax()
        # Set optimizer 
        # Set optimizer 
        self.optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': self.args.pre_lr}], \
                momentum=self.args.pre_custom_momentum, nesterov=True, weight_decay=self.args.pre_custom_weight_decay)

            # Set learning rate scheduler 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.pre_step_size, \
            gamma=self.args.pre_gamma)        

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """  
        torch.save(dict(params=self.model.encoder.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def _reset_metrics(self):
        #self.batch_time = AverageMeter()
        #self.data_time = AverageMeter()
        #self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self,n_class):
        self.n_class=n_class
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.n_class), np.round(IoU, 3)))
        }

    def train(self):
        """The function for the pre-train phase."""

        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['train_iou']=[]
        trlog['val_iou']=[]
        trlog['max_iou'] = 0.0
        trlog['max_iou_epoch'] = 0

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)

        # Start pretrain
        for epoch in range(1, self.args.pre_max_epoch + 1):
            # Update learning rate
            self.lr_scheduler.step()
            # Set the model to train mode
            self.model.train()
            self.model.mode = 'train'
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()
            train_iou_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)

            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number 
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, label = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                    label = batch[1]

                # Output logits for model
                logits = self.model(data)
                # Calculate train loss
                # CD loss is modified in the whole project to incorporate ony Cross Entropy loss. Modify as per requirement.
                #loss = self.FL(logits, label) + self.CD(logits,label) + self.LS(logits,label)
                loss = self.CD(logits,label) 
                
                # Calculate train accuracy
                self._reset_metrics()
                seg_metrics = eval_metrics(logits, label, self.args.num_classes)
                self._update_seg_metrics(*seg_metrics)
                pixAcc, mIoU, _ = self._get_seg_metrics(self.args.num_classes).values()

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(pixAcc)
                train_iou_averager.add(mIoU)

                # Print loss and accuracy till this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f} IOU={:.4f}'.format(epoch, train_loss_averager.item(),train_acc_averager.item()*100.0,train_iou_averager.item()))
                
                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()
            train_iou_averager = train_iou_averager.item()

            writer.add_scalar('data/train_loss(Pre)', float(train_loss_averager), epoch)
            writer.add_scalar('data/train_acc(Pre)', float(train_acc_averager)*100.0, epoch) 
            writer.add_scalar('data/train_iou (Pre)', float(train_iou_averager), epoch)
            
            print('Epoch {}, Train: Loss={:.4f}, Acc={:.4f}, IoU={:.4f}'.format(epoch, train_loss_averager, train_acc_averager*100.0,train_iou_averager))        
            
            # Start validation for this epoch, set model to eval mode
            self.model.eval()
            self.model.mode = 'val'

            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()
            val_iou_averager = Averager()

            # Print previous information  
            if epoch % 1 == 0:
                print('Best Val Epoch {}, Best Val IoU={:.4f}'.format(trlog['max_iou_epoch'], trlog['max_iou']))

            # Run validation
            for i, batch in enumerate(self.val_loader, 1):
                if torch.cuda.is_available():
                    data, labels,_ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                    label=labels[0]
                p = self.args.way*self.args.shot
                data_shot, data_query = data[:p], data[p:]
                label_shot,label=labels[:p],labels[p:]
                
                par=data_shot, label_shot, data_query
                logits = self.model(par)
                # Calculate preval loss
                
                #loss = self.FL(logits, label) + self.CD(logits,label) + self.LS(logits,label)
                loss = self.CD(logits,label)                 
                
                # Calculate val accuracy
                self._reset_metrics()
                seg_metrics = eval_metrics(logits, label, self.args.way)
                self._update_seg_metrics(*seg_metrics)
                pixAcc, mIoU, _ = self._get_seg_metrics(self.args.way).values()

                val_loss_averager.add(loss.item())
                val_acc_averager.add(pixAcc)
                val_iou_averager.add(mIoU)  

            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            val_iou_averager = val_iou_averager.item()
            
            writer.add_scalar('data/val_loss(Pre)', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc(Pre)', float(val_acc_averager)*100.0, epoch) 
            writer.add_scalar('data/val_iou (Pre)', float(val_iou_averager), epoch)  
            
            # Print loss and accuracy for this epoch
            print('Epoch {}, Val: Loss={:.4f} Acc={:.4f} IoU={:.4f}'.format(epoch, val_loss_averager, val_acc_averager*100.0,val_iou_averager))

            # Update best saved model
            if val_iou_averager > trlog['max_iou']:
                trlog['max_iou'] = val_iou_averager
                trlog['max_iou_epoch'] = epoch
                print("model saved in max_iou")
                self.save_model('max_iou')

            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('epoch'+str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)
            trlog['train_iou'].append(train_iou_averager)
            trlog['val_iou'].append(val_iou_averager)

            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 1 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.max_epoch)))
        writer.close()
