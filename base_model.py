# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from config import config
from torch.autograd import Variable
from torch.optim import Adam
from metrics import mrr_mr_hitk
from data_utils import batch_by_size
import logging
import numpy as np

import os
import sys

from copy import deepcopy


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def score(self, src, rel, dst):
        raise NotImplementedError

    def dist(self, src, rel, dst):
        raise NotImplementedError

    def prob_logit(self, src, rel, dst):
        raise NotImplementedError

    def prob(self, src, rel, dst):
        return nnf.softmax(self.prob_logit(src, rel, dst))

    def constraint(self):
        pass

    def pair_loss(self, src, rel, dst, src_bad, dst_bad, rel_smpl):
        d_good = self.dist(src, rel, dst)
        d_bad = self.dist(src_bad, rel_smpl, dst_bad,False)
        return nnf.relu(self.margin + d_good - d_bad)

    def softmax_loss(self, src, rel, dst, truth):
        probs = self.prob(src, rel, dst)
        n = probs.size(0)
        truth_probs = torch.log(probs[torch.arange(0, n).type(torch.LongTensor).cuda(), truth] + 1e-30)
        return -truth_probs
        
    def prob_loss(self,src,rel,dst,src_bad,dst_bad,rel_smpl):
        n,m=src_bad.data.shape
        one_bad_score=self.score(src_bad,rel_smpl,dst_bad,False)
        one_good_score=self.score(src,rel,dst)
        # src_cat=torch.cat((src.resize(n,1),src_bad),1)
        # rel_cat=torch.cat((rel.resize(n,1),rel_smpl),1)
        # dst_cat=torch.cat((dst.resize(n,1),dst_bad),1)
        label_truth=torch.ones(n,1)
        label_fault=torch.zeros(n,1)  # m
        label_fault-=1
        label=torch.cat((label_truth,label_fault),1)
        good_bad_scores=torch.cat((one_good_score.resize(n,1),one_bad_score.resize(n,1)),1)
        return nnf.softplus(Variable(label.cuda())*good_bad_scores)  # ,False # self.score(src_cat,rel_cat,dst_cat)


class BaseModel(object):
    def __init__(self):
        self.mdl = None # type: BaseModule
        self.weight_decay = 0

    def save(self, filename):
        torch.save(self.mdl.state_dict(), filename)

    def load(self, filename):
        self.mdl.load_state_dict(torch.load(filename, map_location=lambda storage, location: storage.cuda()))
        
    def get_tar_bad(self,s, src_bad, t, dst_bad, select):
        n,m=src_bad.size()
        tar=np.zeros((n,))
        tar_bad=np.zeros((n,m))
        result=np.zeros((n,m))
        src_bad_np=src_bad.numpy()
        dst_bad_np=dst_bad.numpy()
        
        for i in range(n):  # make target_bad matrix
            if select[i] == 1:
                tar[i]=s[i]
                tar_bad[i,:]=src_bad_np[i,:]
            else:
                tar[i]=t[i]
                tar_bad[i,:]=dst_bad_np[i,:]
                
        tar=torch.from_numpy(tar).long()
        tar_bad=torch.from_numpy(tar_bad).long()
        
        return tar.cuda(), tar_bad.cuda()

    ##################### get h_prime function ######################
    def get_gat_h_prime(self,model_gat,current_batch_2hop_indices,Corpus_, gat_entity_out_dim, gat_drop_GAT, gat_alpha, gat_nheads_GAT):
        
        entity_embed_h_prime = model_gat(Corpus_, Corpus_.train_adj_matrix, current_batch_2hop_indices)
    
        return entity_embed_h_prime
    ##################### get h_prime function ######################
    


    def gen_step(self,s,r,t, src, rel, dst,select,gat_paras,Corpus_,current_batch_2hop_indices,model_gat,opt, n_sample=15, temperature=1.0, train=True):
        
        n, m = dst.size()
        rel_var = Variable(rel.cuda())
        src_var = Variable(src.cuda())
        dst_var = Variable(dst.cuda())
        
        # get tar, tar_bad
        #tar,tar_bad=self.get_tar_bad(s,src,t,dst,select)
        #print(tar_bad)

        ##################### get h_prime ######################
        if config().g_config == "TransE":
            entity_embeddings = self.mdl.ent_embed.weight.data.cuda()
            relation_embeddings = self.mdl.rel_embed.weight.data.cuda()
        elif config().g_config == "TransD": 
            entity_embeddings = self.mdl.ent_embed.weight.data.cuda()
            relation_embeddings = self.mdl.rel_embed.weight.data.cuda()
        else:  # ComplEx
            entity_embeddings=torch.cat((self.mdl.ent_re_embed.weight.data.cuda(),self.mdl.ent_im_embed.weight.data.cuda()),dim=1)
            relation_embeddings=torch.cat((self.mdl.rel_re_embed.weight.data.cuda(),self.mdl.rel_im_embed.weight.data.cuda()),dim=1)
        
        model_gat.entity_embeddings=entity_embeddings  # every time give current embeddings to GAT
        model_gat.relation_embeddings=relation_embeddings
        model_gat.cuda()
        
        entity_embed_h_prime = self.get_gat_h_prime(model_gat,current_batch_2hop_indices,Corpus_, gat_paras['gat_entity_out_dim'], gat_paras['gat_drop_GAT'], gat_paras['gat_alpha'], gat_paras['gat_nheads_GAT'])
        
        #print("h_prime")
        #print(entity_embed_h_prime.size())     # torch.Size([40943, 50])
        #print(path_2hop_embed.size())          # torch.Size([205303, 50])
        
        #np.set_printoptions(threshold=1000000000000)
        #f=open(os.path.join(os.getcwd(),"plotdata","wn18rr"+"_h_prime.txt"),"a")
        #h_prime_plot=entity_embed_h_prime.data.cpu().numpy()
        #write_str=str(h_prime_plot)
        #f.write(write_str)
        #f.close()
        #print("write"+'\n'+"h_prime"+"to "+"plotdata/"+"wn18rr"+"_h_prime.txt")
        #sys.exit(0)
        
        ##################### get h_prime ######################

        
        
        #ori_logits = self.mdl.prob_logit(src_var, rel_var, dst_var) / temperature
        #ori_probs = nnf.softmax(ori_logits/(torch.max(ori_logits,dim=1,keepdim=True)[0]+1e-30),dim=1)
        #ori_probs+=1e-10
        gat_logits=self.mdl.prob_logit(src_var, rel_var, dst_var,gat_flag=True,h_prime=entity_embed_h_prime) / temperature
        gat_probs=nnf.softmax(gat_logits/(torch.max(gat_logits,dim=1,keepdim=True)[0]+1e-30),dim=1)   # div max
        #gat_probs+=1e-10
        
        probs = (gat_probs+1e-10)#(ori_probs+1e-10) * (gat_probs+1e-10)
        probs+=1e-10        
        #np.set_printoptions(threshold=1000000000000)
        #f=open(os.path.join(os.getcwd(),"plotdata","wn18rr"+"_logits_probs.txt"),"a")
        #ori_logits_plot=ori_logits.data.cpu().numpy()
        #f.write("ori_logits\n")
        #write_str=str(ori_logits_plot)
        #f.write(write_str)
        #f.write("\nori_probs\n")
        #ori_probs_plot=ori_probs.data.cpu().numpy()
        #write_str=str(ori_probs_plot)
        #f.write(write_str)
        #gat_logits_plot=gat_logits.data.cpu().numpy()
        #f.write("\ngat_logits\n")
        #write_str=str(gat_logits_plot)
        #f.write(write_str)
        #f.write("\ngat_probs\n")
        #gat_probs_plot=gat_probs.data.cpu().numpy()
        #write_str=str(gat_probs_plot)
        #f.write(write_str)
        #f.close()
        #print("write"+'\n'+"h_prime"+"to "+"plotdata/"+"wn18rr"+"_logits_probs.txt")
        
        
        row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
        sample_idx = torch.multinomial(probs+1e-10, n_sample, replacement=False)
        sample_srcs = src[row_idx, sample_idx.data.cpu()]
        sample_dsts = dst[row_idx, sample_idx.data.cpu()]
        
        rel_numpy=rel.numpy()
        rel_numpy=rel_numpy[:,:n_sample]
        sample_rels=torch.from_numpy(rel_numpy)
        #print(sample_rels)
        #sample_rels=rel.unsqueeze(1).expand(n,n_sample)
        
        #print()
        #print("sample_rels",sample_rels)
        #print()
        rewards = yield sample_srcs, sample_dsts, sample_rels
        if train:
            #print("in train\n")
            self.mdl.zero_grad()
            model_gat.zero_grad()
            opt.zero_grad()
            #log_probs_ori = nnf.log_softmax(ori_logits,dim=1)  #这样就可以，加了就不行  torch.log(probs)#
            log_probs_gat = nnf.log_softmax(gat_logits,dim=1)  # torch.log(gat_probs)#
                                    
            #reinforce_loss = -torch.sum(Variable(rewards) * log_probs_ori[row_idx.cuda(), sample_idx.data])
            #print("rewards",rewards)
            #print("log_probs_ori:",log_probs_ori[row_idx.cuda(), sample_idx.data])
            #print("loss_ori",reinforce_loss)
            
            #reinforce_loss= -torch.sum(Variable(rewards) * log_probs_gat[row_idx.cuda(), sample_idx.data])
            n=rewards.shape[0]
            oneshot=int(n/2)
            rl1=-torch.sum(Variable(rewards[:oneshot,:]) * log_probs_gat[row_idx[:oneshot,:].cuda(), sample_idx[:oneshot,:].data]).cuda()
            rl2=-torch.sum(Variable(rewards[oneshot:,:]) * log_probs_gat[row_idx[oneshot:,:].cuda(), sample_idx[oneshot:,:].data]).cuda()
            #rl3=-torch.sum(Variable(rewards[2*oneshot:3*oneshot,:]) * log_probs_gat[row_idx[2*oneshot:3*oneshot,:].cuda(), sample_idx[2*oneshot:3*oneshot,:].data])
            #rl4=-torch.sum(Variable(rewards[3*oneshot:,:]) * log_probs_gat[row_idx[3*oneshot:,:].cuda(), sample_idx[3*oneshot:,:].data])
            #rl5=-torch.sum(Variable(rewards[4*oneshot:5*oneshot,:]) * log_probs[row_idx[4*oneshot:5*oneshot,:].cuda(), sample_idx[4*oneshot:5*oneshot,:].data])
            #rl6=-torch.sum(Variable(rewards[5*oneshot:6*oneshot,:]) * log_probs[row_idx[5*oneshot:6*oneshot,:].cuda(), sample_idx[5*oneshot:6*oneshot,:].data])
            #rl7=-torch.sum(Variable(rewards[6*oneshot:7*oneshot,:]) * log_probs[row_idx[6*oneshot:7*oneshot,:].cuda(), sample_idx[6*oneshot:7*oneshot,:].data])
            #rl8=-torch.sum(Variable(rewards[7*oneshot:,:]) * log_probs[row_idx[7*oneshot:,:].cuda(), sample_idx[7*oneshot:,:].data])
            
            reinforce_loss=rl1+rl2#+rl3+rl4#+rl5+rl6+rl7+rl8
            
            #print("rewards",rewards)
            #print("log_probs_gat:",log_probs_gat[row_idx.cuda(), sample_idx.data])
            #print("loss:",reinforce_loss.float())  
            #print("min:",torch.min(log_probs_gat[row_idx.cuda(), sample_idx.data]))                     
            
            reinforce_loss.backward()
            opt.step()
            self.mdl.constraint()

        yield None

    def dis_step(self, src, rel, dst, src_fake, dst_fake, rel_smpl, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        src_var = Variable(src.cuda())
        rel_var = Variable(rel.cuda())
        dst_var = Variable(dst.cuda())
        src_fake_var = Variable(src_fake.cuda())
        dst_fake_var = Variable(dst_fake.cuda())
        rel_smpl_var=Variable(rel_smpl.cuda())
        
        if hasattr(self.mdl,"margin"):
            losses = self.mdl.pair_loss(src_var, rel_var, dst_var, src_fake_var, dst_fake_var, rel_smpl_var)
        else:
            losses = self.mdl.prob_loss(src_var, rel_var, dst_var, src_fake_var, dst_fake_var, rel_smpl_var)
        
        
        fake_scores = self.mdl.score(src_fake_var, rel_smpl_var, dst_fake_var) ##,False
        if train:
            self.mdl.zero_grad()
            torch.sum(losses).backward()
            self.opt.step()
            self.mdl.constraint()
        return losses.data, -fake_scores.data

    def test_link(self, test_data, n_ent, heads, tails,filt=True,write_data=True,epo=0):
        mrr_tot = 0
        mr_tot = 0
        hit10_tot = 0
        count = 0
        for batch_s, batch_r, batch_t in batch_by_size(config().test_batch_size, *test_data):
            batch_size = batch_s.size(0)
            with torch.no_grad():
                rel_var = Variable(batch_r.unsqueeze(1).expand(batch_size, n_ent).cuda())
                src_var = Variable(batch_s.unsqueeze(1).expand(batch_size, n_ent).cuda())
                dst_var = Variable(batch_t.unsqueeze(1).expand(batch_size, n_ent).cuda())
                all_var = Variable(torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent).type(torch.LongTensor).cuda())
                batch_dst_scores = self.mdl.score(src_var, rel_var, all_var).data
                batch_src_scores = self.mdl.score(all_var, rel_var, dst_var).data
                for s, r, t, dst_scores, src_scores in zip(batch_s, batch_r, batch_t, batch_dst_scores, batch_src_scores):
                    if filt:
                        if tails[(s.item(), r.item())]._nnz() > 1:
                            tmp = deepcopy(dst_scores[t])
                            dst_scores += tails[(s.item(), r.item())].cuda() * 1e30
                            dst_scores[t] = tmp
                        if heads[(t.item(), r.item())]._nnz() > 1:
                            tmp = deepcopy(src_scores[s])
                            src_scores += heads[(t.item(), r.item())].cuda() * 1e30
                            src_scores[s] = tmp
                    mrr, mr, hit10 = mrr_mr_hitk(dst_scores, t)
                    mrr_tot += mrr
                    mr_tot += mr
                    hit10_tot += hit10
                    mrr, mr, hit10 = mrr_mr_hitk(src_scores, s)
                    mrr_tot += mrr
                    mr_tot += mr
                    hit10_tot += hit10
                    count += 2
        logging.info('Test_MRR=%f, Test_MR=%f, Test_H@10=%f', mrr_tot / count, mr_tot / count, hit10_tot / count)
        
        if write_data:
            f=open(os.path.join(os.getcwd(),"plotdata",config().task.dir+".txt"),"a")
            write_str=str(mrr_tot / count)+" "+str(hit10_tot / count)+" "+str(epo)+'\n'
            f.write(write_str)
            f.close()
            print("write"+'\n'+write_str+"to "+"plotdata/"+config().task.dir+".txt")
        
        return (mrr_tot / count ,  hit10_tot / count)
