from utils import all_gather_if_needed

from neuronal_pool import min_max_normalization
import torch
import torch.nn.functional as F
import numpy as np
from neuronal_pool import cacu_bandit_reward
import time

def bandit_scoring(_f1,_f2,h_chosen_cls,f1_only=True,f2_weight=0.1):
    scores = [] 
    indices = [] 
    f1_output=list(_f1) # batchsize,
    f2_output=list(_f2)
    # caculate the scores
    for i in range(h_chosen_cls.shape[0]):
        f1=_f1[i].item() 
        f2=_f2[i].item() 
        if f1_only:
            u=f1
        else:
            u = f1 + f2*f2_weight
        scores.append(u)
        indices.append(i)
    
    return scores,indices,f1_output,f2_output


def get_observed_rewards(cur_metrics,pre_metrics,
                         r_rm=None,r_rewards=None,r_logps=None,
                         loss_name="dpo",
                         construct_reward=True,
                         reward_type="add_rm_logp",
                         norm_metrics=True,
                         avg_reward=None,
                         counter=None,
                         avg_weight=0.5,
                         enable_avg=None):
    # batch_level_reward = cur_metrics-pre_metrics
    batch_level_reward = -cur_metrics+pre_metrics
    if norm_metrics:
        batch_level_reward=batch_level_reward/max(cur_metrics,pre_metrics)

    # enable avg weight
    if counter>0 and enable_avg:
        batch_level_reward=avg_reward*avg_weight+batch_level_reward*(1-avg_weight)
        # update avg_reward update counter
        
    counter+=1
    avg_reward=avg_reward+1/counter*(batch_level_reward-avg_reward)



    if loss_name in {'dpo', 'ipo'}:
        y1=[]
        if not construct_reward:
            weights=F.softmax(torch.tensor(r_rm), dim=0)
            for w in weights:
                y1.append(w*batch_level_reward)  

        else: # construct bandit loss
            for rm,r,p in zip(r_rm,r_rewards,r_logps):
                y1.append(cacu_bandit_reward(batch_level_reward,rm,r,p,reward_type))
    
    elif loss_name in {'sft'}:
        y1=[]
        for rm,p in zip(r_rm,r_logps):
            y1.append(cacu_bandit_reward(batch_level_reward,rm,None,p,reward_type))

    return y1,avg_reward,counter



def sampling_based_on_scores(scores,indices,selected_ratio=0.5):
    distribution = []
    for x in scores:
        distribution.append(x)
    input_tensor = torch.tensor(distribution)
    output_tensor = F.softmax(input_tensor, dim=0)
    distribution = output_tensor.tolist()
    total = sum(distribution)
    distribution = [w/total for w in distribution] 
    selected_batchsize=int(selected_ratio*len(distribution))
    selected_sample_ids = np.random.choice(indices, size=selected_batchsize, replace=False, p=distribution)
    selected_sample_ids=torch.from_numpy(selected_sample_ids)
    return selected_sample_ids


def add_bandit_metrics(batch_metrics,f1_loss_list,f2_loss_list,y1,y2,f1_output,f2_output,final_output):
    batch_metrics['f1_loss'].append(torch.tensor(f1_loss_list).mean().item())
    batch_metrics['f2_loss'].append(torch.tensor(f2_loss_list).mean().item())
    batch_metrics['f1_metrics'].append(torch.tensor(y1).mean().item())
    batch_metrics['f2_metrics'].append(torch.tensor(y2).mean().item())
    batch_metrics['estimate_error'].append(torch.tensor([abs(f1_output[i]-y1[i]) for i in range(len(y1))]).mean().item())
    batch_metrics['f2_estimate_error'].append(torch.tensor([abs(f2_output[i]-y2[i]) for i in range(len(y2))]).mean().item())
    batch_metrics['final_estimate_error'].append(torch.tensor([abs(final_output[i]-y1[i]) for i in range(len(y1))]).mean().item())


def process_bandit_train_data(selected_sample_ids,
                              h_chosen_cls,attention_mask,X2_train,
                              f1_output,f2_output,scores,
                              _r=None,_logp=None,_rm=None,
                              loss_name=None,
                              use_all_samples=None):
    
    X1_train=h_chosen_cls
    attention_mask=attention_mask 
    X2_train=torch.cat(X2_train,dim=0)
    f1_output=torch.tensor(f1_output)
    f2_output=torch.tensor(f2_output)
    final_output=torch.tensor(scores)
    if loss_name in {'dpo','ipo'}:
        r_rewards=min_max_normalization(_r)
        r_logps=min_max_normalization(_logp)
        reward_margins=min_max_normalization(_rm)

    elif loss_name in {'sft'}: 
        reward_margins=min_max_normalization(_rm)
        r_logps=min_max_normalization(_logp)

    if not use_all_samples:
        X1_train=X1_train[selected_sample_ids]
        attention_mask=attention_mask[selected_sample_ids] if attention_mask is not None else None
        X2_train=X2_train[selected_sample_ids]
        f1_output=f1_output[selected_sample_ids]
        f2_output=f2_output[selected_sample_ids]
        final_output=final_output[selected_sample_ids]

        if loss_name in {'dpo','ipo'}:
            r_rewards=r_rewards[selected_sample_ids]
            r_logps=r_logps[selected_sample_ids]
            reward_margins=reward_margins[selected_sample_ids]

        elif loss_name in {'sft'}:
            reward_margins=reward_margins[selected_sample_ids]
            r_logps=r_logps[selected_sample_ids]
    
    X1_train=list(X1_train)
    attention_mask=list(attention_mask) if attention_mask is not None else None
    X2_train=list(X2_train)
    f1_output=list(f1_output)
    f2_output=list(f2_output)
    final_output=list(final_output)

    if loss_name in {'dpo','ipo'}:
        r_rewards=list(r_rewards)
        r_logps=list(r_logps)
        reward_margins=list(reward_margins)
        return (X1_train,attention_mask,X2_train,f1_output,f2_output,final_output,r_rewards,r_logps,reward_margins)
    
    elif loss_name in {'sft'}:
        reward_margins=list(reward_margins)
        r_logps=list(r_logps)
        return (X1_train,attention_mask,X2_train,f1_output,f2_output,final_output,r_logps,reward_margins)

    



def bandit_data_preprocess(local_bandit_inputs,rank,world_size,loss_name):
    if loss_name in {'dpo', 'ipo'}:
        (_rm,_logp,_r)=local_bandit_inputs[:3]

        local_losses=local_bandit_inputs[-2]

        cur_bandit_data=[]
        for item in local_bandit_inputs[3:]: 
            cur_bandit_data.append(all_gather_if_needed(item.detach(),rank,world_size))
        # global bandit data
        (h_chosen_cls,_attention_mask,global_losses,global_sig_losses)=tuple(cur_bandit_data)
        return (_rm,_logp,_r,local_losses,h_chosen_cls,_attention_mask,global_losses,global_sig_losses)
    
    elif loss_name in {'sft'}:
        _logp=local_bandit_inputs[0]
        local_losses=local_bandit_inputs[3]
        h_chosen_cls=all_gather_if_needed(local_bandit_inputs[1].detach(),rank,world_size)
        _attention_mask=all_gather_if_needed(local_bandit_inputs[2].detach(),rank,world_size)
        global_losses=all_gather_if_needed(local_bandit_inputs[3].detach(),rank,world_size)

        return (_logp,h_chosen_cls,_attention_mask,local_losses,global_losses)


def detach_unselected_samples(local_losses,selected_sample_ids,rank):
    local_batch_num=local_losses.shape[0]
    local_selected_idx=[]
    for idx in range(local_batch_num):
        mapped_idx=local_batch_num*rank+idx
        if mapped_idx not in list(selected_sample_ids): 
            local_losses[idx]=local_losses[idx].detach() 
        else:
            local_selected_idx.append(idx)

    print(f"rank:{rank},backward {len(local_selected_idx)} samples")
