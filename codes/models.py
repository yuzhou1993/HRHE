import os
import logging
import numpy as np
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import BatchType, TestDataset

from get_relation import Graph_Attention

import json

"""
auther:xufe
time:2024.7.7 12:34
tasK:获取e3关系，即h_img_emb和tail的关系，其中h的关系权重初始化为图像数据初始化，t初始化为openke权重参数初始化。
input:h_img_emb,tail
output:h_img_emb和tail的关系r_3

"""


class KGEModel(nn.Module, ABC):
    """
    Must define
        `self.entity_embedding`
        `self.relation_embedding`
    in the subclasses.
    """

    @abstractmethod
    def func(self, head, rel, tail, batch_type):
        """
        Different tensor shape for different batch types.
        BatchType.SINGLE:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.HEAD_BATCH:
            head: [batch_size, negative_sample_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.TAIL_BATCH:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, negative_sample_size, hidden_dim]
        """
        ...

    def forward(self, sample, batch_type=BatchType.SINGLE):
        """
        Given the indexes in `sample`, extract the corresponding embeddings,
        and call func().

        Args:
            batch_type: {SINGLE, HEAD_BATCH, TAIL_BATCH},
                - SINGLE: positive samples in training, and all samples in validation / testing,
                - HEAD_BATCH: (?, r, t) tasks in training,
                - TAIL_BATCH: (h, r, ?) tasks in training.

            sample: different format for different batch types.
                - SINGLE: tensor with shape [batch_size, 3]
                - {HEAD_BATCH, TAIL_BATCH}: (positive_sample, negative_sample)
                    - positive_sample: tensor with shape [batch_size, 3]
                    - negative_sample: tensor with shape [batch_size, negative_sample_size]
        """
        if batch_type == BatchType.SINGLE:

            h_img = sample[:, 0].detach()
            t_img = sample[:, 2].detach()
            batch_size = sample.size(0)
            h_img = h_img.long()
            t_img = t_img.long()
            h_img_emb = self.img_proj(self.img_embeddings(h_img)).unsqueeze(1)

            t_img_emb = self.img_proj(self.img_embeddings(t_img)).unsqueeze(1)

            tail_rel = torch.index_select(
                self.entity_embedding_random,
                dim=0,
                index=sample[:, 2]
            )
            head_rel = torch.index_select(
                self.entity_embedding_random,
                dim=0,
                index=sample[:, 0]
            )


            # head = self.target_parameters['ent_embeddings.weight'][sample[:, 0]].unsqueeze(1)
            # relation = self.target_parameters['rel_embeddings.weight'][sample[:, 1]].unsqueeze(1)
            # tail = self.target_parameters['ent_embeddings.weight'][sample[:, 2]].unsqueeze(1)

            head = torch.index_select(
                self.entity_embedding_random,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding_random,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding_random,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
            self.get_realtion = Graph_Attention(500, 39, 0.3,  img_dim=batch_size, tail_rel=tail_rel).cuda()
            r_e3 = self.get_realtion(h_img_emb.squeeze(), tail_rel, relation.squeeze()).unsqueeze(1)
            r_e4 = self.get_realtion(h_img_emb.squeeze(), t_img_emb.squeeze(), relation.squeeze()).unsqueeze(1)
            r_e2 = self.get_realtion(head_rel, t_img_emb.squeeze(), relation.squeeze()).unsqueeze(1)


        elif batch_type == BatchType.HEAD_BATCH:
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            t_img = tail_part[:, 2].detach()
            # h_img = h_img.long()
            t_img = t_img.long()
            h_img_emb = self.img_proj(self.img_embeddings(head_part.view(-1).long())).view(batch_size,
                                                                                           negative_sample_size, -1)
            t_img_emb = self.img_proj(self.img_embeddings(t_img)).unsqueeze(1)

            tail_rel = torch.index_select(
                self.entity_embedding_random,
                dim=0,
                index=tail_part[:, 2]
            )
            head_rel = torch.index_select(
                self.entity_embedding_random,
                dim=0,
                index=tail_part[:, 0]
            )

            h_img_rel = self.img_proj(self.img_embeddings(tail_part[:, 0].long()))



            # tail = self.target_parameters['ent_embeddings.weight'][tail_part[:, 2]].unsqueeze(1)
            # head = self.target_parameters['ent_embeddings.weight'][head_part.view(-1).long()].view(batch_size,
            #                                                                                        negative_sample_size,
            #                                                                                        -1)
            # relation = self.target_parameters['rel_embeddings.weight'][tail_part[:, 1]].unsqueeze(1)

            head = torch.index_select(
                self.entity_embedding_random,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding_random,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding_random,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)
            self.get_realtion = Graph_Attention(500, 39, 0.3,  img_dim=batch_size, tail_rel=tail_rel).cuda()
            r_e3 = self.get_realtion(h_img_rel, tail_rel, relation.squeeze()).unsqueeze(1)
            r_e4 = self.get_realtion(h_img_rel, t_img_emb.squeeze(), relation.squeeze()).unsqueeze(1)
            r_e2 = self.get_realtion(head_rel, t_img_emb.squeeze(), relation.squeeze()).unsqueeze(1)


        elif batch_type == BatchType.TAIL_BATCH:

            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            h_img = head_part[:, 0].detach()
            # t_img = tail_part.detach()
            h_img = h_img.long()
            # t_img = t_img.long()
            h_img_emb = self.img_proj(self.img_embeddings(h_img)).unsqueeze(1)

            tail_rel = torch.index_select(
                self.entity_embedding_random,
                dim=0,
                index=head_part[:, 2]
            )

            head_rel = torch.index_select(
                self.entity_embedding_random,
                dim=0,
                index=head_part[:, 0]
            )
            t_img_rel = self.img_proj(self.img_embeddings(head_part[:, 2].long()))



            t_img_emb = self.img_proj(self.img_embeddings(tail_part.view(-1))).view(batch_size, negative_sample_size,
                                                                                    -1)

            # tail_try = tail_part.view(-1)
            # head = self.target_parameters['ent_embeddings.weight'][head_part[:, 0]].unsqueeze(1)
            # tail = self.target_parameters['ent_embeddings.weight'][tail_part.view(-1).long()].view(batch_size, negative_sample_size, -1)
            # relation = self.target_parameters['rel_embeddings.weight'][head_part[:, 1]].unsqueeze(1)

            # tail = self.jg_ent_embeddings(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)
            # relation = self.jg_rel_embeddings(head_part[:, 1]).unsqueeze(1)
            # head = self.jg_ent_embeddings(h_img).unsqueeze(1)

            head = torch.index_select(
                self.entity_embedding_random,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding_random,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding_random,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            self.get_realtion = Graph_Attention(500, 39, 0.3,  img_dim=batch_size, tail_rel=tail_rel).cuda()
            r_e3 = self.get_realtion(h_img_emb.squeeze(), tail_rel, relation.squeeze()).unsqueeze(1)
            r_e4 = self.get_realtion(h_img_emb.squeeze(), t_img_rel, relation.squeeze()).unsqueeze(1)
            r_e2 = self.get_realtion(head_rel, t_img_rel, relation.squeeze()).unsqueeze(1)


        else:
            raise ValueError('batch_type %s not supported!'.format(batch_type))

        # return scores
        return self.func(head, relation, tail, batch_type, h_img_emb, t_img_emb, r_e2, r_e3, r_e4)

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)

        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        subsampling_weight = subsampling_weight.cuda()

        # negative scores
        negative_score, rel_n, re_2_n, re_3_n, re_4_n = model((positive_sample, negative_sample), batch_type=batch_type)

        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        # positive scores
        positive_score, rel_p, re_2_p, re_3_p, re_4_p = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
        rel_1 = torch.norm(rel_p - re_2_p, p=2, dim=1)
        rel_2 = torch.norm(rel_p - re_3_p, p=2, dim=1)
        rel_3 = torch.norm(rel_p - re_4_p, p=2, dim=1)
        rel_p = F.logsigmoid(rel_p).squeeze(dim=1)
        relp_1_score = F.logsigmoid(rel_1)
        relp_2_score = F.logsigmoid(rel_2)
        relp_3_score = F.logsigmoid(rel_3)


        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        relp_1 = - (subsampling_weight * relp_1_score).sum() / subsampling_weight.sum()
        relp_2 = - (subsampling_weight * relp_2_score).sum() / subsampling_weight.sum()
        relp_3 = - (subsampling_weight * relp_3_score).sum() / subsampling_weight.sum()
        # positive_sample_loss += torch.sqrt(torch.sum(torch.pow(rel_p - re_2_p, 2)), p=2)+ torch.norm(torch.sqrt((torch.sum(torch.pow(rel_p - re_3_p, 2)))), p=2)  + torch.norm(torch.sqrt(torch.sum(torch.pow(rel_p - re_4_p, 2))), p=2)
        #positive_sample_loss += torch.norm(rel_p - re_2_p, p=2, dim=1) + torch.norm(rel_p - re_3_p, p=2,dim=1) + torch.norm( rel_p - re_4_p, p=2,dim=1)
        positive_sample_loss = positive_sample_loss + relp_1 + relp_2 + relp_3
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        loss = (positive_sample_loss + negative_sample_loss) / 2
        #loss +=

        loss.backward()

        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, data_reader, mode, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        test_dataloader_head = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.HEAD_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.TAIL_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, batch_type in test_dataset:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score, rel_P, re_2_p, re_3_p, re_4_p = model((positive_sample, negative_sample), batch_type)
                    score += filter_bias

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if batch_type == BatchType.HEAD_BATCH:
                        positive_arg = positive_sample[:, 0]
                    elif batch_type == BatchType.TAIL_BATCH:
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... ({}/{})'.format(step, total_steps))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics


class ModE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, img_dim, img_emb, pretrained_file=None):
        super(ModE, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.img_dim = img_dim
        self.img_proj = nn.Linear(self.img_dim, self.hidden_dim)
        # self.jg_proj = nn.Linear(300, self.hidden_dim)
        self.img_embeddings = img_emb
        self.img_embeddings.requires_grad = False

        # self.jg_ent_embeddings = nn.Embedding(num_entity, hidden_dim)
        # self.jg_rel_embeddings = nn.Embedding(num_relation, hidden_dim)

        # self.target_parameters = self.load_parameters()

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        # self.entity_embedding_openke = nn.Parameter(self.target_parameters["ent_embeddings.weight"])
        # self.relation_embedding_openke = nn.Parameter(self.target_parameters["rel_embeddings.weight"])

        self.entity_embedding_random = nn.Parameter(torch.zeros(num_entity, hidden_dim))
        # self.entity_embedding.data.clamp_(-self.embedding_range, self.embedding_range)

        nn.init.uniform_(
            tensor=self.entity_embedding_random,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding_random = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        # self.relation_embedding.data.clamp_(-self.embedding_range, self.embedding_range)
        nn.init.uniform_(
            tensor=self.relation_embedding_random,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        # print()

    # def load_parameters(self):
    #     f = open('./embed_300d.vec', "r")
    #     parameters = json.loads(f.read())
    #     f.close()
    #     for i in parameters:
    #         parameters[i] = torch.Tensor(parameters[i]).cpu()
    #     return parameters
    def func(self, head, rel, tail, batch_type, h_img_emb, t_img_emb,r_e2, r_e3, r_e4 ):
        return self.gamma.item() - (torch.norm(head * rel - tail, p=1, dim=2) +
                                 torch.norm(head * r_e2- t_img_emb, p=2, dim=2)+
                                 torch.norm(h_img_emb * r_e3 - tail, p=2, dim=2)+
                                  torch.norm(h_img_emb * r_e4- t_img_emb, p=2, dim=2)), rel.squeeze(), r_e2.squeeze(), r_e3.squeeze(), r_e4.squeeze()
        


class HAKE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, modulus_weight=1.0, phase_weight=0.5):
        super(HAKE, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim * 3))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        nn.init.ones_(
            tensor=self.relation_embedding[:, hidden_dim:2 * hidden_dim]
        )

        nn.init.zeros_(
            tensor=self.relation_embedding[:, 2 * hidden_dim:3 * hidden_dim]
        )

        self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))

        self.pi = 3.14159262358979323846

    def func(self, head, rel, tail, batch_type):
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        if batch_type == BatchType.HEAD_BATCH:
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight

        return self.gamma.item() - (phase_score + r_score)
