from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import time
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import torchreid
from torchreid.utils import AverageMeter, visualize_ranked_results, save_checkpoint, re_ranking
from torchreid.losses import DeepSupervision
from torchreid import metrics


class Engine(object):
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_cpu (bool, optional): use cpu. Default is False.
    """

    def __init__(self, datamanager, model, optimizer=None, scheduler=None, use_cpu=False):
        self.datamanager = datamanager
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = (torch.cuda.is_available() and not use_cpu)

        # check attributes
        if not isinstance(self.model, nn.Module):
            raise TypeError('model must be an instance of nn.Module')

    def run(self, save_dir='log', max_epoch=0, start_epoch=0, fixbase_epoch=0, open_layers=None,
            start_eval=0, eval_freq=-1, test_only=False, print_freq=10,
            dist_metric='euclidean', normalize_feature=False, visrank=False, visrank_topk=20,
            use_metric_cuhk03=False, ranks=[1, 5, 10, 20], rerank=False):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            print_freq (int, optional): print_frequency. Default is 10.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. Visualization
                will be performed every test time, so it is recommended to enable ``visrank`` when
                ``test_only`` is True. The ranked images will be saved to
                "save_dir/ranks-epoch/dataset_name", e.g. "save_dir/ranks-60/market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 20.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
        """
        trainloader, testloader, validationloader = self.datamanager.return_dataloaders()

        if test_only:
            self.test(
                "test",
                0,
                testloader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks,
                rerank=rerank
            )
            return

        time_start = time.time()
        print('=> Start training')

        for epoch in range(start_epoch, max_epoch):
            self.train(epoch, max_epoch, trainloader, fixbase_epoch, open_layers, print_freq)
            
            if (epoch+1)>=start_eval and eval_freq>0 and (epoch+1)%eval_freq==0 and (epoch+1)!=max_epoch:
                rank1 = self.test(
                    "validation",
                    epoch,
                    validationloader,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks
                )
                self._save_checkpoint(epoch, rank1, save_dir)

        if max_epoch > 0:
            print('=> Final test')
            rank1 = self.test(
                "test",
                epoch,
                testloader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks
            )
            self._save_checkpoint(epoch, rank1, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))

    def train(self):
        r"""Performs training on source datasets for one epoch.

        This will be called every epoch in ``run()``, e.g.

        .. code-block:: python
            
            for epoch in range(start_epoch, max_epoch):
                self.train(some_arguments)

        .. note::
            
            This needs to be implemented in subclasses.
        """
        raise NotImplementedError

    def test(self, mode, epoch, testloader, dist_metric='euclidean', normalize_feature=False,
             visrank=False, visrank_topk=20, save_dir='', use_metric_cuhk03=False,
             ranks=[1, 5, 10, 20], rerank=False):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()`` when necessary.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``_extract_features()`` and ``_parse_data_for_eval()`` when necessary,
            but not a must. Please refer to the source code for more details.

        Args:
            mode (str): which mode to performing testing in ("validation" or "test")
            epoch (int): current epoch.
            testloader (dict): dictionary containing
                {dataset_name: 'query': queryloader, 'gallery': galleryloader}.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. Visualization
                will be performed every test time, so it is recommended to enable ``visrank`` when
                ``test_only`` is True. The ranked images will be saved to
                "save_dir/ranks-epoch/dataset_name", e.g. "save_dir/ranks-60/market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 20.
            save_dir (str): directory to save visualized results if ``visrank`` is True.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False.
        """
        targets = list(testloader.keys())
        
        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))

            if mode == "test":
                queryloader = testloader[name]['query']
                galleryloader = testloader[name]['gallery']

                rank1 = self._evaluate(
                    epoch,
                    dataset_name=name,
                    queryloader=queryloader,
                    galleryloader=galleryloader,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks,
                    rerank=rerank
                )

            elif mode == "validation":
                rank1 = self._evaluate_for_validation(
                    epoch,
                    dataset_name=name,
                    validationloader=testloader,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks,
                    rerank=rerank
                )
        
        return rank1

    @torch.no_grad()
    def _evaluate(self, epoch, dataset_name='', queryloader=None, galleryloader=None,
                  dist_metric='euclidean', normalize_feature=False, visrank=False,
                  visrank_topk=20, save_dir='', use_metric_cuhk03=False, ranks=[1, 5, 10, 20],
                  rerank=False):
        batch_time = AverageMeter()

        self.model.eval()

        print('Extracting features from query set ...')
        qf, q_pids, q_camids = [], [], [] # query features, query person IDs and query camera IDs
        for batch_idx, data in enumerate(queryloader):
            imgs, pids, camids = self._parse_data_for_eval(data)
            if self.use_gpu:
                imgs = imgs.cuda()
            end = time.time()
            features = self._extract_features(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids, g_camids = [], [], [] # gallery features, gallery person IDs and gallery camera IDs
        end = time.time()
        for batch_idx, data in enumerate(galleryloader):
            imgs, pids, camids = self._parse_data_for_eval(data)
            if self.use_gpu:
                imgs = imgs.cuda()
            end = time.time()
            features = self._extract_features(imgs)
            batch_time.update(time.time() - end)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print('Computing distance matrix with metric={} ...'.format(dist_metric))
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        print('Computing CMC and mAP ...')
        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03
        )

        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r-1]))

        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.return_testdataset_by_name(dataset_name),
                save_dir=osp.join(save_dir, 'visrank-'+str(epoch+1), dataset_name),
                topk=visrank_topk
            )

        return cmc[0]

    @torch.no_grad()
    def _evaluate_for_validation(self, epoch, dataset_name='', validationloader=None,
                  dist_metric='euclidean', normalize_feature=False, visrank=False,
                  visrank_topk=20, save_dir='', use_metric_cuhk03=False, ranks=[1, 5, 10, 20],
                  rerank=False):

        losses_t = AverageMeter()
        losses_x = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.eval()

        print('Checking performance on validation set ...')

        end = time.time()
        for batch_idx, data in enumerate(validationloader):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()

            outputs, features = self.model(imgs)
            loss_t = self._compute_loss(self.criterion_t, features, pids)
            loss_x = self._compute_loss(self.criterion_x, outputs, pids)
            loss = self.weight_t * loss_t + self.weight_x * loss_x

            batch_time.update(time.time() - end)

            losses_t.update(loss_t.item(), pids.size(0))
            losses_x.update(loss_x.item(), pids.size(0))
            losses.update(loss.item(), pids.size(0))
            accs.update(metrics.accuracy(outputs, pids)[0].item())

            print('Validation results:')
            # estimate remaining time
            num_batches = len(validationloader)
            print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_t {loss_t.val:.4f} ({loss_t.avg:.4f})\t'
                  'Loss_x {loss_x.val:.4f} ({loss_x.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                  'Lr {lr:.6f}'.format(
                  batch_time=batch_time,
                  data_time=data_time,
                  loss_t=losses_t,
                  loss_x=losses_x,
                  loss = losses,
                  acc=accs,
                  lr=self.optimizer.param_groups[0]['lr']
                )
            )

            end = time.time()


    def _compute_loss(self, criterion, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        return loss

    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)

    def _parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        return imgs, pids

    def _parse_data_for_eval(self, data):
        imgs = data[0]
        pids = data[1]
        camids = data[2]
        return imgs, pids, camids

    def _save_checkpoint(self, epoch, rank1, save_dir, is_best=False):
        save_checkpoint({
            'state_dict': self.model.state_dict(),
            'epoch': epoch + 1,
            'rank1': rank1,
            'optimizer': self.optimizer.state_dict(),
        }, save_dir, is_best=is_best)
