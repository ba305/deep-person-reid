Torchreid--for training re-ID models
===========

Note from Ben: 
^^^^^^^^^^^^^^^^^^^^^^^^
If you scroll down for a while, to the next major section, you will find the original README provided by Kaiyang Zhou. I've made a number of changes to the original repo, but I decided to leave the original README contents below, since they are still useful. However, in this first section, I am including some of my own comments, so that I can explain what I changed, and how the repo is now different than the original.

**Main changes implemented by Ben**:

- Added a validation set
- New learning rate scheduler (ReduceLROnPLateau)
- Early stopping (based on validation loss)
- Converted argparse flags into a yaml config file
- Save results to CSV file (called "history.csv")
- Save loss curves at the end of training
- Tensorboard support
- Warmup period for LR scheduler, early stopping, and saving best model
- *Note that because of the changes above, some other parts of this repo are now deprecated. See below for further details.*

**KEY POINTS (PLEASE read this section before using this repo!!!!!)**:

- While modifying the repo, I only modified the code that is relevant to image analysis (as opposed to video analysis). For my project, I did not need this particular video code (since the video methods in this repo use tracklets, which was not relevant for my particular task), so I did not update the video code. So, do not use the video scripts unless you make all of the necessary changes.
- Likewise, this repo now only works for the triplet loss scripts, NOT the softmax ones. You would have to add all the changes I made in triplet.py (and related files) to softmax.py (and related files). However, rather than doing that, I believe you can also continue using the triplet script, but set weight_t to 0 and weight_x to 1 in the config file, which technically calculates both softmax and triplet loss, but then just weights the triplet loss as 0, effectively ignoring it (I haven't tested this out, but I believe it should work that way). As I will explain below, the validation set only works when triplet loss is used, so I haven't been using the softmax setting (you probably shouldn't either, unless you DO NOT WANT any validation results), and thus did not update that code.
- To summarize the two bullet points above: in the config file, leave the top two settings ('app' and 'loss') as they are currently set ('image' and 'triplet'), unless you modify other files to bring everything up to date.
- I have also marked some folders and/or files as "DEPRECATED" in the file name, in which case they are deprecated and would need to be udpated if you still wanted to use them.

What else has become deprecated?

- Because I added the validation set, it may have interfered with some of the other features in the original repo, mainly relating to config parameters in the "Datasets" section of the config file. Specifically, I believe the "combineall" functionality (which originally combined training + test data into one large training set) may no longer work. My project does not require this functionality, so I haven't tested it, but I would recommend examining the code first if you need this functionality (if I have time, I will examine this in the future). Additionally, the original repo allows you to choose multiple training datasets to combine (in the "sources" setting) as well as multiple testing datasets to combine (in the "targets" setting). I haven't needed to combine multiple datasets, so I haven't tested this either, but I think this may not work anymore, due to the introduction of the validation set. Again, if I have time I will check on this, but to be safe, please examine the code yourself if you need this functionality. Otherwise, just use a single dataset.

Other comments:

- Due to the problem structure, the validation loss ONLY incorporates triplet loss (because the validation cross-entropy loss cannot be calculated, since the FC layers in the model are designed for the training classes, not the validation classes), and does NOT include the cross-entropy loss. In contrast, the training loss is the sum of triplet and cross-entropy losses (weighted by weight_t and weight_x). So, unfortunately, there is a disconnect between the training and validation losses, which may affect how your model is trained. For example, this potentially may reduce the effectiveness of something like ReduceLROnPlateau, since that LR scheduler is based only on validation loss (triplet only).
- During training, triplets are sampled using "batch hard" approach, to mine for moderately hard triplets. During validation, triplets are sampled using the "batch all" approach, in order to get more accurate and stable results between different model runs. For more info on "batch hard" vs. "batch all," see https://arxiv.org/abs/1703.07737

**QUICKSTART GUIDE**:

As mentioned above, I converted the argparse flags into a yaml config file, so that no matter what you want to do, you only need to modify the config file. This allows you to have a new config file for each "experiment," making it easier to document previous runs.

So, for each of the options below, you simply need to provide your preferred settings in the config file, then run the "main.py" file in the command line, providing just the file path to the config file.

*Option 1: train a new model*

Let's assume we are using config.py, which is located in the configs/ directory. 

- Leave the majority of the settings as they are in config.py
- Choose your preferred architecture (the 'arch' setting). You can use the following python code to see the names of the available models:

.. code-block:: python
    
    import torchreid
    torchreid.models.show_avai_models()

- Make sure to set the save_dir parameter to whichever folder you want to save the results to
- Set val_split to the percentage of the training set that you want to use as the validation set
- Set various training hyperparameters such as the optimizer, max_epoch, warmup period, LR scheduler, etc.
- Batch size is an important setting. When using hard triplet mining, having a larger batch size is better (to allow for harder triplets). So try to set the largest possible batch size without exceeding the GPU memory. (Note: I believe the batch size should be a multiple of num_instances, which is the number of images used per person)
- Setting weight_t and weight_x is important (the weights for triplet vs. cross-entropy loss). This is a scaling factor that needs to be tuned.
- For the most part, all of the other settings should be left as they are in config.py, at least for the Quick Start, if you are just trying to get the model to run. Of course, later on, feel free to try changing different settings.

Finally, run the following command in the command line. (Change the file path if you have a different config file).

.. code-block:: bash
    
    python main.py --config configs/config.yml


*Option 2: resume training from a checkpoint*

- Similar to Option 1, but you must also provide a file path (to a saved checkpoint) to the "resume" setting in the config file
- Then run the same command in the command line:

.. code-block:: bash
    
    python main.py --config configs/config.yml

*Option 3: evaluate a trained model*

- This is very similar to training a new model. The main difference is that in the config file, you should set "evaluate" to "true" (this tells the engine that you ONLY want to evaluate, not train). Also, you should provide a file path to model weights in the "load_weights" setting.
- Then run the same command in the command line, as usual:

.. code-block:: bash
    
    python main.py --config configs/config.yml

**Other assorted notes about config parameters**

- When using triplet loss (i.e., when you set "loss: triplet"), you must set "train_sampler: RandomIdentitySampler" because RandomIdentitySampler performs triplet mining/sampling.
- val_split indicates what % of the TRAINING set you want to split off to use as the validation set (the test set is not modified, so that the test results can be compared with prior work in the literature)
- save_dir: the directory where you want to save the training results

Finally, please also see https://kaiyangzhou.github.io/deep-person-reid/, which is the original documentation website provided by Kaiyang Zhou. It has a lot of useful information, as well as the Model Zoo, which contains pre-trained models that can be downloaded. Obviously, the website doesn't incorporate the changes that I made, but it still has a lot of useful info.

Original README from Kaiyang Zhou:
^^^^^^^^^^^^^^^^^^^^^^^^

Torchreid is a library built on `PyTorch <https://pytorch.org/>`_ for deep-learning person re-identification.

It features:

- multi-GPU training
- support both image- and video-reid
- end-to-end training and evaluation
- incredibly easy preparation of reid datasets
- multi-dataset training
- cross-dataset evaluation
- standard protocol used by most research papers
- highly extensible (easy to add models, datasets, training methods, etc.)
- implementations of state-of-the-art deep reid models
- access to pretrained reid models
- advanced training techniques
- visualization of ranking results


Documentation: https://kaiyangzhou.github.io/deep-person-reid/.

Code: https://github.com/KaiyangZhou/deep-person-reid.


Installation
---------------

The code works with both python2 and python3.

Option 1
^^^^^^^^^^^^
1. Install PyTorch and torchvision following the `official instructions <https://pytorch.org/>`_.
2. Clone ``deep-person-reid`` to your preferred directory

.. code-block:: bash
    
    $ git clone https://github.com/KaiyangZhou/deep-person-reid.git

3. :code:`cd` to :code:`deep-person-reid` and install dependencies

.. code-block:: bash
    
    $ cd deep-person-reid/
    $ pip install -r requirements.txt

4. Install ``torchreid``

.. code-block:: bash
    
    $ python setup.py install # or python3
    $ # If you wanna modify the source code without
    $ # the need to rebuild it, you can do
    $ # python setup.py develop

Option 2 (with conda)
^^^^^^^^^^^^^^^^^^^^^^^^
We also provide an environment.yml file for easy setup with conda.

1. Clone ``deep-person-reid`` to your preferred directory

.. code-block:: bash
    
    $ git clone https://github.com/KaiyangZhou/deep-person-reid.git

2. :code:`cd` to :code:`deep-person-reid` and create an environment (named ``torchreid``)

.. code-block:: bash
    
    $ cd deep-person-reid/
    $ conda env create -f environment.yml

In doing so, the dependencies will be automatically installed.

3. Install PyTorch and torchvision (select the proper cuda version to suit your machine)

.. code-block:: bash
    
    $ conda activate torchreid
    $ conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

4. Install ``torchreid``

.. code-block:: bash

    $ python setup.py install
    $ # If you wanna modify the source code without
    $ # the need to rebuild it, you can do
    $ # python setup.py develop


Get started: 30 seconds to Torchreid
-------------------------------------
1. Import ``torchreid``

.. code-block:: python
    
    import torchreid

2. Load data manager

.. code-block:: python
    
    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='market1501',
        height=256,
        width=128,
        batch_size=32,
        market1501_500k=False
    )

3 Build model, optimizer and lr_scheduler

.. code-block:: python
    
    model = torchreid.models.build_model(
        name='resnet50',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
    )

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )

4. Build engine

.. code-block:: python
    
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

5. Run training and test

.. code-block:: python
    
    engine.run(
        save_dir='log/resnet50',
        max_epoch=60,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )


A unified interface
-----------------------
In "deep-person-reid/scripts/", we provide a unified interface including a default parser file ``default_parser.py`` and the main script ``main.py``. For example, to train an image reid model on Market1501 using softmax, you can do

.. code-block:: bash
    
    python main.py \
    --root path/to/reid-data \
    --app image \
    --loss softmax \
    --label-smooth \
    -s market1501 \
    -a resnet50 \
    --optim adam \
    --lr 0.0003 \
    --max-epoch 60 \
    --stepsize 20 40 \
    --batch-size 32 \
    --save-dir log/resnet50-market-softmax \
    --gpu-devices 0

Please refer to ``default_parser.py`` and ``main.py`` for more details.


Datasets
--------

Image-reid datasets
^^^^^^^^^^^^^^^^^^^^^
- `Market1501 <https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf>`_
- `CUHK03 <https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf>`_
- `DukeMTMC-reID <https://arxiv.org/abs/1701.07717>`_
- `MSMT17 <https://arxiv.org/abs/1711.08565>`_
- `VIPeR <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.331.7285&rep=rep1&type=pdf>`_
- `GRID <http://www.eecs.qmul.ac.uk/~txiang/publications/LoyXiangGong_cvpr_2009.pdf>`_
- `CUHK01 <http://www.ee.cuhk.edu.hk/~xgwang/papers/liZWaccv12.pdf>`_
- `SenseReID <http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Spindle_Net_Person_CVPR_2017_paper.pdf>`_
- `QMUL-iLIDS <http://www.eecs.qmul.ac.uk/~sgg/papers/ZhengGongXiang_BMVC09.pdf>`_
- `PRID <https://pdfs.semanticscholar.org/4c1b/f0592be3e535faf256c95e27982db9b3d3d3.pdf>`_

Video-reid datasets
^^^^^^^^^^^^^^^^^^^^^^^
- `MARS <http://www.liangzheng.org/1320.pdf>`_
- `iLIDS-VID <https://www.eecs.qmul.ac.uk/~sgg/papers/WangEtAl_ECCV14.pdf>`_
- `PRID2011 <https://pdfs.semanticscholar.org/4c1b/f0592be3e535faf256c95e27982db9b3d3d3.pdf>`_
- `DukeMTMC-VideoReID <http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Exploit_the_Unknown_CVPR_2018_paper.pdf>`_

Models
-------

ImageNet classification models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- `ResNet <https://arxiv.org/abs/1512.03385>`_
- `ResNeXt <https://arxiv.org/abs/1611.05431>`_
- `SENet <https://arxiv.org/abs/1709.01507>`_
- `DenseNet <https://arxiv.org/abs/1608.06993>`_
- `Inception-ResNet-V2 <https://arxiv.org/abs/1602.07261>`_
- `Inception-V4 <https://arxiv.org/abs/1602.07261>`_
- `Xception <https://arxiv.org/abs/1610.02357>`_

Lightweight models
^^^^^^^^^^^^^^^^^^^
- `NASNet <https://arxiv.org/abs/1707.07012>`_
- `MobileNetV2 <https://arxiv.org/abs/1801.04381>`_
- `ShuffleNet <https://arxiv.org/abs/1707.01083>`_
- `ShuffleNetV2 <https://arxiv.org/abs/1807.11164>`_
- `SqueezeNet <https://arxiv.org/abs/1602.07360>`_

ReID-specific models
^^^^^^^^^^^^^^^^^^^^^^
- `MuDeep <https://arxiv.org/abs/1709.05165>`_
- `ResNet-mid <https://arxiv.org/abs/1711.08106>`_
- `HACNN <https://arxiv.org/abs/1802.08122>`_
- `PCB <https://arxiv.org/abs/1711.09349>`_
- `MLFN <https://arxiv.org/abs/1803.09132>`_
- `OSNet <https://arxiv.org/abs/1905.00953>`_

Losses
------
- `Softmax (cross entropy loss with label smoothing) <https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf>`_
- `Triplet (hard example mining triplet loss) <https://arxiv.org/abs/1703.07737>`_


Citation
---------
If you find this code useful to your research, please cite the following publication.

.. code-block:: bash
    
    @article{zhou2019osnet,
      title={Omni-Scale Feature Learning for Person Re-Identification},
      author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
      journal={arXiv preprint arXiv:1905.00953},
      year={2019}
    }

