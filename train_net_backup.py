"""
Dyhead Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import os
import sys
import itertools
import logging
import time
# fmt: off
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

from typing import Any, Dict, List, Set

import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from typing import Any, Callable, Dict, List, Optional, Union
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.samplers import TrainingSampler
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.engine import SimpleTrainer, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from collections import OrderedDict
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset, ToIterableDataset
from detectron2.data.samplers import (
    InferenceSampler,
    RandomSubsetTrainingSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from dyhead import add_dyhead_config
from extra import add_extra_config
from extra import ConceptMapper
from extra import ConceptFinder, add_concept_config
from extra import flatten_json, unflatten_json
import os
import nltk
from nltk.corpus import wordnet as wn
import wandb
import yaml


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # TODO: drigoni: add concepts to classes
        concept_finder = ConceptFinder(cfg.CONCEPT.FILE)
        self.coco2synset = concept_finder.extend_descendants(depth=cfg.CONCEPT.DEPTH,
                                                             unique=cfg.CONCEPT.UNIQUE,
                                                             only_name=cfg.CONCEPT.ONLY_NAME)

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg, self.coco2synset)
        # Build DDP Model with find_unused_parameters to add flexibility.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
                find_unused_parameters=True
            )
        self._trainer = SimpleTrainer(model, data_loader, optimizer)
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg, coco2synset):
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

        # TODO drigoni: introduces a new mapper
        mapper = ConceptMapper(cfg, True, coco2synset=coco2synset)

        if cfg.SEED!=-1:
            sampler = TrainingSampler(len(dataset), seed=cfg.SEED)
        else:
            sampler = None
        return build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, sampler=sampler)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, coco2synset):
        if isinstance(dataset_name, str):
            dataset_name = [dataset_name]

        dataset = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=True,  # TODO drigoni: True instead of False. We need at least one annotation.
            proposal_files=[
                cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] for x in dataset_name
            ]
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        # TODO drigoni: introduces a new mapper for the test and remove images with 0 annotations.
        mapper = ConceptMapper(cfg, False, coco2synset=coco2synset)
        return build_detection_test_loader(dataset=dataset, mapper=mapper,
                                           num_workers=cfg.DATALOADER.NUM_WORKERS,
                                           sampler=InferenceSampler(len(dataset)))

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        concept_finder = ConceptFinder(cfg.CONCEPT.FILE)
        coco2synset = concept_finder.extend_descendants(depth=cfg.CONCEPT.DEPTH,
                                                        unique=cfg.CONCEPT.UNIQUE,
                                                        only_name=cfg.CONCEPT.ONLY_NAME)

        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name, coco2synset)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            # TODO drigoni: stop backpropagation in the backbone
            # stop backpropagation in the backbones parameters
            # if "backbone" in key:
            #     print("Gradient stop for layer: ", key)
            #     value.requires_grad = False
            #     continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            # TODO drigoni: the following if causes a problem and need to be fixed/removed.
            # The problem is that no WEIGHT_DECAY_BIAS parameter is used in the config file.
            # print("bias" in key, key, weight_decay, cfg.SOLVER.WEIGHT_DECAY_BIAS)
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_dyhead_config(cfg)
    add_extra_config(cfg)
    add_concept_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    # wandb init only in rank0 and on device 0
    if not args.eval_only and args.machine_rank == 0 and torch.cuda.current_device() == 0:
        raw_cfg = yaml.safe_load(cfg.dump())
        # make flat the nested dictionary from yaml
        flatten_json(raw_cfg)
        wandb.init(project="CATSS", sync_tensorboard=True, config=raw_cfg)
        # unflatten_json(raw_cfg)

    # metaMapping = MetadataCatalog.get('coco_2017_train').thing_dataset_id_to_contiguous_id  # from origin ids to contiguos one
    # thing_classes = MetadataCatalog.get('coco_2017_train').thing_classes  # from origin ids to contiguos one
    # metaMapping = {val: key for key, val in metaMapping.items()}
    # print('metaMapping', metaMapping, len(metaMapping.keys()), max(metaMapping.values()))
    # print('thing_classes', thing_classes, len(thing_classes))
    # exit(1)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    return trainer.train()



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.machine_rank = int(os.environ['SLURM_PROCID']) if 'SLURM_PROCID' in os.environ else args.machine_rank
    print("Command Line Args:", args)
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )