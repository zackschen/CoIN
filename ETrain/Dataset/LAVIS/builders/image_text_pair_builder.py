import os
import logging
import warnings

from ETrain.utils.LAVIS.common.registry import registry
from ETrain.Dataset.LAVIS.builders.base_dataset_builder import BaseDatasetBuilder
from ETrain.Dataset.LAVIS.datasets.laion_dataset import LaionDataset
from ETrain.Dataset.LAVIS.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
from ETrain.Dataset.LAVIS.datasets.text_caps import TextCapDataset
from ETrain.Dataset.LAVIS.datasets.llava_dataset import LlavaDetailDataset, LlavaReasonDataset, LlavaConversationDataset
from ETrain.Dataset.LAVIS.datasets.unnatural_instruction import UnnaturalDataset
from ETrain.Dataset.LAVIS.datasets.multitask_conversation import MultiTaskConversationDataset
from ETrain.Dataset.LAVIS.datasets.flickr import GroundedDetailDataset,CaptionToObjectDataset,PhraseToObjectDataset
from ETrain.Dataset.LAVIS.datasets.vg_dataset import ReferVisualGenomeDataset
from ETrain.Dataset.LAVIS.datasets.coco_dataset import ReferCOCODataset, InvReferCOCODataset
from ETrain.Dataset.LAVIS.datasets.gqa_datasets import GQADataset
from ETrain.Dataset.LAVIS.datasets.aok_vqa_datasets import AOKVQADataset
from ETrain.Dataset.LAVIS.datasets.coco_vqa_datasets import COCOVQADataset
from ETrain.Dataset.LAVIS.datasets.ocrvqa_dataset import OCRVQADataset
from ETrain.Dataset.LAVIS.datasets.coco_caption import COCOCapDataset
from ETrain.Dataset.LAVIS.datasets.CLIT_dataset import *

@registry.register_builder("multitask_conversation")
class MultitaskConversationBuilder(BaseDatasetBuilder):
    train_dataset_cls = MultiTaskConversationDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/multitask_conversation/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets


@registry.register_builder("unnatural_instruction")
class UnnaturalInstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = UnnaturalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/nlp/unnatural_instruction.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
        )

        return datasets



@registry.register_builder("llava_detail")
class LlavaDetailBuilder(BaseDatasetBuilder):
    train_dataset_cls = LlavaDetailDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/llava/detail.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets
    


@registry.register_builder("llava_reason")
class LlavaReasonBuilder(BaseDatasetBuilder):
    train_dataset_cls = LlavaReasonDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/llava/reason.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets

@registry.register_builder("llava_conversation")
class LlavaReasonBuilder(BaseDatasetBuilder):
    train_dataset_cls = LlavaConversationDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/llava/conversation.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets


class AllRefCOCOBuilder(BaseDatasetBuilder):

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        image_path = build_info.image_path
        ann_path = build_info.ann_path

        datasets = dict()

        if not os.path.exists(image_path):
            warnings.warn("image path {} does not exist.".format(image_path))
        if not os.path.exists(ann_path):
            warnings.warn("ann path {} does not exist.".format(ann_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=ann_path,
            vis_root=image_path,
            dataset=build_info.dataset,
            splitBy=build_info.splitBy
        )

        return datasets
    

@registry.register_builder("refcoco")
class RefCOCOBuilder(AllRefCOCOBuilder):
    train_dataset_cls = ReferCOCODataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/coco_bbox/refcoco.yaml",
    }

@registry.register_builder("refcocop")
class RefCOCOPBuilder(AllRefCOCOBuilder):
    train_dataset_cls = ReferCOCODataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/coco_bbox/refcocop.yaml",
    }


@registry.register_builder("refcocog")
class RefCOCOGBuilder(AllRefCOCOBuilder):
    train_dataset_cls = ReferCOCODataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/coco_bbox/refcocog.yaml",
    }

@registry.register_builder("invrefcoco")
class RefCOCOBuilder(AllRefCOCOBuilder):
    train_dataset_cls = InvReferCOCODataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/coco_bbox/invrefcoco.yaml",
    }


@registry.register_builder("invrefcocop")
class RefCOCOPBuilder(AllRefCOCOBuilder):
    train_dataset_cls = InvReferCOCODataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/coco_bbox/invrefcocop.yaml",
    }


@registry.register_builder("invrefcocog")
class RefCOCOGBuilder(AllRefCOCOBuilder):
    train_dataset_cls = InvReferCOCODataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/coco_bbox/invrefcocog.yaml",
    }

@registry.register_builder("refvg")
class RefVisualGenomeBuilder(BaseDatasetBuilder):
    train_dataset_cls = ReferVisualGenomeDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/vg/ref.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        data_dir = build_info.data_dir
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            data_dir=data_dir,
        )

        return datasets


@registry.register_builder("textcaps_caption")
class TextcapCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextCapDataset

    DATASET_CONFIG_DICT = {"default": "configs/LAVIS/datasets/textcaps/caption.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets
    
@registry.register_builder("coco_vqa")
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/coco/defaults_vqa.yaml",
    }

@registry.register_builder("ok_vqa")
class OKVQABuilder(COCOVQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/okvqa/defaults.yaml",
    }


@registry.register_builder("aok_vqa")
class AOKVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = AOKVQADataset

    DATASET_CONFIG_DICT = {"default": "configs/LAVIS/datasets/aokvqa/defaults.yaml"}


@registry.register_builder("gqa")
class GQABuilder(BaseDatasetBuilder):
    train_dataset_cls = GQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/gqa/balanced_val.yaml",
    }




@registry.register_builder("flickr_grounded_caption")
class GroundedCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = GroundedDetailDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/flickr/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets


@registry.register_builder("flickr_CaptionToPhrase")
class CaptionToPhraseBuilder(BaseDatasetBuilder):
    train_dataset_cls = CaptionToObjectDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/flickr/caption_to_phrase.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets

@registry.register_builder("flickr_ObjectToPhrase")
class CaptionToPhraseBuilder(BaseDatasetBuilder):
    train_dataset_cls = PhraseToObjectDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/flickr/object_to_phrase.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets




class DocumentVQABuilder(BaseDatasetBuilder):
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.image_path,
            ann_path=build_info.ann_path
        )

        return datasets
    

@registry.register_builder("ocrvqa")
class OCRVQABuilder(DocumentVQABuilder):
    train_dataset_cls = OCRVQADataset
    DATASET_CONFIG_DICT = {"default": "configs/LAVIS/datasets/ocrvqa/ocrvqa.yaml"}


@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset

    DATASET_CONFIG_DICT = {"default": "configs/LAVIS/datasets/cc_sbu/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("laion")
class LaionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/LAVIS/datasets/laion/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets



@registry.register_builder("coco_caption")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/coco/caption.yaml",
    }



@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets

@registry.register_builder("clit_dataset")
class CLITBuilder(BaseDatasetBuilder):
    train_dataset_cls = CLITDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/CLIT/CLIT_scienceqa.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets

@registry.register_builder("clit_scienceqa")
class CLITScienceQABuilder(CLITBuilder):
    train_dataset_cls = CLIT_ScientQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/CLIT/CLIT_scienceqa.yaml",
    }

@registry.register_builder("clit_gqa")
class CLITGQABuilder(CLITBuilder):
    train_dataset_cls = CLIT_GQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/CLIT/CLIT_gqa.yaml",
    }

@registry.register_builder("clit_grounding")
class CLITGroundingBuilder(CLITBuilder):
    train_dataset_cls = CLIT_GroundingDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/CLIT/CLIT_grounding.yaml",
    }
@registry.register_builder("clit_imagenet")
class CLITImageNetBuilder(CLITBuilder):
    train_dataset_cls = CLIT_ImageNetDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/CLIT/CLIT_imagenet.yaml",
    }

@registry.register_builder("clit_ocrvqa")
class CLITOCRVQABuilder(CLITBuilder):
    train_dataset_cls = CLIT_OCRVQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/CLIT/CLIT_ocrvqa.yaml",
    }

@registry.register_builder("clit_textvqa")
class CLITTextVQABuilder(CLITBuilder):
    train_dataset_cls = CLIT_TextVQADataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/CLIT/CLIT_textvqa.yaml",
    }

@registry.register_builder("clit_vizwiz")
class CLITVizWizBuilder(CLITBuilder):
    train_dataset_cls = CLIT_VizWizDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/CLIT/CLIT_vizwiz.yaml",
    }

@registry.register_builder("clit_vqav2")
class CLITVQAV2Builder(CLITBuilder):
    train_dataset_cls = CLIT_VQAv2Dataset
    DATASET_CONFIG_DICT = {
        "default": "configs/LAVIS/datasets/CLIT/CLIT_vqav2.yaml",
    }