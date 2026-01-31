
import json
import os
import re
import io
import traceback
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb
from ..distributed_iterable_dataset import DistributedIterableDataset


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class ThinkingGenerationIterableDataset(DistributedIterableDataset):

    
    DEFAULT_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

    
    def __init__(
        self, 
        dataset_name, 
        transform, 
        tokenizer, 
        vit_transform,
        jsonl_path_list, 
        data_dir_list, 
        num_used_data,
        local_rank=0, 
        world_size=1, 
        num_workers=8, 
        data_status=None,
        shuffle_lines=True, 
        shuffle_seed=0,
        use_system_prompt=True,
        custom_system_prompt=None,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.use_system_prompt = use_system_prompt
        self.system_prompt = custom_system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        self.data_paths = self.get_data_paths(
            jsonl_path_list, 
            data_dir_list, 
            num_used_data, 
            shuffle_lines, 
            shuffle_seed,
        )
        self.set_epoch()

    def get_data_paths(self, jsonl_path_list, data_dir_list, num_used_data, shuffle_lines, shuffle_seed):
        data_paths = []
        for jsonl_path, image_dir, num_data_point in zip(jsonl_path_list, data_dir_list, num_used_data):
            with open(jsonl_path, 'r') as f:
                raw_data = f.readlines()
            if shuffle_lines:
                self.rng.seed(shuffle_seed)
                self.rng.shuffle(raw_data)
            raw_data = raw_data[:num_data_point]
            data_paths.extend([(json_data, image_dir) for json_data in raw_data])
        return data_paths

    def _init_data(self):
        return {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            'num_tokens': 0,
        }

    def _add_text(self, data, text, need_loss):
        text_ids = self.tokenizer.encode(text)
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append({
            'type': 'text',
            'enable_cfg': 1,
            'loss': int(need_loss),
            'special_token_loss': 0,
            'special_token_label': None,
        })
        return data

    def _add_image_for_understanding(self, data, image):

        # We only provide demo-level code at this stage. 
        # The complete implementation will be released upon paper acceptance.
        
        return data

    def _add_image_for_generation(self, data, image, is_intermediate=False):

        # We only provide demo-level code at this stage. 
        # The complete implementation will be released upon paper acceptance.
        
        return data

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, (json_data, image_dir) in enumerate(data_paths_per_worker_, start=row_start_id):
                try:
                    data_item = json.loads(json_data)
                    
                    if 'instruction' not in data_item or 'thinking' not in data_item:
                        print(f"Missing required fields in row {row_idx}, skipped.")
                        continue
                    
                    data = self._init_data()
                    
                    if self.use_system_prompt:
                        system_prompt = data_item.get('system_prompt', self.system_prompt)
                        data = self._add_text(data, system_prompt, need_loss=False)
                    
                    if 'source_image' in data_item:
                        source_img = pil_img2rgb(Image.open(os.path.join(image_dir, data_item['source_image'])))
                        data = self._add_image_for_understanding(data, source_img)
                    
                    data = self._add_text(data, data_item['instruction'], need_loss=False)
                    
                    thinking = data_item['thinking']
                    
                    img_pattern = r'<IMG_(\d+)>'
                    parts = re.split(img_pattern, thinking)
                    
                    for i, part in enumerate(parts):
                        if i % 2 == 0:  
                            if part.strip():
                                wrapped_text = f"<think>{part.strip()}</think>" 
                        else:  
                            img_key = f"thinking_image_{part}"
                            if img_key in data_item:
                                thinking_img = pil_img2rgb(Image.open(os.path.join(image_dir, data_item[img_key])))
                                data = self._add_image_for_generation(data, thinking_img, is_intermediate=True)
                    
                    if 'target_image' in data_item:
                        target_img = pil_img2rgb(Image.open(os.path.join(image_dir, data_item['target_image'])))
                        data = self._add_image_for_generation(data, target_img, is_intermediate=False)
                    
                    if 'answer' in data_item:
                        wrapped_answer = f"<answer>{data_item['answer']}</answer>"
                        data = self._add_text(data, wrapped_answer, need_loss=True)
                    
                    has_loss = [item['loss'] for item in data['sequence_plan']]
                    if sum(has_loss) == 0:
                        print(f'No loss defined, skipped.')
                        continue
                    
                    data['data_indexes'] = {
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    }
                    
                    yield data

                except Exception as e:
                    traceback.print_exc()
                    continue

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
