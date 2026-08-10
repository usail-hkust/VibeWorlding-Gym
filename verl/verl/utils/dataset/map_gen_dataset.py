"""
地图生成任务专用数据集

处理由 json_to_parquet.py 生成的 parquet 格式数据
"""

import json
import logging
import re
from io import BytesIO
from PIL import Image

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class MapGenRLDataset(RLHFDataset):
    """
    地图生成任务数据集
    
    与标准 RLHFDataset 的区别：
    1. 处理 prompt 可能是 JSON 字符串或列表的情况
    2. extra_info 可能是 JSON 字符串，需要解析
    3. images 是 bytes 列表，需要转换为 PIL Image
    """

    def _parse_prompt(self, doc: dict) -> list:
        """
        解析 prompt 字段，支持字符串(JSON)或列表格式
        """
        prompt = doc.get(self.prompt_key, [])
        
        # 如果是字符串，尝试解析为 JSON 列表
        if isinstance(prompt, str):
            try:
                prompt = json.loads(prompt)
            except json.JSONDecodeError:
                # 如果解析失败，作为纯文本包装成单条消息
                return [{"role": "user", "content": prompt}]
        
        # 确保是列表
        if isinstance(prompt, list):
            return prompt
        
        # 其他情况，转为字符串包装
        return [{"role": "user", "content": str(prompt)}]

    def _process_extra_info(self, doc: dict) -> dict:
        """处理 extra_info 字段，解析 JSON 字符串"""
        extra_info = doc.get("extra_info", "{}")
        if isinstance(extra_info, str):
            try:
                return json.loads(extra_info)
            except json.JSONDecodeError:
                return {"raw": extra_info}
        return extra_info if isinstance(extra_info, dict) else {}
    
    def _build_messages(self, example: dict):
        """
        重写 _build_messages 以处理 JSON 字符串格式的 prompt
        """
        import numpy as np
        
        # 先解析 prompt
        messages = self._parse_prompt(example)
        
        # 解析图片 bytes -> PIL Image
        images_raw = example.pop(self.image_key, None) or []
        
        # 处理 numpy array 情况 (parquet 读取的列表可能是 numpy array)
        if isinstance(images_raw, np.ndarray):
            images_raw = images_raw.tolist()
        elif not isinstance(images_raw, (list, tuple)):
            images_raw = [images_raw] if images_raw else []
        
        images = []
        for img_data in images_raw:
            # 处理 bytes 类型
            if isinstance(img_data, bytes):
                try:
                    img = Image.open(BytesIO(img_data))
                    images.append(img.convert("RGB"))
                except Exception as e:
                    logger.warning(f"Failed to load image from bytes: {e}")
            # 处理 numpy bytes 类型 (parquet 可能存储为 numpy.bytes_)
            elif isinstance(img_data, (np.bytes_, type(np.bytes_()))):
                try:
                    img = Image.open(BytesIO(bytes(img_data)))
                    images.append(img.convert("RGB"))
                except Exception as e:
                    logger.warning(f"Failed to load image from numpy bytes: {e}")
            # 处理 PIL.Image 类型 (已经转换过)
            elif isinstance(img_data, Image.Image):
                images.append(img_data.convert("RGB"))
            # 处理 dict 类型
            elif isinstance(img_data, dict):
                images.append(img_data)
            else:
                logger.warning(f"Unsupported image type: {type(img_data)}, skipping")
        
        videos = example.pop(self.video_key, None) or []

        image_offset, video_offset = 0, 0
        for message in messages:
            if not images and not videos:
                continue
            assert self.processor is not None, "processor is needed to process image and video"

            content = message["content"]
            if not isinstance(content, str):
                continue

            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    assert image_offset < len(images), f"image_offset {image_offset} >= len(images) {len(images)}"
                    image = images[image_offset]
                    if isinstance(image, Image.Image):
                        image = image.convert("RGB")
                        content_list.append({"type": "image", "image": image})
                    elif isinstance(image, dict):
                        if "bytes" in image:
                            image["image"] = Image.open(BytesIO(image["bytes"]))
                        content_list.append({"type": "image", **image})
                    else:
                        raise TypeError(f"image must be dict or PIL.Image, unsupported image type: {type(image)}")
                    image_offset += 1
                elif segment == "<video>":
                    assert video_offset < len(videos), f"video_offset {video_offset} >= len(videos) {len(videos)}"
                    content_list.append({"type": "video", **videos[video_offset]})
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        return messages
