"""
地图生成任务专用工具实现
包括：
1. PCG渲染工具 - 调用PCG服务器生成场景图片
2. 地图编辑工具 - rotation_and_translation, delete, add
3. 资产检索工具 - retrieve_assets (generate 任务专用)
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
import shutil
from typing import Any, Dict, List, Optional
from pathlib import Path

from pydantic import BaseModel, Field

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse, OpenAIFunctionToolSchema


logger = logging.getLogger(__name__)


# ============== 数据模型定义 ==============

class CorrectionItem(BaseModel):
    """rotation_and_translation的参数项"""
    original_data: dict = Field(..., description="原始元件数据")
    modified_data: dict = Field(..., description="修改后的元件数据")


class RotationTranslationInput(BaseModel):
    """rotation_and_translation工具输入"""
    corrections: list[CorrectionItem] = Field(..., description="需要修改的元件列表")


class ModifiedItem(BaseModel):
    """add/delete的参数项"""
    name: str = Field(..., description="元件名称")
    pos: list = Field(..., description="位置坐标")
    Extend: list = Field(..., description="尺寸")
    rotate: Optional[list] = Field(None, description="旋转角度")
    reason: Optional[str] = Field(None, description="操作原因")


class AddDeleteInput(BaseModel):
    """add/delete工具输入"""
    modified_data: list[ModifiedItem] = Field(..., description="需要添加/删除的元件列表")


class PCGRenderInput(BaseModel):
    """PCG渲染工具输入"""
    map_data: dict = Field(..., description="地图JSON数据")


class PCGRenderOutput(BaseModel):
    """PCG渲染工具输出"""
    image_paths: list[str] = Field(..., description="生成的图片路径列表")
    success: bool = Field(..., description="是否成功")
    error_msg: Optional[str] = Field(None, description="错误信息")


# ============== 工具实现 ==============

class PCGRenderTool(BaseTool):
    """
    PCG渲染工具：将地图JSON转换为场景图片
    
    调用PCG服务器渲染5张视角图（左、右、前、后、俯视图）
    """
    # 类级别默认schema，会被config中的tool_schema覆盖
    DEFAULT_SCHEMA = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "pcg_render",
            "description": "Render the current map to 5 view images (left, right, front, back, top)",
            "parameters": {
                "type": "object",
                "properties": {
                    "map_data": {
                        "type": "object",
                        "description": "Current map JSON data to render"
                    }
                },
                "required": ["map_data"]
            }
        }
    )
    
    # PCG服务器配置
    pcg_server_url: str = "http://localhost:8080"
    
    def _make_hashable(self, obj):
        """递归将不可哈希的对象转为可哈希的元组"""
        if isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(self._make_hashable(item) for item in obj)
        elif isinstance(obj, (tuple, set)):
            return tuple(self._make_hashable(item) for item in obj)
        else:
            return obj
    
    def _get_match_key(self, item_dict):
        """提取核心字段生成匹配哈希键"""
        if not isinstance(item_dict, dict):
            return None
        target_fields = (
            item_dict.get("name"),
            item_dict.get("pos"),
            item_dict.get("Extend")
        )
        return self._make_hashable(target_fields)
    
    def _parse2pcg(self, llm_output: dict, component_info: dict) -> tuple:
        """
        将LLM输出转换为PCG格式
        简化版实现，实际应该使用data_process.py中的完整逻辑
        """
        # 这里简化处理，实际需要调用data_process.parse2pcg
        # 为简化依赖，我们假设输入已经是合理的格式
        return llm_output, [], "", False
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """
        执行PCG渲染
        
        Args:
            instance_id: 工具实例ID
            parameters: 包含 map_data (dict)
            **kwargs: 额外参数，包含 tool_context
            
        Returns:
            (tool_response, reward_score, metrics)
        """
        try:
            map_data = parameters.get("map_data", {})
            tool_context = kwargs.get("tool_context", {})
            
            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix="pcg_render_")
            json_dir = os.path.join(temp_dir, "pcg_json")
            output_image_dir = os.path.join(temp_dir, "images")
            os.makedirs(json_dir, exist_ok=True)
            os.makedirs(output_image_dir, exist_ok=True)
            
            # 保存map JSON
            result_path = os.path.join(json_dir, "result.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(map_data, f, indent=2, ensure_ascii=False)
            
            # 调用PCG服务器
            command = [
                "python", "pcg_request_batch.py",
                "--server", self.pcg_server_url,
                "--local_folder", json_dir,
                "--batch_mode", "cmd_folder",
                "--out_dir", output_image_dir,
                "--stream"
            ]
            
            proc = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            )
            
            # 获取生成的图片
            import glob
            image_paths = sorted(glob.glob(os.path.join(output_image_dir, "*.jpg")))
            
            if not image_paths:
                return (
                    ToolResponse(text="渲染失败：未生成图片"),
                    0.0,
                    {"success": False, "error": "No images generated"}
                )
            
            # 读取图片为bytes
            images_bytes = []
            for img_path in image_paths:
                with open(img_path, 'rb') as f:
                    images_bytes.append(f.read())
            
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return (
                ToolResponse(text=f"成功渲染 {len(images_bytes)} 张图片"),
                1.0,  # 成功奖励
                {"success": True, "image_count": len(images_bytes)}
            )
            
        except subprocess.CalledProcessError as e:
            return (
                ToolResponse(text=f"渲染服务错误: {e.stderr}"),
                0.0,
                {"success": False, "error": f"PCG server error: {e}"}
            )
        except Exception as e:
            return (
                ToolResponse(text=f"渲染失败: {str(e)}"),
                0.0,
                {"success": False, "error": str(e)}
            )


class RotationTranslationTool(BaseTool):
    """
    旋转和平移工具：修改场景中已有元件的位置、尺寸、旋转
    """
    DEFAULT_SCHEMA = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "rotation_and_translation",
            "description": "Rotate and translate existing components in the scene",
            "parameters": {
                "type": "object",
                "properties": {
                    "corrections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original_data": {"type": "object"},
                                "modified_data": {"type": "object"}
                            },
                            "required": ["original_data", "modified_data"]
                        }
                    }
                },
                "required": ["corrections"]
            }
        }
    )
    
    def _make_hashable(self, obj):
        if isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(self._make_hashable(item) for item in obj)
        elif isinstance(obj, (tuple, set)):
            return tuple(self._make_hashable(item) for item in obj)
        else:
            return obj
    
    def _get_match_key(self, item_dict):
        if not isinstance(item_dict, dict):
            return None
        target_fields = (
            item_dict.get("name"),
            item_dict.get("pos"),
            item_dict.get("Extend")
        )
        return self._make_hashable(target_fields)
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """执行旋转和平移操作"""
        try:
            corrections = parameters.get("corrections", [])
            tool_context = kwargs.get("tool_context", {})
            
            # 从context获取当前地图状态
            current_map = tool_context.get("current_map", {})
            
            # 预处理：生成 原数据哈希 -> 修正值 的映射
            corr_map = {}
            for item in corrections:
                original_data = item.get("original_data")
                if not original_data:
                    continue
                key = self._get_match_key(original_data)
                corr_map[key] = item.get("modified_data", {})
            
            # 应用修改
            modified_count = 0
            for main_key, sub_dict in current_map.items():
                if not isinstance(sub_dict, dict):
                    continue
                for sub_key, v1 in sub_dict.items():
                    if not isinstance(v1, list):
                        continue
                    
                    new_v1 = []
                    for elem in v1:
                        elem_key = self._get_match_key(elem)
                        if elem_key in corr_map:
                            corrected_val = corr_map[elem_key]
                            if isinstance(corrected_val, list):
                                new_v1.extend(corrected_val)
                                modified_count += len(corrected_val)
                            elif isinstance(corrected_val, dict):
                                if "name" in corrected_val:
                                    new_v1.append(corrected_val)
                                    modified_count += 1
                        else:
                            new_v1.append(elem)
                    sub_dict[sub_key] = new_v1
            
            # 更新context中的地图状态
            tool_context["current_map"] = current_map
            
            return (
                ToolResponse(text=f"成功修改 {modified_count} 个元件"),
                1.0 if modified_count > 0 else 0.0,
                {"modified_count": modified_count, "modified_map": current_map}
            )
            
        except Exception as e:
            return (
                ToolResponse(text=f"操作失败: {str(e)}"),
                0.0,
                {"error": str(e)}
            )


class DeleteTool(BaseTool):
    """
    删除工具：删除场景中不合理的元件
    """
    DEFAULT_SCHEMA = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "delete",
            "description": "Delete unreasonable components from the scene",
            "parameters": {
                "type": "object",
                "properties": {
                    "modified_data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "pos": {"type": "array"},
                                "Extend": {"type": "array"}
                            },
                            "required": ["name", "pos", "Extend"]
                        }
                    }
                },
                "required": ["modified_data"]
            }
        }
    )
    
    def _make_hashable(self, obj):
        if isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(self._make_hashable(item) for item in obj)
        elif isinstance(obj, (tuple, set)):
            return tuple(self._make_hashable(item) for item in obj)
        else:
            return obj
    
    def _get_match_key(self, item_dict):
        if not isinstance(item_dict, dict):
            return None
        target_fields = (
            item_dict.get("name"),
            item_dict.get("pos"),
            item_dict.get("Extend")
        )
        return self._make_hashable(target_fields)
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """执行删除操作"""
        try:
            modified_data = parameters.get("modified_data", [])
            tool_context = kwargs.get("tool_context", {})
            current_map = tool_context.get("current_map", {})
            
            # 构建删除键集合
            delete_keys = set()
            for item in modified_data:
                delete_keys.add(self._get_match_key(item))
            
            # 执行删除
            deleted_count = 0
            for main_key, sub_dict in current_map.items():
                if not isinstance(sub_dict, dict):
                    continue
                for sub_key, v1 in sub_dict.items():
                    if not isinstance(v1, list):
                        continue
                    
                    new_v1 = []
                    for elem in v1:
                        elem_key = self._get_match_key(elem)
                        if elem_key not in delete_keys:
                            new_v1.append(elem)
                        else:
                            deleted_count += 1
                    sub_dict[sub_key] = new_v1
            
            tool_context["current_map"] = current_map
            
            return (
                ToolResponse(text=f"成功删除 {deleted_count} 个元件"),
                1.0 if deleted_count > 0 else 0.0,
                {"deleted_count": deleted_count, "modified_map": current_map}
            )
            
        except Exception as e:
            return (
                ToolResponse(text=f"删除失败: {str(e)}"),
                0.0,
                {"error": str(e)}
            )


class AddTool(BaseTool):
    """
    添加工具：在场景中添加新元件
    """
    DEFAULT_SCHEMA = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "add",
            "description": "Add new components to the scene. type_id (from retrieve_assets) and name are both required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "modified_data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name":    {"type": "string"},
                                "type_id": {"type": "string",
                                            "description": "8-digit asset ID from retrieve_assets, e.g. \"20007733\""},
                                "pos":    {"type": "array"},
                                "Extend": {"type": "array"}
                            },
                            "required": ["name", "type_id", "pos", "Extend"]
                        }
                    }
                },
                "required": ["modified_data"]
            }
        }
    )
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """执行添加操作"""
        try:
            modified_data = parameters.get("modified_data", [])
            tool_context = kwargs.get("tool_context", {})
            current_map = tool_context.get("current_map", {})

            # 规范化 type_id，过滤缺失项
            valid_items = []
            skipped = 0
            for item in modified_data:
                if not isinstance(item, dict):
                    continue
                # typeId → type_id 统一
                if item.get("typeId") and not item.get("type_id"):
                    item["type_id"] = str(item.pop("typeId"))
                if not item.get("type_id"):
                    logger.warning(f"[AddTool] item 缺少 type_id，跳过: name={item.get('name','?')}")
                    skipped += 1
                    continue
                item["type_id"] = str(item["type_id"])
                valid_items.append(item)

            if not valid_items:
                return (
                    ToolResponse(text=f"添加失败：所有 item 均缺少 type_id（跳过 {skipped} 个）"),
                    0.0,
                    {"added_count": 0, "skipped": skipped}
                )

            # 将新元件添加到第一个可用的列表中
            inserted = False
            added_count = 0

            for main_key, sub_dict in current_map.items():
                if not isinstance(sub_dict, dict):
                    continue
                for sub_key, v1 in sub_dict.items():
                    if isinstance(v1, list):
                        v1.extend(valid_items)
                        inserted = True
                        added_count = len(valid_items)
                        break
                if inserted:
                    break

            # 如果结构中没找到列表，创建默认结构
            if not inserted:
                if not current_map:
                    current_map["默认场景"] = {"新增组件": valid_items}
                else:
                    first_main = list(current_map.keys())[0]
                    if isinstance(current_map[first_main], dict):
                        current_map[first_main]["新增组件"] = valid_items
                    else:
                        current_map["新增组件"] = {"列表": valid_items}
                added_count = len(valid_items)

            tool_context["current_map"] = current_map

            msg = f"成功添加 {added_count} 个元件"
            if skipped:
                msg += f"（{skipped} 个缺少 type_id 已跳过）"
            return (
                ToolResponse(text=msg),
                1.0 if added_count > 0 else 0.0,
                {"added_count": added_count, "skipped": skipped, "modified_map": current_map}
            )

        except Exception as e:
            return (
                ToolResponse(text=f"添加失败: {str(e)}"),
                0.0,
                {"error": str(e)}
            )


# ============================================================
# RetrieveAssetsTool — generate 任务专用资产检索工具
# ============================================================

class RetrieveAssetsTool(BaseTool):
    """
    资产检索工具：根据中文实体名从 qy 资产库检索 top-K 候选资产。

    - 调用 AssetRetrievalClient（默认 http://localhost:8081，见 RETRIEVE_SERVER_URL）
    - 用 PCG 白名单过滤（仅保留渲染服务支持的 type_id）
    - 从 item_infos 补充 native_bbox_m（供 agent 确定合理 Extend）
    - 返回格式与 main_distill_v4.format_retrieve_responses_for_user 完全对齐，
      agent 能直接解读并在 add 时使用正确的 type_id

    tool_context 传入字段（可选）：
        pcg_whitelist:  set of str，PCG 渲染服务支持的 type_id 集合
        pcg_item_infos: dict，item_infos 白名单元数据（用于补 native_bbox_m）
        retrieve_url:   str，检索服务地址（默认 http://localhost:8081）
    """

    DEFAULT_RETRIEVE_URL = os.environ.get("RETRIEVE_SERVER_URL", "http://localhost:8081")

    DEFAULT_SCHEMA = OpenAIFunctionToolSchema(
        type="function",
        function={
            "name": "retrieve_assets",
            "description": (
                "从 qy 轻游资产库（6559 条）检索 top-K 候选资产。"
                "必填 entity_name（中文实体名，如「高大松树」「石火盆」）；"
                "可选 top_k（默认5）、size_class（大/中/小尺寸物体）、scene_limit（室内/沙地等）。"
                "返回每条资产的 type_id/name/score/native_bbox_m/description/color，"
                "add 时 type_id 必须来自本工具返回的结果。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "要检索的中文实体名",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回候选数，默认5，上限100",
                    },
                    "size_class": {
                        "type": "string",
                        "description": "大尺寸物体 / 中尺寸物体 / 小尺寸物体（慎用，易过严）",
                    },
                    "scene_limit": {
                        "type": "string",
                        "description": "室内 / 沙地 / 雪地等场景限定（慎用）",
                    },
                },
                "required": ["entity_name"],
            },
        },
    )

    async def create(self, create_kwargs: dict = None) -> tuple[str, dict]:
        instance_id = f"retrieve_{id(self)}"
        return instance_id, {}

    async def release(self, instance_id: str) -> None:
        pass

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """执行资产检索，返回格式化文本给 agent。"""
        try:
            entity_name = parameters.get("entity_name", "").strip()
            if not entity_name:
                return (
                    ToolResponse(text="retrieve_assets 失败: 缺少 entity_name 参数"),
                    0.0,
                    {"error": "missing entity_name"},
                )

            top_k = int(parameters.get("top_k", 5))
            top_k = max(1, min(top_k, 100))
            size_class = parameters.get("size_class") or None
            scene_limit = parameters.get("scene_limit") or None

            # 从 tool_context 获取运行时配置（由 MapGenAgentLoop 注入）
            tool_context = kwargs.get("tool_context", {})
            pcg_whitelist: Optional[set] = tool_context.get("pcg_whitelist")
            pcg_item_infos: Optional[Dict] = tool_context.get("pcg_item_infos")
            retrieve_url: str = tool_context.get(
                "retrieve_url", self.DEFAULT_RETRIEVE_URL
            )

            # 导入 AssetRetrievalClient（懒加载，避免 import 时路径问题）
            _client_module = self._import_retrieval_client()
            if _client_module is None:
                return (
                    ToolResponse(text="retrieve_assets 失败: 无法导入 AssetRetrievalClient"),
                    0.0,
                    {"error": "import_failed"},
                )
            AssetRetrievalClient = _client_module

            # 白名单过滤时放大 fetch_k，保证过滤后仍有足够候选（与 main_distill_v4 对齐）
            fetch_k = max(top_k * 6, 30) if pcg_whitelist else top_k

            client = AssetRetrievalClient(base_url=retrieve_url)
            results = client.retrieve(
                entity_name=entity_name,
                top_k=fetch_k,
                size_class=size_class,
                scene_limit=scene_limit,
            )

            # 白名单过滤 + 补充 native_bbox_m，收集 top_k 条
            slim: List[Dict] = []
            for r in results:
                tid = r.get("type_id", "")
                if pcg_whitelist and tid not in pcg_whitelist:
                    continue
                entry = {
                    "type_id": tid,
                    "name": r.get("name", ""),
                    "score": round(float(r.get("score", 0.0)), 3),
                    "category_minor": r.get("category_minor"),
                    "type": r.get("type"),
                    "size_class": r.get("size_class"),
                    "placement": r.get("placement"),
                    "caption_visual": (r.get("caption_visual") or "")[:80],
                    "colors": r.get("colors"),
                    "native_bbox_m": None,
                }
                # 从 item_infos 补 native_bbox_m
                if pcg_item_infos and tid in pcg_item_infos:
                    ext = (pcg_item_infos[tid].get("BoundingBox") or {}).get("Extend") or {}
                    try:
                        entry["native_bbox_m"] = [
                            round(float(ext.get("X", 100)) / 100, 2),
                            round(float(ext.get("Y", 100)) / 100, 2),
                            round(float(ext.get("Z", 100)) / 100, 2),
                        ]
                    except (TypeError, ValueError):
                        pass
                slim.append(entry)
                if len(slim) >= top_k:
                    break

            # 格式化为 agent 可读的文本（与 format_retrieve_responses_for_user 对齐）
            response_text = self._format_retrieve_response(entity_name, slim)

            return (
                ToolResponse(text=response_text),
                0.0,   # retrieve 本身不计 reward，reward 由最终 verify 决定
                {
                    "name": "retrieve_assets",
                    "arguments": parameters,
                    "response": {"entity_name": entity_name, "results": slim},
                    "entity_name": entity_name,
                    "result_count": len(slim),
                },
            )

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return (
                ToolResponse(text=f"retrieve_assets 异常: {type(e).__name__}: {e}"),
                0.0,
                {"error": str(e), "traceback": tb[:500]},
            )

    @staticmethod
    def _import_retrieval_client():
        """懒加载 AssetRetrievalClient（来自仓库根目录的 utils/）。"""
        try:
            from asset_retrieval_client import AssetRetrievalClient
            return AssetRetrievalClient
        except ImportError:
            pass
        # verl/verl/tools/ → 向上 3 层到仓库根，再进 utils/
        try:
            _here = os.path.dirname(os.path.abspath(__file__))
            _utils = os.path.normpath(os.path.join(_here, *([".."] * 3), "utils"))
            if _utils not in sys.path:
                sys.path.insert(0, _utils)
            from asset_retrieval_client import AssetRetrievalClient
            return AssetRetrievalClient
        except ImportError:
            pass
        return None

    @staticmethod
    def _format_retrieve_response(entity_name: str, items: List[Dict]) -> str:
        """
        格式化检索结果为 agent 可读的文本。
        与 main_distill_v4.format_retrieve_responses_for_user 输出格式完全对齐，
        确保 SFT 和 RL 的 tool_response 格式一致，agent 能正确解读。
        """
        lines = [f"资产检索结果:"]
        if not items:
            lines.append(f"  [retrieve_assets({entity_name})] ❌ 未找到匹配资产（可换表述再试）")
        else:
            lines.append(f"  [retrieve_assets({entity_name})] top-{len(items)}:")
            for r in items:
                bbox = r.get("native_bbox_m")
                bbox_str = (
                    f" native_bbox(m)=[{bbox[0]},{bbox[1]},{bbox[2]}]" if bbox else ""
                )
                lines.append(
                    f"    type_id={r['type_id']} name={r['name']} score={r['score']:.3f}"
                    f" cat={r.get('category_minor')}/{r.get('type')}"
                    f" size={r.get('size_class')}{bbox_str}"
                )
                cap = r.get("caption_visual")
                colors = r.get("colors")
                if cap or colors:
                    color_str = f" color={colors}" if colors else ""
                    cap_str = f" description={cap}" if cap else ""
                    lines.append(f"        {cap_str}{color_str}")

        lines.append("⚠️ 提示:")
        lines.append(
            "  1) 请结合每条资产的 description/color，挑选风格与色调最契合当前场景主题的 type_id；"
            "若 top-K 都不契合可换表述再检索一次。"
        )
        lines.append(
            "  2) add 时建议把 Extend 设为 native_bbox_m 或在其 0.7~1.5x 范围内，"
            "避免过度拉伸（如把 3m 高的树拉到 8m）。"
        )
        return "\n".join(lines)


# ============== RetrieveAssetsTool（generate 任务专用）==============

class RetrieveAssetsTool(BaseTool):
    """
    资产检索工具：从 qy 轻游资产库检索 top-K 候选资产。

    仅用于 generate (from-scratch) 任务。agent 调用此工具获取 type_id，
    再通过 add 工具把资产摆入场景。

    检索服务地址优先级（2026-07-03 调整）：
      环境变量 RETRIEVE_SERVER_URL（脚本 run_map_gen_grpo.sh 设置）> tool_config yaml
      的 config.retrieve_url > 默认值。即脚本里的设置优先级最高，切换地址只需改脚本。
    默认 http://localhost:8081（见 RETRIEVE_SERVER_URL）。
    """

    DEFAULT_RETRIEVE_URL = os.environ.get("RETRIEVE_SERVER_URL", "http://localhost:8081")
    # agent 视觉选型需要的字段（与 main_2_v4.py 一致）
    _RETRIEVE_FIELDS = [
        "category_minor", "type", "subtype",
        "size_class", "placement", "scene_limit",
        "native_bbox_m", "caption_visual", "colors",
    ]

    def __init__(self, config=None, tool_schema=None):
        super().__init__(config=config, tool_schema=tool_schema)
        # 优先级：env RETRIEVE_SERVER_URL（脚本设置，最高）> yaml config.retrieve_url > 默认。
        # 先取 yaml/默认作 base，再让显式设置的环境变量覆盖它。
        retrieve_url = self.DEFAULT_RETRIEVE_URL
        if config and isinstance(config, dict):
            retrieve_url = config.get("retrieve_url", retrieve_url)
        env_retrieve_url = os.environ.get("RETRIEVE_SERVER_URL", "").strip()
        if env_retrieve_url:
            retrieve_url = env_retrieve_url
        self._retrieve_url = retrieve_url.rstrip("/")
        # PCG 白名单（type_id 过滤），从环境变量或默认路径加载
        self._pcg_whitelist: set = set()
        self._pcg_item_infos: dict = {}
        self._load_whitelist()

    def _load_whitelist(self):
        """加载 PCG 白名单，与 main_2_v4.py 的 _PCG_WHITELIST/_PCG_ITEM_INFOS 对齐。

        路径优先级：环境变量 RETRIEVE_WHITELIST_PATH > 默认候选路径。
        注意：白名单 item_infos 的 type_id 体系必须与检索服务
        （RETRIEVE_SERVER_URL）一致，否则检索结果会被全部过滤。
        """
        env_path = os.environ.get("RETRIEVE_WHITELIST_PATH", "").strip()
        item_infos_candidates = ([env_path] if env_path else []) + [
            # <repo-root>/render_in_blender/assets/item_infos.json
            # (this file lives at <repo-root>/verl/verl/tools/map_gen_tools.py)
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))),
                "render_in_blender", "assets", "item_infos.json"
            ),
        ]
        for p in item_infos_candidates:
            if os.path.exists(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        data = json.load(f)
                    self._pcg_item_infos = data
                    self._pcg_whitelist = set(data.keys())
                    import logging as _logging
                    _logging.getLogger(__name__).info(
                        f"[RetrieveAssetsTool] PCG 白名单加载: {len(self._pcg_whitelist)} 个 type_id, path={p}"
                    )
                    return
                except Exception as e:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(f"[RetrieveAssetsTool] 白名单加载失败: {e}")
                    return

    async def create(self, create_kwargs=None):
        instance_id = f"retrieve_{id(self)}"
        return instance_id, {}

    async def release(self, instance_id: str):
        pass

    def _format_result_for_agent(self, items: list, entity_name: str, pcg_whitelist: set) -> str:
        """
        将检索结果格式化为 agent 可读的文本，与 main_2_v4.py 的
        format_retrieve_responses_for_user 输出完全对齐。
        """
        if not items:
            return f"资产检索结果:\n[retrieve_assets({entity_name})] 未找到匹配资产，请尝试其他关键词。"

        lines = [f"资产检索结果:", f"[retrieve_assets({entity_name})] top-{len(items)}:"]
        for item in items:
            tid = str(item.get("type_id", ""))
            name = item.get("name", "?")
            score = item.get("score", 0.0)
            cat = item.get("category_minor", "")
            size = item.get("size_class", "")
            bbox = item.get("native_bbox_m")
            desc = (item.get("caption_visual") or "")[:80]
            colors = item.get("colors") or []

            # 白名单标记
            wl_mark = "" if (not pcg_whitelist or tid in pcg_whitelist) else " [⚠️不在渲染白名单]"
            bbox_str = f" native_bbox(m)={bbox}" if bbox else ""
            line = (
                f"  type_id={tid} name={name} score={score:.3f}"
                f" cat={cat} size={size}{bbox_str}{wl_mark}"
            )
            if desc:
                line += f"\n       description={desc}"
            if colors:
                line += f" color={colors}"
            lines.append(line)

        lines.append("\n请从以上结果中选择 type_id，通过 add 工具将资产摆入场景。")
        return "\n".join(lines)

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """执行资产检索。"""
        import sys as _sys
        import logging as _logging
        _logger = _logging.getLogger(__name__)

        try:
            entity_name = parameters.get("entity_name", "").strip()
            if not entity_name:
                return (
                    ToolResponse(text="retrieve_assets 失败: entity_name 不能为空"),
                    0.0,
                    {"error": "empty entity_name"},
                )

            # 动态加载 AssetRetrievalClient（避免 import 时环境未就绪）
            top_k = int(parameters.get("top_k", 5))
            top_k = max(1, min(top_k, 100))
            size_class = parameters.get("size_class") or None
            scene_limit = parameters.get("scene_limit") or None

            # [诊断] 打印真正生效的检索配置
            print(
                f"[retrieve_assets DIAG] entity={entity_name!r} "
                f"retrieve_url={self._retrieve_url} "
                f"whitelist={len(self._pcg_whitelist)}",
                flush=True,
            )

            # 动态加载 AssetRetrievalClient（避免 import 时环境未就绪）
            # verl/verl/tools/ → 向上 3 层到仓库根，再进 utils/
            _utils_dir = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "utils"
            ))
            if _utils_dir not in _sys.path:
                _sys.path.insert(0, _utils_dir)
            from asset_retrieval_client import AssetRetrievalClient

            # 召回 top_k * 4 条，过滤白名单后取 top_k（与 main_2_v4.py call_retrieve_for_fc 对齐）
            fetch_k = max(top_k * 4, 20) if self._pcg_whitelist else top_k

            client = AssetRetrievalClient(base_url=self._retrieve_url, timeout=10.0)
            try:
                raw_items = client.retrieve(
                    entity_name=entity_name,
                    top_k=fetch_k,
                    size_class=size_class,
                    scene_limit=scene_limit,
                    fields=self._RETRIEVE_FIELDS,
                )
            finally:
                client.close()

            # 白名单过滤
            if self._pcg_whitelist:
                filtered = [it for it in raw_items if str(it.get("type_id", "")) in self._pcg_whitelist]
                items = filtered[:top_k]
                _logger.info(
                    f"[RetrieveAssets] entity='{entity_name}' "
                    f"raw={len(raw_items)} → whitelist_filtered={len(filtered)} → top_k={len(items)}"
                )
            else:
                items = raw_items[:top_k]
                _logger.info(
                    f"[RetrieveAssets] entity='{entity_name}' results={len(items)} (no whitelist)"
                )

            response_text = self._format_result_for_agent(items, entity_name, self._pcg_whitelist)

            # 同步更新 tool_context["component_info"]（为本轮后续 add 工具准备 name→type_id 映射）
            tool_context = kwargs.get("tool_context", {})
            comp_info = tool_context.get("component_info", {})
            for item in items:
                name = item.get("name", "")
                if name:
                    comp_info[name] = {
                        "typeId": str(item.get("type_id", "")),
                        "native_bbox_m": item.get("native_bbox_m"),
                        "category": item.get("category_minor", ""),
                    }
            tool_context["component_info"] = comp_info

            return (
                ToolResponse(text=response_text),
                0.0,   # retrieve 本身不给 reward，reward 由最终 verify 决定
                {"entity_name": entity_name, "result_count": len(items)},
            )

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            _logging.getLogger(__name__).error(f"[RetrieveAssets] 异常: {e}\n{tb}")
            return (
                ToolResponse(text=f"retrieve_assets 失败: {type(e).__name__}: {e}"),
                0.0,
                {"error": str(e)},
            )
