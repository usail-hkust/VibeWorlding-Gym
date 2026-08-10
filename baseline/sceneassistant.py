#!/usr/bin/env python3
"""sceneassistant.py — SceneAssistant (arXiv:2603.12238) 复现于 VWE-Bench sandbox。

SceneAssistant = 视觉反馈 agent：VLM 每步接收渲染图，用一套原子操作迭代调整场景，直到 Finish；
最擅长"编辑已有场景"（与我们的 refine 设定最贴合）。原子操作(Create/Duplicate/Delete/
Translate/Place/Rotate/Scale/FocusOn/…) 映射到我们的工具，其中：
  - Create -> retrieve_assets + add ；Delete -> delete ；Duplicate -> add 同一 type_id 多次
  - Translate/Place/Rotate -> rotation_and_translation（绝对位姿修正）
  - Scale -> 无运行时缩放：改为 delete + 用不同 size_class 重新 retrieve_assets + add
  - FocusOn / 自由相机 -> 无（我们固定 5 视角），略去；每轮直接给全部 5 视角
论文最大步数 T_M=20；创建批次与操纵批次分开（不在同一轮混合）。

用法：
  python baseline_sceneassistant.py \
    --base_data_dir data/test_data/20260630_mix \
    --log_dir log/eval_20260630_mix/sceneassistant \
    --model_type gemini --model_name gemini-2.5-pro \
        --server http://localhost:8080 --retrieve_server http://localhost:8081
"""
from common import Controller, run_baseline, M

AGENT_ADDENDUM = """
========================= SceneAssistant 工作模式 =========================
你是一个视觉反馈驱动的 3D 场景 agent。每一步你都会收到当前场景的 5 视角渲染图（左/右/前/后/俯）、
元件信息，以及上一步的【视觉反馈】。你通过一组原子操作迭代调整场景，直到满足用户需求后停手。

原子操作（用我们的工具实现）：
- 新建物体：retrieve_assets(检索候选) 再 add(放入)，新物体默认出现在场景中心，之后再观察并摆正。
- 复制物体：add 同一 type_id 多次。
- 删除物体：delete。
- 平移/摆放/旋转：rotation_and_translation（给出目标绝对位置与朝向的修正）。
- 尺寸不对：本 sandbox 不能运行时缩放——请 delete 后用不同 size_class 重新 retrieve_assets + add。

规则：
1. 不要在同一轮里混合"新建(retrieve/add)"与"操纵(delete/摆放)"——先建、下一轮观察后再调整。
2. 每步先看渲染图判断问题，再发出对应原子操作。优先修正明显的悬空/穿模/朝向错误。
3. 【Finish】当场景已符合用户指令时，不要再调用任何工具，直接输出结论文本（终止本 case）。

严格保持工具调用格式与参数规范。
"""

CRITIC_SYS = """你是 SceneAssistant 的视觉反馈模块。给定用户指令、当前元件信息与 5 视角渲染图，
判断场景是否已满足指令，并对每个明显有问题的物体给出简明修正建议。输出简洁 JSON：
{
 "satisfied":bool,
 "per_object_issues":[{"name_or_type_id":str,"issue":"floating|clipping|wrong_orientation|wrong_position|wrong_size|misplaced","suggestion":str}],
 "missing":[{"entity_name":str,"size_class":str}],
 "redundant":[{"name_or_type_id":str,"reason":str}],
 "next_focus":str
}
"""


class SceneAssistantController(Controller):
    name = "SceneAssistant"
    max_turns = 20                           # 论文 T_M=20
    critic_system_prompt = CRITIC_SYS
    feedback_header = "【视觉反馈(逐物体问题 + 缺失/冗余)】"

    def system_prompt(self, task_setting):
        base = M.get_system_prompt("generate" if task_setting == "generate" else "refine")
        return base + "\n" + AGENT_ADDENDUM


if __name__ == "__main__":
    run_baseline(SceneAssistantController())
