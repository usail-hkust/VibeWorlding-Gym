#!/usr/bin/env python3
"""sceneweaver.py — SceneWeaver (arXiv:2509.20414) 复现于 VWE-Bench sandbox。

SceneWeaver = 反射式 agent：closed-loop **reason -> act -> reflect**，由 LLM planner
从工具集中选一个工具执行，再由 MLLM 对场景做自评（物理合理性 / 视觉真实感 / 语义对齐），
反馈驱动下一步；memory length l=1，最大迭代 T=10（见论文 Sec. 3.2 / 4.3）。

映射到我们的 sandbox：工具集合 = retrieve_assets / add / delete / rotation_and_translation；
每轮渲染 5 视角作为观测；reflect 用一个 MLLM critic 打分（realism / functionality /
layout / completion + 物理门：碰撞/悬空/高度），并把"最低分维度 + 建议 + 若某问题上一步
未解决则降低该工具置信度 + 若无明显问题则停手(不再调用工具)"作为反馈注入下一轮 planner。

简化说明（我们缺其数据驱动生成子模型 / 物理求解器）：
  - 所有 Initializer/Implementer/Refiner 折叠为 retrieve+add+transform+delete；
  - "physics-aware executor" 的自动消碰折叠为 planner 依据碰撞反馈发 rotation_and_translation；
  - 硬 rollback（状态回退）近似为：planner 被指示用 delete/重摆撤销明显变差的编辑。

用法：
  python baseline_sceneweaver.py \
    --base_data_dir data/test_data/20260630_mix \
    --log_dir log/eval_20260630_mix/sceneweaver \
    --model_type gemini --model_name gemini-2.5-pro \
        --server http://localhost:8080 --retrieve_server http://localhost:8081
"""
from common import Controller, run_baseline, M

PLANNER_ADDENDUM = """
========================= SceneWeaver 工作模式 =========================
你是一个自反思的 3D 场景构建 agent，遵循闭环：REASON -> ACT -> REFLECT。
每一轮你会看到：用户需求、当前场景的结构化元件信息、5 视角渲染图（左/右/前/后/俯），
以及上一轮场景的【评审反馈】（各维度分数 0-10 + 物理问题 + 改进建议）。

每一轮请：
1. 简述上一步工具做了什么、是否解决了上一轮指出的问题。
2. 找出当前【最严重的一个问题】：优先看物理门（碰撞/悬空/高度不合理），再看评审里最低分的维度
   （realism / functionality / layout / completion）。
3. 只针对这一个问题，选择最合适的一个工具并调用（retrieve_assets->add 补缺、delete 去除不合理、
   rotation_and_translation 修正位姿/消碰）。若同一问题上一步已尝试但未解决，请换一种工具/策略。
4. 若明显做坏了（新引入碰撞/场景更差），下一步用 delete 或重新摆放撤销该改动。
5. 【停止条件】若场景已无明显问题、或仅剩微小改进、或进一步改动可能变差：不要再调用任何工具，
   直接输出结论文本即可（这会终止本 case）。

严格保持工具调用格式与参数规范。最多迭代若干轮，尽早收敛。
"""

CRITIC_SYS = """你是严格的 3D 场景评审员（SceneWeaver reflect 模块）。给定用户需求、当前元件信息与 5 视角渲染图，
对场景逐维度打分（0-10 整数），忽略贴图/光照/门：
- realism        场景是否真实可信、物体是否符合现实逻辑（物体类型离谱 -> <5）
- functionality  是否具备满足用户意图所需的关键物体（缺关键物体 -> <6）
- layout         布局是否合理、位姿是否正确（悬空/穿模/碰撞/朝向错/过挤 -> 不高于5）
- completion     场景是否充实（空旷>50% -> <5）
另给物理门判断：collision(是否有穿模/碰撞)、floating(是否悬空)、height(高度是否合理)。
输出简洁 JSON：
{"realism":{"grade":int,"comment":str,"suggestion":str},
 "functionality":{...},"layout":{...},"completion":{...},
 "physics":{"collision":bool,"floating":bool,"height_ok":bool,"comment":str},
 "most_serious_problem":str, "advice_tool":"retrieve_assets|add|delete|rotation_and_translation|none"}
suggestion 里请指明该用哪个工具修复。"""


class SceneWeaverController(Controller):
    name = "SceneWeaver"
    max_turns = 10                          # 论文 T=10
    critic_system_prompt = CRITIC_SYS
    feedback_header = "【评审反馈(reflect, 0-10 分 + 物理门 + 建议)】"

    def system_prompt(self, task_setting):
        base = M.get_system_prompt("generate" if task_setting == "generate" else "refine")
        return base + "\n" + PLANNER_ADDENDUM


if __name__ == "__main__":
    run_baseline(SceneWeaverController())
