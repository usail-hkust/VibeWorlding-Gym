#!/usr/bin/env python3
"""sage.py — SAGE (arXiv:2602.10116) 复现于 VWE-Bench sandbox。

SAGE = agentic 场景生成：多个 generator（布局/物体组合）+ critics（语义合理/视觉真实/物理稳定），
通过迭代推理与自适应工具选择自我精炼，直到 critics 均满足（论文 Sec. 3）。实现里其实是
**两个 critic**：visual critic（语义+视觉合一，看多视角渲染给出"新增/删除/调整"建议）与
physics critic（原文用 Isaac Sim 测稳定性/碰撞）。Table 3 消融显示物理 critic 主导稳定性、
视觉 critic 降碰撞——故我们按【物理优先】排序。

映射到我们的 sandbox：
  - generators -> retrieve_assets(+add) 摆放、rotation_and_translation 重摆、delete 移除；
  - visual critic -> MLLM 看 5 视角渲染，输出 missing/remove/adjust；
  - physics critic -> 用同一 MLLM 从渲染判碰撞/悬空/高度（替代 Isaac Sim，避免 mid-loop 跑重仿真）。
  - 终止：两个 critic 均 satisfied（模型据反馈自行停手不再调工具），或到 MAX_ROUNDS。

用法：
  python baseline_SAGE.py \
    --base_data_dir data/test_data/20260630_mix \
    --log_dir log/eval_20260630_mix/sage \
    --model_type gemini --model_name gemini-2.5-pro \
        --server http://localhost:8080 --retrieve_server http://localhost:8081
"""
from common import Controller, run_baseline, M

AGENT_ADDENDUM = """
============================= SAGE 工作模式 =============================
你是一个 3D 场景构建 agent，采用 generator + critic 的自我精炼循环。
每轮你会看到：用户需求、当前元件信息、5 视角渲染图，以及【Critic 反馈】(视觉 critic + 物理 critic)。

决策规则（自适应工具选择）：
0. 【初始化 / 从零搭建】若当前场景为空：先把用户需求分解成所需物体清单，对每个物体
   retrieve_assets 后【必须紧接着 add】把它放进场景，先把初始场景搭出来。检索到 type_id 却不 add
   是错误的——retrieve 只是查询，add 才会真正放置。场景为空时【绝对不要终止、不要停手】。
1. 【物理优先】场景搭起来后再修物理问题：若物理 critic 报碰撞/悬空/高度不合理，先用
   rotation_and_translation 重摆修正；若重摆一次仍不行，则 delete 该物体后用更小 size_class
   重新 retrieve_assets + add。
2. 再修语义/视觉：视觉 critic 的 missing_objects -> retrieve_assets + add；remove_objects -> delete；
   adjust_objects(位姿/朝向/尺度不对) -> rotation_and_translation。
3. add 前必须先 retrieve_assets 拿到 type_id。
4. 【终止】仅当场景【已非空】且视觉 critic 与物理 critic 都判定满足时，才不要再调用任何工具、
   直接输出结论文本（终止本 case）。场景为空时禁止终止。

严格保持工具调用格式与参数规范。
"""

CRITIC_SYS = """你同时扮演 SAGE 的 visual critic 与 physics critic。给定用户需求、当前元件信息与 5 视角渲染图：
- visual：判断语义合理性（该有的物体是否齐全、是否生态/风格一致）与视觉真实感（尺度/朝向/排布）。
- physics：从渲染判断是否存在碰撞/穿模、悬空、高度不合理（替代物理仿真）。
输出简洁 JSON：
{
 "visual":{"satisfied":bool,
           "missing_objects":[{"entity_name":str,"size_class":str}],
           "remove_objects":[{"name_or_type_id":str,"reason":str}],
           "adjust_objects":[{"name_or_type_id":str,"issue":"wrong_scale|wrong_orientation|wrong_position|floating|clipping","suggestion":str}]},
 "physics":{"stable":bool,
            "violations":[{"name_or_type_id":str,"kind":"collision|floating|height_implausible|intersection","suggestion":"reduce_size|reposition|delete"}]},
 "both_satisfied":bool
}
"""


class SAGEController(Controller):
    name = "SAGE"
    max_turns = 8                            # 论文未给上限；6-8 足够 plan + 若干次修正
    critic_system_prompt = CRITIC_SYS
    feedback_header = "【Critic 反馈(visual + physics; 物理优先修复)】"

    def system_prompt(self, task_setting):
        base = M.get_system_prompt("generate" if task_setting == "generate" else "refine")
        return base + "\n" + AGENT_ADDENDUM


if __name__ == "__main__":
    run_baseline(SAGEController())
