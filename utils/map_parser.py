"""map_parser.py — 把 agent 输出的 3D 世界 map 解析成 PCG actors。

`parse2pcg()` 是渲染链路的入口：它把 LLM 产出的中文结构化 3D map
（自然元件 / 建筑元件 …）连同 component_info 展开成 PCG 引擎需要的 actor列表，
再由 utils/pcg_render.py 交给 Blender 渲染服务出图。

散布类元件（成片的树/花/草等）会按规则程序化生成实例位置，并用 scatter_cache
保证同一场景多轮渲染之间的布局稳定（否则每轮抖动，agent 无法判断自己的修改效果）。
"""

import json
import logging
import math
import os
import random

import numpy as np


def hex_to_rgba_255(hex_code):
    """把6 位（无 #）十六进制颜色码转成 [R, G, B, 255]。"""
    if not isinstance(hex_code, str) or len(hex_code) != 6:
        raise ValueError(f"颜色码 {hex_code} 格式错误，必须是 6 位无 # 号的十六进制字符串")
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    return [r, g, b, 255]


def update_component_info(llm_plan_sample, component_info_sample, re_colors={}):
    """用 llm 输出的 pos / rotate / Extend 覆盖 component_info 里的资产模板，产出一个 actor。

    Extend 是米制的目标尺寸，component_info 的 box 是厘米制的原始 bbox，
    两者相除得到缩放比例 sca。
    """
    box_old = component_info_sample["box"]
    box_nex = [x * 100 for x in llm_plan_sample["Extend"]]
    sca = []
    for o, n in zip(box_old, box_nex):
        # 防御：某些资产的原始 bbox 在某维度为 0（如"喷泉03损毁"），
        # 直接 n/o 会 ZeroDivisionError 拖垮整个 case。此时降级为比例 1.0。
        if not o:
            sca.append(1.0)
        else:
            sca.append(n / o)
    return dict(
        c=component_info_sample["c"],
        name=component_info_sample["name"],
        pos=llm_plan_sample["pos"],
        rot=(euler_to_quaternion(llm_plan_sample["rotate"])
             if "rotate" in llm_plan_sample else component_info_sample["rot"]),
        gname=component_info_sample["gname"],
        id=component_info_sample["id"],
        m=component_info_sample["m"],
        col=([hex_to_rgba_255(code) for code in re_colors[component_info_sample["name"]]]
             if component_info_sample["name"] in re_colors else component_info_sample["col"]),
        typeId=component_info_sample["typeId"],
        sca=sca,
    )


def euler_to_quaternion(euler):
    """
    欧拉角转四元数
    :param euler: 列表 [rx, ry, rz]  单位：角度(°)  含义：绕X轴角度、绕Y轴角度、绕Z轴角度
    :return: 列表 [x, y, z, w] 四元数，可直接赋值给JSON的rot字段
    """
    # 1. 角度转弧度（编程语言三角函数默认用弧度计算）
    rx = math.radians(euler[0])
    ry = math.radians(euler[1])
    rz = math.radians(euler[2])

    # 2. 计算半角的正弦、余弦值（核心公式）
    sx = math.sin(rx / 2)
    cx = math.cos(rx / 2)
    sy = math.sin(ry / 2)
    cy = math.cos(ry / 2)
    sz = math.sin(rz / 2)
    cz = math.cos(rz / 2)

    # 3. 欧拉角转四元数 标准公式 (XYZ顺规，3D引擎通用)
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    w = cx * cy * cz + sx * sy * sz
    # 返回四元数 [x,y,z,w]
    return [round(i, 4) for i in [x, y, z, w]]



def generate_pine_trees_loose_distribution(input_rule):
    config = {
        "name": input_rule["name"],
        "pos_range": {
            "x": [input_rule["pos"][0][0], input_rule["pos"][1][0]],
            "y": [input_rule["pos"][0][1], input_rule["pos"][1][1]],
            "z": input_rule["pos"][0][2]
        },
        "extend_range": {
            "x": [input_rule["Extend"][0][0], input_rule["Extend"][1][0]],
            "y": [input_rule["Extend"][0][1], input_rule["Extend"][1][1]],
            "z": [input_rule["Extend"][0][2], input_rule["Extend"][1][2]]
        },
        "interval_range": {
            "x": [input_rule["Interval"][0][0], input_rule["Interval"][1][0]],
            "y": [input_rule["Interval"][0][1], input_rule["Interval"][1][1]],
        },
        "num": input_rule["num"]
    }

    pine_trees = []

    # 【关键参数】候选采样次数
    candidates_per_attempt = 20

    # 限制最大尝试次数，避免 num 过大时（如 2000）导致循环 100 万次卡住
    # 500→150：注定放不满的复杂场景常跑满上限也只放出 0/N，砍预算几乎不损质量、大幅提速
    max_attempts = min(config["num"] * 150, 100000)
    total_attempts = 0

    # 时间限制：单次撒点最多 8 秒（原 30s；CPU 争用时超时检查被拖过头，见下 %200）
    import time as _time
    _scatter_start_time = _time.time()
    _scatter_timeout = 8  # 秒

    # 当前间隔缩放因子（空间不够时逐步缩减间隔）
    interval_scale = 1.0

    # 辅助函数：生成一个随机的候选树数据（坐标+尺寸）
    def get_random_candidate():
        ext = [
            round(random.uniform(*config["extend_range"]["x"]), 3),
            round(random.uniform(*config["extend_range"]["y"]), 3),
            round(random.uniform(*config["extend_range"]["z"]), 3)
        ]
        p = [
            round(random.uniform(*config["pos_range"]["x"]), 3),
            round(random.uniform(*config["pos_range"]["y"]), 3),
            config["pos_range"]["z"]
        ]
        return {"pos": p, "Extend": ext}

    # 记录上一次成功放置时的 total_attempts，用于检测是否陷入停滞
    last_success_at = 0

    # 循环生成
    while len(pine_trees) < config["num"]:
        total_attempts += 1
        if total_attempts > max_attempts:
            logging.warning(
                f"撒点 [{config['name']}]: 达到最大尝试次数 {max_attempts}，"
                f"已生成 {len(pine_trees)}/{config['num']}，"
                f"interval_scale={interval_scale:.2f}"
            )
            break

        # 时间超限保护：避免单个撒点规则卡住整个进程
        # %1000→%200：检查更频繁，防止 CPU 被抢时 8s 超时被拖到 1-2 分钟
        if total_attempts % 200 == 0 and (_time.time() - _scatter_start_time) > _scatter_timeout:
            logging.warning(
                f"撒点 [{config['name']}]: 超时 {_scatter_timeout}s，"
                f"已生成 {len(pine_trees)}/{config['num']}，"
                f"total_attempts={total_attempts}"
            )
            break

        # 如果连续很多次都没成功放置，缩减间隔
        if total_attempts - last_success_at > config["num"] * 50 and interval_scale > 0.3:
            interval_scale *= 0.7
            last_success_at = total_attempts  # 重置以避免连续缩减
            logging.info(
                f"撒点 [{config['name']}]: 放置停滞，缩减间隔 → scale={interval_scale:.2f}"
            )

        best_candidate = None
        max_dist_to_closest_neighbor = -1

        current_samples = 1 if len(pine_trees) == 0 else candidates_per_attempt

        for _ in range(current_samples):
            candidate = get_random_candidate()
            pos = candidate["pos"]
            extend = candidate["Extend"]

            is_in_bound = (
                    config["pos_range"]["x"][0] + extend[0] / 2 <= pos[0] <= config["pos_range"]["x"][1] - extend[0] / 2
                    and
                    config["pos_range"]["y"][0] + extend[1] / 2 <= pos[1] <= config["pos_range"]["y"][1] - extend[1] / 2
            )
            if not is_in_bound:
                continue

            if len(pine_trees) == 0:
                best_candidate = candidate
                break

            min_dist_for_this_candidate = float('inf')

            for existing in pine_trees:
                d = math.sqrt((pos[0] - existing["pos"][0]) ** 2 + (pos[1] - existing["pos"][1]) ** 2)
                if d < min_dist_for_this_candidate:
                    min_dist_for_this_candidate = d

            if min_dist_for_this_candidate > max_dist_to_closest_neighbor:
                max_dist_to_closest_neighbor = min_dist_for_this_candidate
                best_candidate = candidate

        if best_candidate is None:
            continue

        final_pos = best_candidate["pos"]
        final_extend = best_candidate["Extend"]
        is_overlap = False

        for existing_tree in pine_trees:
            dx = abs(final_pos[0] - existing_tree["pos"][0])
            dy = abs(final_pos[1] - existing_tree["pos"][1])

            current_interval_x = random.uniform(*config["interval_range"]["x"]) * interval_scale
            current_interval_y = random.uniform(*config["interval_range"]["y"]) * interval_scale

            min_safe_dx = (final_extend[0] / 2 + existing_tree["Extend"][0] / 2) + current_interval_x
            min_safe_dy = (final_extend[1] / 2 + existing_tree["Extend"][1] / 2) + current_interval_y

            if dx < min_safe_dx and dy < min_safe_dy:
                is_overlap = True
                break

        if not is_overlap:
            pine_trees.append({
                "name": config["name"],
                "pos": final_pos,
                "Extend": final_extend
            })
            last_success_at = total_attempts

    return {
        "input_rule": input_rule,
        "generated_pine_trees": pine_trees
    }

def pcg_generate(pcg_json):
    data = {"actors": pcg_json}
    escaped_string_ascii = json.dumps(data, ensure_ascii=True)
    result = [
        {
            "Type": "PCGGenerate",
            "Param": {
                "AIID": "398a907a-c40f-477c-b278-734e42af1df5",
                "bNeedGroup": False,
                "Info": escaped_string_ascii
            }
        }
    ]
    return result


def _make_scatter_cache_key(v2):
    """为撒点规则生成稳定的缓存键（基于 name + pos + Extend + Interval + num）"""
    import hashlib
    key_data = json.dumps({
        "name": v2.get("name"),
        "pos": v2.get("pos"),
        "Extend": v2.get("Extend"),
        "Interval": v2.get("Interval"),
        "num": v2.get("num"),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(key_data.encode()).hexdigest()


def _build_scatter_name_num_index(scatter_cache):
    """把预构建的 scatter_cache 按 (撒点组名, 实例数) 重新索引。

    scatter_cache 是 {不透明hash -> 实例列表}；每个实例带其组的 name，列表长度即
    该组的 num。历史上写 cache 用的 hash key 与现在算的不一致（键函数改过），导致
    100% miss、每次渲染重新随机撒点。改用 (name, len) 索引即可稳定命中预构建结果，
    与 init_image 保持一致。返回 {(name, num): 实例列表}。同一 cache 内 (name,num)
    唯一（数据已验证），冲突时保留首个。
    """
    idx = {}
    if not scatter_cache:
        return idx
    for instances in scatter_cache.values():
        if not instances:
            continue
        key = (instances[0].get("name"), len(instances))
        if key not in idx:
            idx[key] = instances
    return idx


def parse2pcg(llm_output, component_info, re_colors={}, see_detail=[], scatter_cache=None):
    """
    将 llm_output 转换为 PCG 渲染格式。

    Args:
        scatter_cache: 可选的撒点缓存字典。如果提供，则对带 num 字段的撒点元件
                       首次生成后缓存结果，后续相同规则直接复用，避免每轮重新随机
                       导致元件位置/数量变化。
    """
    result = []
    asset_list = []
    detail = []
    use_img = []
    # 预构建 cache 的 (name, num) 索引：优先按此命中，避免历史 hash key 失配。
    scatter_name_num_idx = _build_scatter_name_num_index(scatter_cache)
    for k, v in llm_output.items():
        for k1, v1 in v.items():
            for v2 in v1:
                elem_name = v2.get("name", "")
                # 跳过不在 component_info 中的元件（agent 可能自创了不存在的元件名）
                if elem_name not in component_info:
                    logging.warning(f"parse2pcg: 跳过未知元件 '{elem_name}'（不在 component_info 中）")
                    continue
                use_img.append(os.path.join("data/imgs", f'{elem_name}.png'))
                asset_list.append(elem_name)
                if "num" in v2:
                    # 撒点结果解析优先级：
                    #   1) 按 (name, num) 命中预构建 cache —— 与 init_image 一致；
                    #   2) 退回历史 hash key（兼容老路径写入的条目）；
                    #   3) 都未命中才重新随机生成，并同时写回两种键。
                    nn_key = (v2.get("name"), v2.get("num"))
                    temp = scatter_name_num_idx.get(nn_key)
                    if temp is None:
                        cache_key = _make_scatter_cache_key(v2)
                        if scatter_cache is not None and cache_key in scatter_cache:
                            temp = scatter_cache[cache_key]
                        else:
                            temp = generate_pine_trees_loose_distribution(v2)["generated_pine_trees"]
                            if scatter_cache is not None:
                                scatter_cache[cache_key] = temp
                                scatter_name_num_idx[nn_key] = temp
                    for temp_v2 in temp:
                        component_info_sample = component_info[temp_v2["name"]]
                        result.append(update_component_info(temp_v2, component_info_sample, re_colors))
                    continue
                
                # 兼容 LLM 输出 Extend 为嵌套列表的情况，如 [[x, y, z]] -> [x, y, z]
                if isinstance(v2.get("Extend"), list) and len(v2["Extend"]) > 0 and isinstance(v2["Extend"][0], list):
                    v2["Extend"] = v2["Extend"][0]
                component_info_sample = component_info[elem_name]
                if component_info_sample["name"] in see_detail:
                    detail.append(update_component_info(v2, component_info_sample, re_colors))
                result.append(update_component_info(v2, component_info_sample, re_colors))
    
    keys_to_flatten = ["pos", "rot", "sca", "Extend"]
    
    # 清理 result 列表
    for item in result:
        for key in keys_to_flatten:
            if key in item:
                val = item[key]
                # 判断条件：是列表 && 长度为 1 && 内部的单一元素也是列表
                if isinstance(val, list) and len(val) == 1 and isinstance(val[0], list):
                    item[key] = val[0] # 提取内部列表，完成降维
                    
    # 清理 detail 列表（保持数据结构一致性）
    for item in detail:
        for key in keys_to_flatten:
            if key in item:
                val = item[key]
                if isinstance(val, list) and len(val) == 1 and isinstance(val[0], list):
                    item[key] = val[0]
    
    return result, list(set(asset_list)), detail, list(set(use_img))

# ==========================================
# 新增的封装函数 1：获取 component_info
# ==========================================
