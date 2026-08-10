import json

def make_hashable(obj):
    """递归将不可哈希的对象（列表、字典）转为可哈希的元组"""
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, list):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, (tuple, set)):
        return tuple(make_hashable(item) for item in obj)
    else:
        return obj

def get_match_key(item_dict):
    """提取核心字段生成匹配哈希键"""
    if not isinstance(item_dict, dict):
        return None
    target_fields = (
        item_dict.get("name"),
        item_dict.get("pos"),
        item_dict.get("Extend")
    )
    return make_hashable(target_fields)


def rotation_and_translation(llm_output, corrections=None, **kwargs):
    """处理组件的旋转和平移、属性修改等"""
    if not corrections:
        return llm_output

    # 预处理：生成 原数据哈希 -> 修正值 的映射
    corr_map = {}
    for item in corrections:
        original_data = item.get("original_data")
        if not original_data: 
            continue
        key = get_match_key(original_data)
        corr_map[key] = item.get("modified_data", {})

    for main_key, sub_dict in llm_output.items():
        if not isinstance(sub_dict, dict): continue
        for sub_key, v1 in sub_dict.items():
            if not isinstance(v1, list): continue
            
            new_v1 = []
            for elem in v1:
                elem_key = get_match_key(elem)
                if elem_key in corr_map:
                    corrected_val = corr_map[elem_key]
                    if isinstance(corrected_val, list):
                        new_v1.extend(corrected_val)
                    elif isinstance(corrected_val, dict):
                        # 如果修改了，保存新状态；如果字典连名字都没了，等同删除。
                        if "name" in corrected_val:
                            new_v1.append(corrected_val)
                else:
                    new_v1.append(elem)
            sub_dict[sub_key] = new_v1

    return llm_output


def delete(llm_output, modified_data=None, **kwargs):
    """
    处理组件的删除。传入的是 modified_data 列表，
    我们将比对 name, pos, Extend 从而剔除。
    """
    if not modified_data:
        return llm_output

    delete_keys = set()
    for item in modified_data:
        delete_keys.add(get_match_key(item))

    for main_key, sub_dict in llm_output.items():
        if not isinstance(sub_dict, dict): continue
        for sub_key, v1 in sub_dict.items():
            if not isinstance(v1, list): continue
            
            new_v1 = []
            for elem in v1:
                elem_key = get_match_key(elem)
                # 只有不在删除列表里的，才会被重新放回
                if elem_key not in delete_keys:
                    new_v1.append(elem)
            sub_dict[sub_key] = new_v1

    return llm_output


def add(llm_output, modified_data=None, **kwargs):
    """处理组件的新增。

    每个 item 必须同时包含 name 和 type_id（来自 retrieve_assets 返回）。
    type_id 会保留在 map JSON 中，供后续渲染时 component_info fallback 使用。
    缺少 type_id 的 item 会记录警告后跳过。
    typeId 字段会自动规范化为 type_id。
    """
    if not modified_data:
        return llm_output

    import logging as _log
    valid_items = []
    for item in modified_data:
        if not isinstance(item, dict):
            continue
        # 统一 typeId → type_id
        if item.get("typeId") and not item.get("type_id"):
            item["type_id"] = str(item.pop("typeId"))
        if not item.get("type_id"):
            _log.warning(
                f"[add] item 缺少 type_id，已跳过: name={item.get('name', '?')} "
                f"pos={item.get('pos', '?')}"
            )
            continue
        item["type_id"] = str(item["type_id"])
        valid_items.append(item)

    if not valid_items:
        return llm_output
        return llm_output

    inserted = False
    for main_key, sub_dict in llm_output.items():
        if not isinstance(sub_dict, dict):
            continue
        for sub_key, v1 in sub_dict.items():
            if isinstance(v1, list):
                v1.extend(valid_items)
                inserted = True
                break
        if inserted:
            break

    if not inserted:
        if not llm_output:
            llm_output["默认场景"] = {"新增组件": valid_items}
        else:
            first_main = list(llm_output.keys())[0]
            if isinstance(llm_output[first_main], dict):
                llm_output[first_main]["新增组件"] = valid_items
            else:
                llm_output["新增组件"] = {"列表": valid_items}

    return llm_output