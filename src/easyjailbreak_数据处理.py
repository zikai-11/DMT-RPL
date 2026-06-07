import json


def convert_jailbreak_format(input_file_path, output_file_path, max_count=50):

    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 自动判断格式
    if isinstance(data, list):
        data_list = data

    elif isinstance(data, dict) and "root" in data:
        data_list = data["root"]

    else:
        raise ValueError(
            f"无法识别格式，根类型为 {type(data)}，键为 "
            f"{list(data.keys()) if isinstance(data, dict) else 'N/A'}"
        )

    jailbreak_prompts = []

    for item in data_list:

        if not isinstance(item, dict):
            continue

        jb = item.get("jailbreak", "").strip()

        if jb:
            jailbreak_prompts.append(jb)

        if len(jailbreak_prompts) >= max_count:
            break

    output_data = {
            "prompts": jailbreak_prompts
        }


    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"提取成功，共 {len(jailbreak_prompts)} 条")

if __name__ == "__main__":

    INPUT_FILE = "data/llama-3_test (1).json"
    OUTPUT_FILE = "data/llama-3_pair.json"

    convert_jailbreak_format(
        INPUT_FILE,
        OUTPUT_FILE,
        max_count=50
    )