import json
import argparse

def main():
    parser = argparse.ArgumentParser() #创建解释器对象

    #定义参数
    parser.add_argument("--logfile", type=str,default="../../data/DeepInception-vicuna.json",help='Path to the logfile')
    parser.add_argument("--results_logfile", type=str,default="../../data/DeepInception-vicuna_1.json", help='Path to the attack logfile')
    parser.add_argument("--model",type=str,default="null",help='Name of the model')
    #解析命令行参数
    args = parser.parse_args()

    with open(args.logfile, 'r') as f:
        log = json.load(f)

    
    gcg_individual = [data for data in log ] #取出vicuna模型的数据

    target = [content["target"] for content in gcg_individual]
    goals = []
    controls = []
    prompts = []
    for content in gcg_individual:
        goal = content["goal"]
        prompt = content["prompt"]
        control = prompt.replace(goal,"").strip()
        goals.append(goal)
        controls.append(control)
        prompts.append(prompt)
    with open(args.results_logfile,"w",encoding="utf_8") as result:
        json.dump({
                "goal":goals,
                "target":target,
                "controls":controls,
                "prompts":prompts
        },result,ensure_ascii=False,indent=4) #ensure_ascii=False 中文不进行转义



if __name__ == "__main__":
    main()
