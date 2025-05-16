# EXAMPLE COMMAND: from folder examples/open_deep_research, run: python run_gaia.py --concurrency 32 --run-name generate-traces-03-apr-noplanning --model-id gpt-4o
import argparse
import json
import os
import threading
import sys  # 添加sys导入
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from phoenix.otel import register
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from scripts.reformulator import prepare_response
from scripts.run_agents import (
    get_single_file_description,
    get_zip_description,
)
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer
from tqdm import tqdm

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    LiteLLMModel,
    Model,
    ToolCallingAgent,
)
from smolagents.models import InferenceClientModel
from smolagents.models import OpenAIServerModel


load_dotenv(override=True)
register()
SmolagentsInstrumentor().instrument()
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model-id", type=str, default="o1")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--set-to-run", type=str, default="validation")
    parser.add_argument("--use-open-models", type=bool, default=False)
    parser.add_argument("--use-raw-dataset", action="store_true")
    parser.add_argument(
        "--enable-telemetry", action="store_true", help="启用Phoenix遥测功能"
    )
    parser.add_argument(
        "--task-ids",
        type=lambda s: [str(item) for item in s.split(",")],
        help="指定要运行的task_id列表，用逗号分隔",
    )
    return parser.parse_args()


### IMPORTANT: EVALUATION SWITCHES

print(
    "Make sure you deactivated any VPN like Tailscale, else some URLs will be blocked!"
)

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}


user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def distill_question(question: str, model: Model) -> Dict:
    """
    Use distiller to structure the question and extract key information

    Args:
        question: Original question text
        model: Model used to process the question

    Returns:
        Dictionary containing structured information
    """
    prompt = f"""Please analyze and structure the following question:

{question}

Please provide the following structured output:
1. Main Task: What is the main goal of this question
2. Prerequisites: What are the prerequisites needed to solve this question
3. Expected Output: What kind of answer is expected
"""

    response = model.generate(
        messages=[
            {
                "role": "system",
                "content": "You are a professional question analysis assistant, skilled at breaking down complex questions into structured components.",
            },
            {"role": "user", "content": prompt},
        ]
    )

    # 提取字符串内容，兼容ChatMessage对象
    if hasattr(response, "content"):
        response_content = response.content
    else:
        response_content = response

    # Try to parse JSON response
    try:
        result = json.loads(response_content)
    except json.JSONDecodeError:
        # If unable to parse as JSON, return original response in a simple dictionary
        result = {
            "Main Task": "Failed to parse structured information",
            "Original Response": response_content,
            "Original Question": question,
        }

    return response_content


def create_agent_team(model: Model):
    text_limit = 100000
    ti_tool = TextInspectorTool(model, text_limit)

    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    WEB_TOOLS = [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]

    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    text_webbrowser_agent.prompt_templates["managed_agent"][
        "task"
    ] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information.
    """

    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, ti_tool],
        max_steps=12,
        verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )

    #     # 修改manager_agent的提示，增加提前停止的指导
    #     manager_agent.prompt_templates["planning"][
    #         "system"
    #     ] += """
    # IMPORTANT: If at any point you determine that the task cannot be completed, or if you encounter an insurmountable obstacle,
    # do NOT continue trying endlessly. Instead:
    # 1. Clearly identify which step of your plan is failing
    # 2. Explain why it's failing (be specific about what information is missing or what approach isn't working)
    # 3. Suggest what would need to change to make progress
    # 4. Then stop execution - do not waste resources on approaches that will not work

    # This information will be used to improve the system.
    # """

    return manager_agent


def load_gaia_dataset(use_raw_dataset: bool, set_to_run: str) -> datasets.Dataset:
    if not os.path.exists("data/gaia"):
        if use_raw_dataset:
            snapshot_download(
                repo_id="gaia-benchmark/GAIA",
                repo_type="dataset",
                local_dir="data/gaia",
                ignore_patterns=[".gitattributes", "README.md"],
            )
        else:
            # WARNING: this dataset is gated: make sure you visit the repo to require access.
            snapshot_download(
                repo_id="smolagents/GAIA-annotated",
                repo_type="dataset",
                local_dir="data/gaia",
                ignore_patterns=[".gitattributes", "README.md"],
            )

    def preprocess_file_paths(row):
        if len(row["file_name"]) > 0:
            row["file_name"] = f"data/gaia/{set_to_run}/" + row["file_name"]
        return row

    eval_ds = datasets.load_dataset(
        "data/gaia/GAIA.py",
        name="2023_all",
        split=set_to_run,
        trust_remote_code=True,
        # data_files={"validation": "validation/metadata.jsonl", "test": "test/metadata.jsonl"},
    )

    eval_ds = eval_ds.rename_columns(
        {"Question": "question", "Final answer": "true_answer", "Level": "task"}
    )
    eval_ds = eval_ds.map(preprocess_file_paths)
    return eval_ds


def append_answer(entry: dict, jsonl_file: str) -> None:
    jsonl_path = Path(jsonl_file)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")
    assert jsonl_path.exists(), "File not found!"
    print("Answer exported to file:", jsonl_path.resolve())


def answer_single_question(
    example: dict,
    model_id: str,
    answers_file: str,
    visual_inspection_tool: TextInspectorTool,
) -> None:
    model_params: dict[str, Any] = {
        "model_id": model_id,
        "custom_role_conversions": custom_role_conversions,
    }
    if model_id == "o1":
        model_params["reasoning_effort"] = "high"
        model_params["max_completion_tokens"] = 8192
    else:
        model_params["max_tokens"] = 4096

    # 先检查命令行参数中是否有 api_base，如果没有则从环境变量中读取
    api_base = os.getenv("SILICONFLOW_API_BASE")
    if api_base:
        model_params["api_base"] = api_base

    api_key = os.getenv("SILICONFLOW_API_KEY")
    if api_key:
        model_params["api_key"] = api_key

    # model = LiteLLMModel(**model_params)
    model = OpenAIServerModel(**model_params)
    document_inspection_tool = TextInspectorTool(model, 100000)

    # 第一步：使用distiller结构化问题
    # structured_question = distill_question(example["question"], model)

    agent = create_agent_team(model)

    augmented_question = (
        """You have one question to answer. 
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist).
Run verification steps if needed. Here is the task:

"""
        + example["question"]
        # + "\n\nStructured Analysis:\n"
        # + json.dumps(structured_question, ensure_ascii=False, indent=2)
    )

    if example["file_name"]:
        if ".zip" in example["file_name"]:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"],
                example["question"],
                visual_inspection_tool,
                document_inspection_tool,
            )
        else:
            prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:\n"
            prompt_use_files += get_single_file_description(
                example["file_name"],
                example["question"],
                visual_inspection_tool,
                document_inspection_tool,
            )
        augmented_question += prompt_use_files

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # Run agent 🚀
        final_result = agent.run(task=augmented_question)

        agent_memory = agent.write_memory_to_messages()

        # final_result = prepare_response(
        #     augmented_question, agent_memory, reformulation_model=model
        # )

        output = str(final_result)
        for memory_step in agent.memory.steps:
            memory_step.model_input_messages = None
        intermediate_steps = agent_memory

        # Check for parsing errors which indicate the LLM failed to follow the required format
        parsing_error = (
            True
            if any(["AgentParsingError" in step for step in intermediate_steps])
            else False
        )

        # check if iteration limit exceeded
        iteration_limit_exceeded = (
            True
            if "Agent stopped due to iteration limit or time limit." in output
            else False
        )
        raised_exception = False

        # 初始化失败信息，即使正常执行也需要这些字段
        failed_step = None
        failure_reason = None

        # 分析是否有失败步骤
        for i, step in enumerate(agent.memory.steps):
            if hasattr(step, "error") and step.error is not None:
                failed_step = i
                failure_reason = str(step.error)
                break

        # 如果没有找到含有error属性的步骤，则通过关键词搜索
        if failed_step is None:
            for i, step in enumerate(agent.memory.steps):
                step_str = str(step)
                if any(
                    error_term in step_str.lower()
                    for error_term in [
                        "error",
                        "failure",
                        "failed",
                        "cannot",
                        "unable",
                        "impossible",
                    ]
                ):
                    failed_step = i
                    failure_reason = step_str
                    break
    except Exception as e:
        print("Error on ", augmented_question, e)
        output = f"未处理异常: {str(e)}"
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        failed_step = 0
        failure_reason = str(e)
        raised_exception = True

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 计算token使用情况
    try:
        token_counts_manager = agent.monitor.get_total_token_counts()
        token_counts_web = list(agent.managed_agents.values())[
            0
        ].monitor.get_total_token_counts()
        total_token_counts = {
            "input": token_counts_manager["input"] + token_counts_web["input"],
            "output": token_counts_manager["output"] + token_counts_web["output"],
        }
    except Exception as e:
        total_token_counts = {"input": 0, "output": 0, "error": str(e)}

    # 准备结果记录
    annotated_example = {
        "agent_name": model.model_id,
        "question": example["question"],
        # "structured_question": structured_question,
        "augmented_question": augmented_question,
        "prediction": output,
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": (
            str(e) if raised_exception and "exception" in locals() else failure_reason
        ),
        "failed_step": failed_step,
        "failure_reason": failure_reason,
        "task": example["task"],
        "task_id": example["task_id"],
        "true_answer": example["true_answer"],
        "start_time": start_time,
        "end_time": end_time,
        "token_counts": total_token_counts,
    }
    append_answer(annotated_example, answers_file)


def get_examples_to_answer(answers_file: str, eval_ds: datasets.Dataset) -> list[dict]:
    print(f"Loading answers from {answers_file}...")
    try:
        done_questions = pd.read_json(answers_file, lines=True)["question"].tolist()
        print(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        print("Error when loading records: ", e)
        print("No usable records! ▶️ Starting new.")
        done_questions = []
    return [
        line for line in eval_ds.to_list() if line["question"] not in done_questions
    ]


def main():
    # 硬编码的参数配置
    args = type(
        "Args",
        (),
        {
            "concurrency": 1,  # 并发数
            "model_id": "Qwen/Qwen2.5-72B-Instruct-128K",  # 模型ID
            "run_name": "validation-1",  # 运行名称
            "set_to_run": "validation",  # 数据集，只能是 validation 或 test
            "use_open_models": False,  # 是否使用开源模型
            "use_raw_dataset": False,  # 是否使用原始数据集
            "enable_telemetry": True,  # 是否启用遥测
            "task_ids": [
                # "17b5a6a3-bc87-42e8-b0fb-6ab0781ef2cc",
                # "e1fc63a2-da7a-432f-be78-7c4a95598703",
                # "8e867cd7-cff9-4e6c-867a-ff5ddc2550be",
                # "bec74516-02fc-48dc-b202-55e78d0e17cf",
                # "84d0dd8-e8a4-4cfe-963c-d37f256e7662",
            ],  # 指定要运行的task_id列表
        },
    )()

    print(f"Starting run with configuration: {args.__dict__}")

    # 如果启用遥测，启动Phoenix服务器
    # if args.enable_telemetry:
    #     try:
    #         import subprocess
    #         import threading

    #         def run_phoenix_server():
    #             subprocess.run([sys.executable, "-m", "phoenix.server.main", "serve"])

    #         # 在后台线程中启动Phoenix服务器
    #         phoenix_thread = threading.Thread(target=run_phoenix_server, daemon=True)
    #         phoenix_thread.start()
    #         print("Phoenix telemetry server started in background")
    #     except Exception as e:
    #         print(f"Warning: Failed to start Phoenix server: {e}")
    #         print("Continuing without telemetry...")

    eval_ds = load_gaia_dataset(args.use_raw_dataset, args.set_to_run)
    print("Loaded evaluation dataset:")
    print(pd.DataFrame(eval_ds)["task"].value_counts())

    # 确保输出目录存在
    output_dir = f"output/{args.set_to_run}"
    os.makedirs(output_dir, exist_ok=True)
    answers_file = f"{output_dir}/{args.run_name}.jsonl"

    tasks_to_run = get_examples_to_answer(answers_file, eval_ds)
    # 如果指定了task_ids，只运行这些tasks；否则运行所有tasks
    if args.task_ids is not None and len(args.task_ids) > 0:
        tasks_to_run = [
            task for task in tasks_to_run if task["task_id"] in args.task_ids
        ]
        if not tasks_to_run:
            print(f"Warning: No tasks found with task_ids {args.task_ids}")
            return
        print(f"Running tasks with task_ids: {args.task_ids}")
    else:
        print("Running all tasks")
    print(f"Found {len(tasks_to_run)} tasks to run")

    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(
                answer_single_question, example, args.model_id, answers_file, visualizer
            )
            for example in tasks_to_run
        ]
        for f in tqdm(
            as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"
        ):
            f.result()

    print("All tasks processed.")


if __name__ == "__main__":
    main()
