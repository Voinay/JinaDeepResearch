# JinaDeepResearch

"""
title: Deep Research
author: Voinay
description: 在OpwenWebUI中支持jina-ai/node-DeepResearch的思维链和回复模型分离 - 仅支持0.5.6及以上版本
version: 1.0
licence: MIT
"""

import json
import httpx
import re
from typing import AsyncGenerator, Callable, Awaitable
from pydantic import BaseModel, Field
import asyncio


class Pipe:
    class Valves(BaseModel):
        DEEPRESEARCH: str = Field(
            default="http://localhost:3000/v1",
            description="DeepResearch本地的基础请求地址",
        )
        DEEPRESEARCH_API_MODEL: str = Field(
            default="gemini-2.0-flash",
            description="API请求的模型名称，默认为 gemini-2.0-flash ",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.data_prefix = "data:"
        self.emitter = None
        self.current_response_id = None
        self.client = httpx.AsyncClient(http2=True)  # 创建一个持久化的客户端

    def pipes(self):
        return [
            {
                "id": self.valves.DEEPRESEARCH_API_MODEL,
                "name": self.valves.DEEPRESEARCH_API_MODEL,
            }
        ]

    async def pipe(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> AsyncGenerator[str, None]:
        """主处理管道（已移除缓冲）"""
        thinking_state = {"thinking": -1}  # 使用字典来存储thinking状态
        self.emitter = __event_emitter__

        # 准备请求参数
        headers = {
            "Content-Type": "application/json",
        }

        try:
            # 模型ID提取
            model_id = body["model"].split(".", 1)[-1]
            payload = {**body, "model": model_id}

            # 提取当前消息并重置消息列表
            current_message = payload["messages"][-1]
            payload["messages"] = [current_message]

            # 发起API请求
            async with self.client.stream(
                "POST",
                f"{self.valves.DEEPRESEARCH}/chat/completions",
                json=payload,
                headers=headers,
                timeout=300,
            ) as response:
                # 错误处理
                if response.status_code != 200:
                    error = await response.aread()
                    yield self._format_error(response.status_code, error)
                    return

                # 流式处理响应
                async for line in response.aiter_lines():
                    if not line.startswith(self.data_prefix):
                        continue

                    # 截取 JSON 字符串
                    json_str = line[len(self.data_prefix) :]

                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # 格式化错误信息，这里传入错误类型和详细原因（包括出错内容和异常信息）
                        error_detail = f"解析失败 - 内容：{json_str}，原因：{e}"
                        yield self._format_error("JSONDecodeError", error_detail)
                        return

                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    response_id = data.get("id")

                    # 检查是否是新的响应
                    if response_id != self.current_response_id:
                        self.current_response_id = response_id
                        thinking_state["thinking"] = -1  # 重置思考状态

                    # 结束条件判断
                    finish_reason = choice.get("finish_reason")

                    # 状态机处理
                    state_output = await self._update_thinking_state(
                        delta, thinking_state
                    )
                    if state_output:
                        yield state_output
                        if state_output == "<think>":
                            yield "\n"

                    # 内容处理
                    content = self._process_content(delta)
                    if content:
                        yield content

                    # 在发送内容后处理结束条件
                    if finish_reason == "stop":
                        return

        except Exception as e:
            yield self._format_exception(e)

    async def _update_thinking_state(self, delta: dict, thinking_state: dict) -> str:
        """更新思考状态机（简化版）"""
        state_output = ""

        # 状态转换：未开始 -> 思考中
        if thinking_state["thinking"] == -1 and delta.get("content") == "<think>":
            thinking_state["thinking"] = 0
            state_output = "<think>"

        # 状态转换：思考中 -> 已回答
        elif thinking_state["thinking"] == 0 and delta.get("content") == "</think>\n\n":
            thinking_state["thinking"] = 1
            state_output = "\n\n"

        return state_output

    def _process_content(self, delta: dict) -> str:
        """直接返回处理后的内容"""
        return delta.get("content", "")

    def _format_error(self, status_code: int, error: bytes) -> str:
        # 如果 error 已经是字符串，则无需 decode
        if isinstance(error, str):
            error_str = error
        else:
            error_str = error.decode(errors="ignore")

        try:
            err_msg = json.loads(error_str).get("message", error_str)[:200]
        except Exception as e:
            err_msg = error_str[:200]
        return json.dumps(
            {"error": f"HTTP {status_code}: {err_msg}"}, ensure_ascii=False
        )

    def _format_exception(self, e: Exception) -> str:
        """异常格式化保持不变"""
        err_type = type(e).__name__
        return json.dumps({"error": f"{err_type}: {str(e)}"}, ensure_ascii=False)
