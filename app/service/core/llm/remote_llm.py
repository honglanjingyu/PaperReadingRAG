# app/service/core/llm/remote_llm.py

import os
import time
import logging
from typing import List, Dict, Any, Optional, Iterator
from openai import OpenAI

from .base_llm import BaseLLM

logger = logging.getLogger(__name__)


def get_llm_api_key() -> str:
    """获取 LLM API Key：优先获取专用 key，再获取通用 key"""
    return os.getenv("LLM_API_KEY") or os.getenv("MODEL_API_KEY")


def get_llm_base_url() -> str:
    """获取 LLM Base URL"""
    return os.getenv("LLM_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")


def get_llm_model_name() -> str:
    """获取 LLM 模型名称"""
    return os.getenv("LLM_MODEL", "qwen-turbo")


# P0-3：采样参数从环境变量读，降低 temperature 让同一题连跑多次答案稳定
def _env_float(name: str, default: float) -> float:
    """读浮点型环境变量，没设或解析失败时用默认值。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(f"环境变量 {name}={raw!r} 不是合法浮点数，改用默认值 {default}")
        return default


def _env_int(name: str, default: int) -> int:
    """读整型环境变量，没设或解析失败时用默认值。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(f"环境变量 {name}={raw!r} 不是合法整数，改用默认值 {default}")
        return default


def _env_bool(name: str, default: bool) -> bool:
    """读布尔型环境变量：1/true/yes/on 为真，0/false/no/off 为假，其余用默认值。"""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    logger.warning(f"环境变量 {name}={raw!r} 不是合法布尔值，改用默认值 {default}")
    return default


def get_llm_enable_thinking():
    """思考链开关：LLM_ENABLE_THINKING，默认 False（关闭）。

    qwen3 系列默认开启 thinking，实测同一段 4000 字上下文里：
    长回答 thinking 开 28.49s / 关 2.25s，首 token 几乎无差别（0.60s vs 0.64s）——
    慢在思考链逐字输出，不是慢在首包。RAG 事实问答不需要长推理，默认关掉。
    返回 True / False；返回 None 表示不透传该字段（值写成 auto，换非 qwen3 模型时用）。
    """
    raw = os.getenv("LLM_ENABLE_THINKING")
    if raw is not None and raw.strip().lower() == "auto":
        return None
    return _env_bool("LLM_ENABLE_THINKING", False)


def get_llm_temperature() -> float:
    """采样温度：LLM_TEMPERATURE，默认 0.2（低温度，答案稳定可复现）"""
    return _env_float("LLM_TEMPERATURE", 0.2)


def get_llm_max_tokens() -> int:
    """最大生成 token 数：LLM_MAX_TOKENS，默认 2048"""
    return _env_int("LLM_MAX_TOKENS", 2048)


def get_llm_top_p() -> float:
    """核采样：LLM_TOP_P，默认 0.9"""
    return _env_float("LLM_TOP_P", 0.9)


def get_llm_timeout() -> float:
    """单次请求超时（秒）：LLM_TIMEOUT，默认 60。

    OpenAI SDK 默认 600s，一次网络抖动就能把整个问答请求挂住；
    收到 60s 后交给上层业务重试。
    """
    return _env_float("LLM_TIMEOUT", 60.0)


def get_llm_max_retries() -> int:
    """业务层重试次数（不含 SDK 自带的 1 次）：LLM_MAX_RETRIES，默认 2"""
    return max(_env_int("LLM_MAX_RETRIES", 2), 0)


class RemoteLLM(BaseLLM):
    """远程 LLM 实现"""

    def __init__(
            self,
            api_key: str = None,
            base_url: str = None,
            model_name: str = None,
            temperature: float = None,
            max_tokens: int = None,
            top_p: float = None,
            max_retries: int = None
    ):
        # 优先级：参数 > 专用环境变量 > 通用环境变量 > 默认值
        self._api_key = api_key or get_llm_api_key()
        self._base_url = base_url or get_llm_base_url()
        self._model_name = model_name or get_llm_model_name()
        # P0-3：采样参数走配置。显式传 None 时读 env（LLM_TEMPERATURE /
        # LLM_MAX_TOKENS / LLM_TOP_P），env 也没设才用代码默认值
        # （temperature 0.2 / max_tokens 2048 / top_p 0.9）。
        # temperature 从 0.7 降到 0.2：事实型 RAG 问答要可复现，别让同题两次跑偏。
        self._temperature = get_llm_temperature() if temperature is None else temperature
        self._max_tokens = get_llm_max_tokens() if max_tokens is None else max_tokens
        self._top_p = get_llm_top_p() if top_p is None else top_p
        # 思考链：True/False 显式透传 enable_thinking；None（LLM_ENABLE_THINKING=auto）不透传
        self._enable_thinking = get_llm_enable_thinking()
        if self._enable_thinking is None:
            self._extra_body = None
        else:
            self._extra_body = {"enable_thinking": self._enable_thinking}
        logger.info(f"思考链 enable_thinking={self._enable_thinking}（None 表示不透传）")
        # P1-2：业务层重试次数（SDK 自身另有 1 次），默认 2
        self._max_retries = get_llm_max_retries() if max_retries is None else max(max_retries, 0)

        if not self._api_key:
            raise ValueError("未配置 LLM API Key，请设置 LLM_API_KEY 或 MODEL_API_KEY")

        if not self._base_url:
            logger.warning("未配置 LLM Base URL，将使用默认值")
            self._base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        # P1-2：默认超时是 600s，一次网络抖动就能挂住整个请求；这里收紧到 60s，
        # 并让 SDK 自己最多重试 1 次（我们上面还有一层业务重试）。
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=float(get_llm_timeout()),
            max_retries=1,
        )
        logger.info(f"远程 LLM 初始化完成: {self._model_name}, base_url={self._base_url}")

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def sampling_params(self) -> Dict[str, Any]:
        """当前生效的采样参数（P0-3：便于回显与排查）. """
        return {
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": self._top_p,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """回显模型名与当前采样参数，验收 P0-3 用。"""
        info: Dict[str, Any] = {
            "type": "remote",
            "model_name": self._model_name,
        }
        info.update(self.sampling_params)
        return info

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        # P1-2：超时/瞬时错误做有限次重试，指数退避；仍失败才返回 None（调用方按失败处理）
        max_retries = kwargs.get('max_retries', self._max_retries)
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                req = dict(
                    model=self._model_name,
                    messages=messages,
                    temperature=kwargs.get('temperature', self._temperature),
                    max_tokens=kwargs.get('max_tokens', self._max_tokens),
                    top_p=kwargs.get('top_p', self._top_p)
                )
                # 思考链开关走 extra_body 透传（OpenAI SDK 无顶层参数）
                if self._extra_body:
                    req["extra_body"] = self._extra_body
                response = self._client.chat.completions.create(**req)
                content = response.choices[0].message.content if response.choices else None
                return content
            except Exception as e:
                last_err = e
                logger.warning(
                    f"LLM 生成失败（第 {attempt + 1}/{max_retries + 1} 次）: {type(e).__name__}: {e}"
                )
                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 8))
        logger.error(f"LLM 生成最终失败（已重试 {max_retries} 次）: {last_err}")
        return None

    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
        try:
            req = dict(
                model=self._model_name,
                messages=messages,
                temperature=kwargs.get('temperature', self._temperature),
                max_tokens=kwargs.get('max_tokens', self._max_tokens),
                top_p=kwargs.get('top_p', self._top_p),
                stream=True
            )
            # 思考链开关走 extra_body 透传（OpenAI SDK 无顶层参数）
            if self._extra_body:
                req["extra_body"] = self._extra_body
            stream = self._client.chat.completions.create(**req)
            for chunk in stream:
                # 流末尾 DashScope 会额外返回一个只带 usage 的 chunk，其 choices 为空列表，
                # 直接取 choices[0] 会抛 IndexError，必须跳过
                choices = getattr(chunk, 'choices', None)
                if not choices:
                    continue
                delta = getattr(choices[0], 'delta', None)
                content = getattr(delta, 'content', None) if delta is not None else None
                if content:
                    yield content
        except Exception as e:
            # P1-2：失败要抛出去，由上层（chat.py 的 except）转成 error 事件。
            # 绝不能把错误文本 yield 进答案流——那会被当成正文拼到回答末尾。
            logger.exception(f"流式生成失败: {type(e).__name__}: {e}")
            raise