import json
import logging
from typing import Any, List, Optional, Type
import httpx
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.output_parsers import PydanticOutputParser

logger = logging.getLogger(__name__)


class ChatITI(BaseChatModel):
    """Custom LangChain Chat Model Adapter for ITI Student API Gateway (http://apiaccess.iti.net.eg)."""

    model_name: str = "google.gemma-3-27b-it"
    api_key: str = ""
    base_url: str = "http://apiaccess.iti.net.eg/api/v1"
    timeout: float = 30.0
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "iti-student-chat"

    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_prompt = "You are a helpful AI assistant."
        chat_messages = []

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = str(msg.get("content", ""))
                if role == "system":
                    system_prompt = content
                elif role == "assistant":
                    chat_messages.append({"role": "assistant", "content": content})
                else:
                    chat_messages.append({"role": "user", "content": content})
            elif isinstance(msg, SystemMessage):
                system_prompt = str(msg.content)
            elif isinstance(msg, HumanMessage):
                chat_messages.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                chat_messages.append({"role": "assistant", "content": str(msg.content)})
            else:
                chat_messages.append({"role": "user", "content": getattr(msg, "content", str(msg))})

        base = self.base_url.rstrip("/")
        if base.endswith("/student/chat"):
            endpoint = base
        elif base.endswith("/api/v1"):
            endpoint = f"{base}/student/chat"
        else:
            endpoint = f"{base}/student/chat"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        payload = {
            "model_id": self.model_name,
            "messages": chat_messages,
            "system_prompt": system_prompt
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            output_text = data.get("output_text", "")

        message = AIMessage(content=output_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def with_structured_output(
        self,
        schema: Type[BaseModel],
        **kwargs: Any,
    ) -> Any:
        """Helper to enforce JSON structured output adhering to the Pydantic schema."""
        parser = PydanticOutputParser(pydantic_object=schema)
        instructions = parser.get_format_instructions()

        target_llm = self

        class StructuredOutputRunnable:
            def __init__(self, llm: ChatITI, output_parser: PydanticOutputParser):
                self.llm = llm
                self.parser = output_parser

            def invoke(self, input_val: Any, config: Optional[dict] = None) -> Any:
                if isinstance(input_val, list):
                    messages = list(input_val)
                    if messages:
                        first_msg = messages[0]
                        if isinstance(first_msg, SystemMessage):
                            messages[0] = SystemMessage(content=f"{first_msg.content}\n\n{instructions}")
                        elif isinstance(first_msg, dict) and first_msg.get("role") == "system":
                            messages[0] = {
                                "role": "system",
                                "content": f"{first_msg.get('content', '')}\n\n{instructions}"
                            }
                        else:
                            messages.insert(0, SystemMessage(content=instructions))
                    res = self.llm.invoke(messages, config=config)
                else:
                    prompt_str = f"{input_val}\n\n{instructions}"
                    res = self.llm.invoke(prompt_str, config=config)

                content = res.content if hasattr(res, "content") else str(res)
                content_clean = content.strip()
                if content_clean.startswith("```"):
                    lines = content_clean.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    content_clean = "\n".join(lines).strip()

                return self.parser.parse(content_clean)

        return StructuredOutputRunnable(target_llm, parser)
