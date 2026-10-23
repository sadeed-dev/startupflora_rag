"""Gemini compatibility adapter for the pinned LangChain OpenAI client.

The current LangChain serializer drops Google's per-tool extra_content.
Preserve it by tool-call ID so Gemini thought signatures survive tool round trips.
Only synchronous, non-streaming calls are used by this application's agent.
"""

from copy import deepcopy

from langchain_openai import ChatOpenAI

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiChat(ChatOpenAI):
    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump(
            exclude={"choices": {"__all__": {"message": {"parsed"}}}}, warnings=False,
        )
        for generation, choice in zip(result.generations, raw.get("choices", [])):
            extensions = {
                call["id"]: deepcopy(call["extra_content"])
                for call in (choice.get("message", {}).get("tool_calls") or [])
                if call.get("id") and call.get("extra_content")
            }
            if extensions:
                generation.message.additional_kwargs["gemini_tool_extensions"] = extensions
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(messages, stop=stop, **kwargs)
        # Google's compatibility API documents max_tokens, rather than the newer
        # OpenAI-only spelling emitted by the installed ChatOpenAI serializer.
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        payload.pop("store", None)
        for message, serialized in zip(messages, payload.get("messages", [])):
            extensions = message.additional_kwargs.get("gemini_tool_extensions", {})
            for call in serialized.get("tool_calls", []):
                if call.get("id") in extensions:
                    call["extra_content"] = deepcopy(extensions[call["id"]])
        return payload
