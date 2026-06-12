import logging
import re
from abc import ABC, abstractmethod
from typing import Type, List, Dict, Any, Optional
from pydantic import BaseModel
from openai import OpenAI

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_answer(
        self, 
        prompt: str, 
        system_prompt: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generates text answer based on context prompt, system instructions, and optional chat history.
        """
        pass

    @abstractmethod
    async def generate_structured_response(
        self, 
        prompt: str, 
        system_prompt: str, 
        response_schema: Type[BaseModel]
    ) -> BaseModel:
        """
        Generates structured response mapped to a Pydantic schema model.
        """
        pass


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    async def generate_answer(
        self, 
        prompt: str, 
        system_prompt: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        
        # Inject conversation history if available
        if history:
            for message in history:
                messages.append({"role": message["role"], "content": message["content"]})
                
        messages.append({"role": "user", "content": prompt})

        try:
            # Run in executor or call synchronously since OpenAI client performs blocking requests
            # For simplicity, call directly; in production, wrap in run_in_executor
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI generate_answer failure: {e}", exc_info=True)
            raise RuntimeError(f"LLM execution failed: {e}")

    async def generate_structured_response(
        self, 
        prompt: str, 
        system_prompt: str, 
        response_schema: Type[BaseModel]
    ) -> BaseModel:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            # Use Beta Structured Outputs feature in newer OpenAI SDKs
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_schema,
                temperature=0.0
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Parsed structured output returned None")
            return parsed
        except Exception as e:
            logger.warning(f"Structured parse failed, attempting fallback parsing: {e}")
            # Fallback to standard JSON completion if parse fails
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                raw_json = completion.choices[0].message.content or "{}"
                return response_schema.model_validate_json(raw_json)
            except Exception as e_inner:
                logger.error(f"Fallback structured parse also failed: {e_inner}", exc_info=True)
                raise RuntimeError(f"LLM structured completion failed: {e_inner}")


class MockLLMProvider(BaseLLMProvider):
    async def generate_answer(
        self, 
        prompt: str, 
        system_prompt: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        # Determine what source indexes are in the context
        source_indices = re.findall(r'Source \[([0-9]+)\]', system_prompt)
        
        if not source_indices:
            return "I could not find sufficient evidence in the retrieved sources."
            
        # Build a deterministic grounded answer using retrieved document content
        ans_parts = []
        for idx in source_indices:
            content_match = re.search(fr'Source \[{idx}\]:.*?Content:\n(.*?)(?:\n\nSource \[\d+\]:|\Z)', system_prompt, re.DOTALL)
            if content_match:
                content_text = content_match.group(1).strip()
                # Extract first 2 sentences to form a context-aware summary
                sentences = [s.strip() for s in content_text.split('.') if s.strip()]
                summary = ". ".join(sentences[:2])
                if summary:
                    # Append citation marker properly
                    ans_parts.append(f"{summary} [{idx}].")
            
        return " ".join(ans_parts)

    async def generate_structured_response(
        self, 
        prompt: str, 
        system_prompt: str, 
        response_schema: Type[BaseModel]
    ) -> BaseModel:
        mock_data = {}
        for name, field_info in response_schema.model_fields.items():
            annotation = field_info.annotation
            origin = getattr(annotation, "__origin__", None)
            
            # Unwrap Union / Optionals
            if origin is not None:
                args = getattr(annotation, "__args__", [])
                non_none = [a for a in args if a is not type(None)]
                if non_none:
                    annotation = non_none[0]
            
            if annotation == str:
                mock_data[name] = f"Mock structured string for {name}."
            elif annotation == int:
                mock_data[name] = 1
            elif annotation == float:
                mock_data[name] = 0.95
            elif annotation == bool:
                mock_data[name] = True
            elif getattr(annotation, "__origin__", None) == list:
                mock_data[name] = []
            else:
                mock_data[name] = None
                
        return response_schema.model_validate(mock_data)


class LLMService:
    _provider: Optional[BaseLLMProvider] = None

    @classmethod
    def get_provider(cls) -> BaseLLMProvider:
        """
        Dynamically initializes the configured LLM provider.
        """
        if cls._provider is None:
            provider_type = settings.LLM_PROVIDER.lower().strip()
            api_key = settings.OPENAI_API_KEY
            
            # Detect mock indicator keys
            mock_indicators = [
                "mock-api-key-for-development",
                "your-actual-api-key-here",
                "mock-key",
                ""
            ]
            is_mock_key = not api_key or api_key.strip() in mock_indicators
            
            if provider_type == "mock" or (provider_type == "auto" and is_mock_key):
                logger.info("Using MockLLMProvider")
                cls._provider = MockLLMProvider()
            else:
                logger.info("Using OpenAIProvider")
                cls._provider = OpenAIProvider(
                    api_key=api_key or "mock-key",
                    model=settings.OPENAI_MODEL
                )
        return cls._provider


    @classmethod
    async def generate_answer(
        cls, 
        prompt: str, 
        system_prompt: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        provider = cls.get_provider()
        return await provider.generate_answer(prompt, system_prompt, history)

    @classmethod
    async def generate_structured_response(
        cls, 
        prompt: str, 
        system_prompt: str, 
        response_schema: Type[BaseModel]
    ) -> BaseModel:
        provider = cls.get_provider()
        return await provider.generate_structured_response(prompt, system_prompt, response_schema)
