import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from backend.app.services.llm import LLMService, OpenAIProvider

class DummySchema(BaseModel):
    summary: str
    rating: int

@pytest.mark.asyncio
async def test_openai_generate_answer():
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content="Mocked LLM Response"))
    ]
    
    with patch("backend.app.services.llm.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_client
        
        # Reset provider to force re-initialization
        LLMService._provider = None
        
        answer = await LLMService.generate_answer(
            prompt="Is aspirin safe for HF?",
            system_prompt="Answer from context only."
        )
        
        assert answer == "Mocked LLM Response"
        mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_openai_generate_structured_response():
    mock_choice = MagicMock()
    mock_choice.message.parsed = DummySchema(summary="Aspirin usage has risks.", rating=2)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    
    with patch("backend.app.services.llm.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = mock_completion
        mock_openai_class.return_value = mock_client
        
        LLMService._provider = None
        
        response = await LLMService.generate_structured_response(
            prompt="Summarize aspirin trials",
            system_prompt="Format summary.",
            response_schema=DummySchema
        )
        
        assert isinstance(response, DummySchema)
        assert response.summary == "Aspirin usage has risks."
        assert response.rating == 2
        mock_client.beta.chat.completions.parse.assert_called_once()


@pytest.mark.asyncio
async def test_mock_llm_provider_generate_answer():
    from backend.app.services.llm import MockLLMProvider
    provider = MockLLMProvider()
    
    # Text with Source Index
    system_prompt = "Source [1]:\n- Title: Hypertension Guideline 2026\nContent:\nAlways give ACEI first line."
    
    answer = await provider.generate_answer(
        prompt="First line HTN",
        system_prompt=system_prompt
    )
    
    assert "Always give ACEI first line" in answer
    assert "[1]" in answer


@pytest.mark.asyncio
async def test_mock_llm_provider_generate_structured_response():
    from backend.app.services.llm import MockLLMProvider
    provider = MockLLMProvider()
    
    response = await provider.generate_structured_response(
        prompt="Summarize AFib",
        system_prompt="Format summary.",
        response_schema=DummySchema
    )
    
    assert isinstance(response, DummySchema)
    assert "Mock structured string" in response.summary
    assert response.rating == 1

