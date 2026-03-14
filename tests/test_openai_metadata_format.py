from pydantic import BaseModel

from langfuse.openai import OpenAiArgsExtractor


def test_openai_args_extractor_adds_text_format_to_metadata():
    extractor = OpenAiArgsExtractor(metadata={"source": "test"}, text_format={"a": 1})

    langfuse_args = extractor.get_langfuse_args()

    assert langfuse_args["metadata"] == {"source": "test", "text_format": {"a": 1}}


def test_openai_args_extractor_serializes_pydantic_text_format():
    class ResponseModel(BaseModel):
        answer: str

    extractor = OpenAiArgsExtractor(text_format=ResponseModel)
    langfuse_args = extractor.get_langfuse_args()

    assert langfuse_args["metadata"]["text_format"] == ResponseModel.model_json_schema()


def test_openai_args_extractor_preserves_pydantic_metadata_when_adding_text_format():
    class MetadataModel(BaseModel):
        source: str

    extractor = OpenAiArgsExtractor(
        metadata=MetadataModel(source="test"),
        text_format={"type": "json_schema"},
    )

    langfuse_args = extractor.get_langfuse_args()

    assert langfuse_args["metadata"] == {
        "source": "test",
        "text_format": {"type": "json_schema"},
    }


def test_openai_args_extractor_store_removes_structured_output_metadata():
    extractor = OpenAiArgsExtractor(
        store=True,
        metadata={"source": "test"},
        response_format={"type": "json_schema"},
        text_format={"type": "json_schema"},
    )

    openai_args = extractor.get_openai_args()

    assert openai_args["metadata"] == {"source": "test"}


def test_openai_args_extractor_store_preserves_user_text_format_metadata():
    extractor = OpenAiArgsExtractor(
        store=True,
        metadata={"source": "test", "text_format": "user-value"},
    )

    openai_args = extractor.get_openai_args()

    assert openai_args["metadata"] == {
        "source": "test",
        "text_format": "user-value",
    }
