# Copyright (c) Microsoft. All rights reserved.

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from agent_framework import (
    AgentResponse,
    AgentResponseUpdate,
    AgentSession,
    Content,
    ContextProvider,
    InMemoryHistoryProvider,
    Message,
    SecretString,
    SessionContext,
)
from agent_framework.exceptions import AgentException
from microsoft_agents.copilotstudio.client import ConnectionSettings, CopilotClient

from agent_framework_copilotstudio import CopilotStudioAgent
from agent_framework_copilotstudio._agent import DEFAULT_READ_BUFSIZE
from agent_framework_copilotstudio._feature_usage import FeatureIndex


def create_async_generator(items: list[Any]) -> Any:
    """Helper to create async generator mock."""

    async def async_gen() -> Any:
        for item in items:
            yield item

    return async_gen()


class RecordingContextProvider(ContextProvider):
    """Context provider that contributes context and records the hooks it receives."""

    def __init__(self) -> None:
        super().__init__(source_id="recording")
        self.calls: list[str] = []
        self.response_text: str | None = None

    async def before_run(
        self, *, agent: Any, session: AgentSession, context: SessionContext, state: dict[str, Any]
    ) -> None:
        self.calls.append("before_run")
        context.extend_instructions(self.source_id, "Answer in French.")
        context.extend_messages(self.source_id, [Message(role="user", contents=["The user is in Paris."])])

    async def after_run(
        self, *, agent: Any, session: AgentSession, context: SessionContext, state: dict[str, Any]
    ) -> None:
        self.calls.append("after_run")
        self.response_text = context.response.text if context.response else None


class TestCopilotStudioAgent:
    """Test cases for CopilotStudioAgent."""

    @pytest.fixture
    def mock_activity(self) -> MagicMock:
        activity = MagicMock()
        activity.text = "Test response"
        activity.type = "message"
        activity.id = "test-id"
        activity.from_property.name = "Test Bot"
        return activity

    @pytest.fixture
    def mock_copilot_client(self) -> MagicMock:
        return MagicMock(spec=CopilotClient)

    @patch("agent_framework_copilotstudio._acquire_token.acquire_token")
    @patch("agent_framework_copilotstudio._agent.load_settings")
    def test_init_missing_environment_id(self, mock_load_settings: MagicMock, mock_acquire_token: MagicMock) -> None:
        mock_acquire_token.return_value = "fake-token"
        mock_load_settings.return_value = {
            "environmentid": None,
            "schemaname": "test-bot",
            "tenantid": "test-tenant",
            "agentappid": "test-client",
        }

        with pytest.raises(ValueError, match="environment ID is required"):
            CopilotStudioAgent()

    @patch("agent_framework_copilotstudio._acquire_token.acquire_token")
    @patch("agent_framework_copilotstudio._agent.load_settings")
    def test_init_missing_bot_id(self, mock_load_settings: MagicMock, mock_acquire_token: MagicMock) -> None:
        mock_acquire_token.return_value = "fake-token"
        mock_load_settings.return_value = {
            "environmentid": "test-env",
            "schemaname": None,
            "tenantid": "test-tenant",
            "agentappid": "test-client",
        }

        with pytest.raises(ValueError, match="agent identifier"):
            CopilotStudioAgent()

    @patch("agent_framework_copilotstudio._acquire_token.acquire_token")
    @patch("agent_framework_copilotstudio._agent.load_settings")
    def test_init_missing_tenant_id(self, mock_load_settings: MagicMock, mock_acquire_token: MagicMock) -> None:
        mock_acquire_token.return_value = "fake-token"
        mock_load_settings.return_value = {
            "environmentid": "test-env",
            "schemaname": "test-bot",
            "tenantid": None,
            "agentappid": "test-client",
        }

        with pytest.raises(ValueError, match="tenant ID is required"):
            CopilotStudioAgent()

    @patch("agent_framework_copilotstudio._acquire_token.acquire_token")
    @patch("agent_framework_copilotstudio._agent.load_settings")
    def test_init_missing_client_id(self, mock_load_settings: MagicMock, mock_acquire_token: MagicMock) -> None:
        mock_acquire_token.return_value = "fake-token"
        mock_load_settings.return_value = {
            "environmentid": "test-env",
            "schemaname": "test-bot",
            "tenantid": "test-tenant",
            "agentappid": None,
        }

        with pytest.raises(ValueError, match="client ID is required"):
            CopilotStudioAgent()

    def test_init_with_client(self, mock_copilot_client: MagicMock) -> None:
        agent = CopilotStudioAgent(client=mock_copilot_client)
        assert agent.client == mock_copilot_client
        assert agent.id is not None

    @pytest.mark.parametrize("token", ["fake-token", SecretString("fake-token")], ids=["str", "secret"])
    @patch("agent_framework_copilotstudio._agent.CopilotClient")
    def test_init_unwraps_token_for_client(self, client_type: MagicMock, token: str | SecretString) -> None:
        CopilotStudioAgent(
            environment_id="env-id",
            agent_identifier="agent-id",
            token=token,
        )

        client_type.assert_called_once()
        assert type(client_type.call_args.kwargs["token"]) is str
        assert client_type.call_args.kwargs["token"] == "fake-token"

    def test_init_applies_default_read_bufsize(self) -> None:
        """The internally built client raises aiohttp's per-line limit to avoid LineTooLong (issue #7257)."""
        agent = CopilotStudioAgent(
            environment_id="env-id",
            agent_identifier="agent-id",
            token="fake-token",
        )
        assert agent.client.settings.client_session_settings["read_bufsize"] == DEFAULT_READ_BUFSIZE

    def test_init_respects_custom_client_session_settings(self) -> None:
        """User-provided client_session_settings are honored, and other keys are preserved."""
        agent = CopilotStudioAgent(
            environment_id="env-id",
            agent_identifier="agent-id",
            token="fake-token",
            client_session_settings={"read_bufsize": 123, "trust_env": True},
        )
        session_settings = agent.client.settings.client_session_settings
        assert session_settings["read_bufsize"] == 123
        assert session_settings["trust_env"] is True

    def test_init_injects_default_read_bufsize_into_partial_settings(self) -> None:
        """When client_session_settings omits read_bufsize, the default is added without dropping other keys."""
        agent = CopilotStudioAgent(
            environment_id="env-id",
            agent_identifier="agent-id",
            token="fake-token",
            client_session_settings={"trust_env": True},
        )
        session_settings = agent.client.settings.client_session_settings
        assert session_settings["read_bufsize"] == DEFAULT_READ_BUFSIZE
        assert session_settings["trust_env"] is True

    def test_init_applies_default_read_bufsize_to_supplied_settings(self) -> None:
        """When settings (but no client) is supplied, the agent still injects the read_bufsize default."""
        settings = ConnectionSettings(environment_id="env-id", agent_identifier="agent-id")
        agent = CopilotStudioAgent(settings=settings, token="fake-token")
        assert agent.client.settings.client_session_settings["read_bufsize"] == DEFAULT_READ_BUFSIZE

    def test_init_supplied_settings_keep_explicit_read_bufsize(self) -> None:
        """An explicit read_bufsize on supplied settings is preserved and not overridden by the default."""
        settings = ConnectionSettings(
            environment_id="env-id",
            agent_identifier="agent-id",
            client_session_settings={"read_bufsize": 42},
        )
        agent = CopilotStudioAgent(settings=settings, token="fake-token")
        assert agent.client.settings.client_session_settings["read_bufsize"] == 42

    @patch("agent_framework_copilotstudio._acquire_token.acquire_token")
    def test_init_empty_environment_id(self, mock_acquire_token: MagicMock) -> None:
        mock_acquire_token.return_value = "fake-token"
        with patch("agent_framework_copilotstudio._agent.load_settings") as mock_load_settings:
            mock_load_settings.return_value = {
                "environmentid": "",
                "schemaname": "test-bot",
                "tenantid": "test-tenant",
                "agentappid": "test-client",
            }

            with pytest.raises(ValueError, match="environment ID is required"):
                CopilotStudioAgent()

    @patch("agent_framework_copilotstudio._acquire_token.acquire_token")
    def test_init_empty_schema_name(self, mock_acquire_token: MagicMock) -> None:
        mock_acquire_token.return_value = "fake-token"
        with patch("agent_framework_copilotstudio._agent.load_settings") as mock_load_settings:
            mock_load_settings.return_value = {
                "environmentid": "test-env",
                "schemaname": "",
                "tenantid": "test-tenant",
                "agentappid": "test-client",
            }

            with pytest.raises(ValueError, match="agent identifier"):
                CopilotStudioAgent()

    async def test_run_with_string_message(self, mock_copilot_client: MagicMock, mock_activity: MagicMock) -> None:
        """Test run method with string message."""
        agent = CopilotStudioAgent(client=mock_copilot_client)

        conversation_activity = MagicMock()
        conversation_activity.conversation.id = "test-conversation-id"

        mock_copilot_client.start_conversation.return_value = create_async_generator([conversation_activity])
        mock_copilot_client.ask_question.return_value = create_async_generator([mock_activity])

        with patch("agent_framework_copilotstudio._agent.mark_feature_used") as mark_feature_used:
            response = await agent.run("test message")

        mark_feature_used.assert_called_once_with(FeatureIndex.COPILOTSTUDIO)
        assert isinstance(response, AgentResponse)
        assert len(response.messages) == 1
        content = response.messages[0].contents[0]
        assert content.type == "text"
        assert content.text == "Test response"
        assert response.messages[0].role == "assistant"

    async def test_run_with_chat_message(self, mock_copilot_client: MagicMock, mock_activity: MagicMock) -> None:
        """Test run method with Message."""
        agent = CopilotStudioAgent(client=mock_copilot_client)

        conversation_activity = MagicMock()
        conversation_activity.conversation.id = "test-conversation-id"

        mock_copilot_client.start_conversation.return_value = create_async_generator([conversation_activity])
        mock_copilot_client.ask_question.return_value = create_async_generator([mock_activity])

        chat_message = Message(role="user", contents=[Content.from_text("test message")])
        response = await agent.run(chat_message)

        assert isinstance(response, AgentResponse)
        assert len(response.messages) == 1
        content = response.messages[0].contents[0]
        assert content.type == "text"
        assert content.text == "Test response"
        assert response.messages[0].role == "assistant"

    async def test_run_with_session(self, mock_copilot_client: MagicMock, mock_activity: MagicMock) -> None:
        """Test run method with existing session."""
        agent = CopilotStudioAgent(client=mock_copilot_client)
        session = AgentSession()

        conversation_activity = MagicMock()
        conversation_activity.conversation.id = "test-conversation-id"

        mock_copilot_client.start_conversation.return_value = create_async_generator([conversation_activity])
        mock_copilot_client.ask_question.return_value = create_async_generator([mock_activity])

        response = await agent.run("test message", session=session)

        assert isinstance(response, AgentResponse)
        assert len(response.messages) == 1
        assert session.service_session_id == "test-conversation-id"

    async def test_run_start_conversation_failure(self, mock_copilot_client: MagicMock) -> None:
        """Test run method when conversation start fails."""
        agent = CopilotStudioAgent(client=mock_copilot_client)

        mock_copilot_client.start_conversation.return_value = create_async_generator([])

        with pytest.raises(AgentException, match="Failed to start a new conversation"):
            await agent.run("test message")

    async def test_run_streaming_with_string_message(self, mock_copilot_client: MagicMock) -> None:
        """Test run(stream=True) method with string message."""
        agent = CopilotStudioAgent(client=mock_copilot_client)

        conversation_activity = MagicMock()
        conversation_activity.conversation.id = "test-conversation-id"

        typing_activity = MagicMock()
        typing_activity.text = "Streaming response"
        typing_activity.type = "typing"
        typing_activity.id = "test-typing-id"
        typing_activity.from_property.name = "Test Bot"

        mock_copilot_client.start_conversation.return_value = create_async_generator([conversation_activity])
        mock_copilot_client.ask_question.return_value = create_async_generator([typing_activity])

        response_count = 0
        async for response in agent.run("test message", stream=True):
            assert isinstance(response, AgentResponseUpdate)
            content = response.contents[0]
            assert content.type == "text"
            assert content.text == "Streaming response"
            assert content.merge_mode == "append"
            response_count += 1

        assert response_count == 1

    async def test_run_streaming_with_session(self, mock_copilot_client: MagicMock) -> None:
        """Test run(stream=True) method with existing session."""
        agent = CopilotStudioAgent(client=mock_copilot_client)
        session = AgentSession()

        conversation_activity = MagicMock()
        conversation_activity.conversation.id = "test-conversation-id"

        typing_activity = MagicMock()
        typing_activity.text = "Streaming response"
        typing_activity.type = "typing"
        typing_activity.id = "test-typing-id"
        typing_activity.from_property.name = "Test Bot"

        mock_copilot_client.start_conversation.return_value = create_async_generator([conversation_activity])
        mock_copilot_client.ask_question.return_value = create_async_generator([typing_activity])

        response_count = 0
        async for response in agent.run("test message", session=session, stream=True):
            assert isinstance(response, AgentResponseUpdate)
            content = response.contents[0]
            assert content.type == "text"
            assert content.text == "Streaming response"
            response_count += 1

        assert response_count == 1
        assert session.service_session_id == "test-conversation-id"

    async def test_run_reuses_existing_conversation(
        self, mock_copilot_client: MagicMock, mock_activity: MagicMock
    ) -> None:
        """Test run method reuses an existing conversation ID from the session."""
        agent = CopilotStudioAgent(client=mock_copilot_client)
        session = AgentSession()
        session.service_session_id = "existing-conversation-id"

        mock_copilot_client.ask_question.return_value = create_async_generator([mock_activity])

        response = await agent.run("test message", session=session)

        assert isinstance(response, AgentResponse)
        assert session.service_session_id == "existing-conversation-id"
        mock_copilot_client.start_conversation.assert_not_called()
        mock_copilot_client.ask_question.assert_called_once_with("test message", "existing-conversation-id")

    async def test_run_streaming_reuses_existing_conversation(self, mock_copilot_client: MagicMock) -> None:
        """Test run(stream=True) method reuses an existing conversation ID from the session."""
        agent = CopilotStudioAgent(client=mock_copilot_client)
        session = AgentSession()
        session.service_session_id = "existing-conversation-id"

        typing_activity = MagicMock()
        typing_activity.text = "Streaming response"
        typing_activity.type = "typing"
        typing_activity.id = "test-typing-id"
        typing_activity.from_property.name = "Test Bot"

        mock_copilot_client.ask_question.return_value = create_async_generator([typing_activity])

        response_count = 0
        async for response in agent.run("test message", session=session, stream=True):
            assert isinstance(response, AgentResponseUpdate)
            response_count += 1

        assert response_count == 1
        assert session.service_session_id == "existing-conversation-id"
        mock_copilot_client.start_conversation.assert_not_called()
        mock_copilot_client.ask_question.assert_called_once_with("test message", "existing-conversation-id")

    async def test_run_streaming_no_typing_activity(self, mock_copilot_client: MagicMock) -> None:
        """Test run(stream=True) method with non-typing activity."""
        agent = CopilotStudioAgent(client=mock_copilot_client)

        conversation_activity = MagicMock()
        conversation_activity.conversation.id = "test-conversation-id"

        message_activity = MagicMock()
        message_activity.text = "Message response"
        message_activity.type = "message"
        message_activity.id = "test-message-id"

        mock_copilot_client.start_conversation.return_value = create_async_generator([conversation_activity])
        mock_copilot_client.ask_question.return_value = create_async_generator([message_activity])

        response_count = 0
        async for _response in agent.run("test message", stream=True):
            response_count += 1

        assert response_count == 0

    async def test_run_multiple_activities(self, mock_copilot_client: MagicMock) -> None:
        """Test run method with multiple message activities."""
        agent = CopilotStudioAgent(client=mock_copilot_client)

        conversation_activity = MagicMock()
        conversation_activity.conversation.id = "test-conversation-id"

        activity1 = MagicMock()
        activity1.text = "First response"
        activity1.type = "message"
        activity1.id = "test-id-1"
        activity1.from_property.name = "Test Bot"

        activity2 = MagicMock()
        activity2.text = "Second response"
        activity2.type = "message"
        activity2.id = "test-id-2"
        activity2.from_property.name = "Test Bot"

        mock_copilot_client.start_conversation.return_value = create_async_generator([conversation_activity])
        mock_copilot_client.ask_question.return_value = create_async_generator([activity1, activity2])

        response = await agent.run("test message")

        assert isinstance(response, AgentResponse)
        assert len(response.messages) == 2

    async def test_run_list_of_messages(self, mock_copilot_client: MagicMock, mock_activity: MagicMock) -> None:
        """Test run method with list of messages."""
        agent = CopilotStudioAgent(client=mock_copilot_client)

        conversation_activity = MagicMock()
        conversation_activity.conversation.id = "test-conversation-id"

        mock_copilot_client.start_conversation.return_value = create_async_generator([conversation_activity])
        mock_copilot_client.ask_question.return_value = create_async_generator([mock_activity])

        messages = ["Hello", "How are you?"]
        response = await agent.run(messages)

        assert isinstance(response, AgentResponse)
        assert len(response.messages) == 1

    async def test_run_streaming_start_conversation_failure(self, mock_copilot_client: MagicMock) -> None:
        """Test run(stream=True) method when conversation start fails."""
        agent = CopilotStudioAgent(client=mock_copilot_client)

        mock_copilot_client.start_conversation.return_value = create_async_generator([])

        with pytest.raises(AgentException, match="Failed to start a new conversation"):
            async for _ in agent.run("test message", stream=True):
                pass

    async def test_run_invokes_context_providers(
        self, mock_copilot_client: MagicMock, mock_activity: MagicMock
    ) -> None:
        """Test run calls the context providers and sends their context ahead of the question."""
        provider = RecordingContextProvider()
        agent = CopilotStudioAgent(client=mock_copilot_client, context_providers=[provider])

        conversation_activity = MagicMock()
        conversation_activity.conversation.id = "test-conversation-id"

        mock_copilot_client.start_conversation.return_value = create_async_generator([conversation_activity])
        mock_copilot_client.ask_question.return_value = create_async_generator([mock_activity])

        response = await agent.run("What is the weather?")

        assert response.text == "Test response"
        assert provider.calls == ["before_run", "after_run"]
        assert provider.response_text == "Test response"
        mock_copilot_client.ask_question.assert_called_once_with(
            "Answer in French.\nThe user is in Paris.\nWhat is the weather?", "test-conversation-id"
        )

    async def test_run_streaming_invokes_context_providers(self, mock_copilot_client: MagicMock) -> None:
        """Test streaming run calls after_run once, with the whole response, when the stream ends."""
        provider = RecordingContextProvider()
        agent = CopilotStudioAgent(client=mock_copilot_client, context_providers=[provider])

        conversation_activity = MagicMock()
        conversation_activity.conversation.id = "test-conversation-id"

        typing_activities = []
        for text in ("Bon", "jour"):
            typing_activity = MagicMock()
            typing_activity.text = text
            typing_activity.type = "typing"
            typing_activity.id = "test-typing-id"
            typing_activity.from_property.name = "Test Bot"
            typing_activities.append(typing_activity)

        mock_copilot_client.start_conversation.return_value = create_async_generator([conversation_activity])
        mock_copilot_client.ask_question.return_value = create_async_generator(typing_activities)

        updates: list[AgentResponseUpdate] = []
        async for update in agent.run("What is the weather?", stream=True):
            assert provider.calls == ["before_run"]
            updates.append(update)

        assert [update.text for update in updates] == ["Bon", "jour"]
        assert provider.calls == ["before_run", "after_run"]
        assert provider.response_text == "Bonjour"
        mock_copilot_client.ask_question.assert_called_once_with(
            "Answer in French.\nThe user is in Paris.\nWhat is the weather?", "test-conversation-id"
        )

    async def test_run_history_provider_stores_messages_without_replaying_them(
        self, mock_copilot_client: MagicMock
    ) -> None:
        """Test a history provider stores each exchange while the question stays the new input only."""
        history = InMemoryHistoryProvider()
        agent = CopilotStudioAgent(client=mock_copilot_client, context_providers=[history])
        session = agent.create_session()

        conversation_activity = MagicMock()
        conversation_activity.conversation.id = "test-conversation-id"
        mock_copilot_client.start_conversation.return_value = create_async_generator([conversation_activity])

        for question, answer in (("First question", "First answer"), ("Second question", "Second answer")):
            activity = MagicMock()
            activity.text = answer
            activity.type = "message"
            activity.id = "test-id"
            activity.from_property.name = "Test Bot"
            mock_copilot_client.ask_question.return_value = create_async_generator([activity])

            await agent.run(question, session=session)

            assert mock_copilot_client.ask_question.call_args.args == (question, "test-conversation-id")

        stored = await history.get_messages(session.session_id, state=session.state[history.source_id])
        assert [(message.role, message.text) for message in stored] == [
            ("user", "First question"),
            ("assistant", "First answer"),
            ("user", "Second question"),
            ("assistant", "Second answer"),
        ]
