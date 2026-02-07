"""Chat session management with OpenAI API.

元ファイル: preference_kg/experiments2/chat_session.py
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from .prompts import ELICITATION_SYSTEM_PROMPT, GREETING_MESSAGE


load_dotenv()


class ChatSession:
    """Manages a single chat session with dialogue history."""

    def __init__(self, user_id: str, model: str = "gpt-4o"):
        """Initialize a new chat session.

        Args:
            user_id: Unique identifier for the user
            model: OpenAI model to use for responses
        """
        self.user_id = user_id
        self.model = model
        self.created_at = datetime.now().isoformat()
        self.dialogue_history: list[dict[str, str]] = []

        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        # Add system message (not stored in dialogue_history for user visibility)
        self._system_message = {"role": "system", "content": ELICITATION_SYSTEM_PROMPT}

    def get_greeting(self) -> str:
        """Get the initial greeting message."""
        return GREETING_MESSAGE

    def add_user_message(self, content: str) -> None:
        """Add a user message to the dialogue history."""
        self.dialogue_history.append({"role": "user", "content": content})

    def generate_response(self) -> str:
        """Generate an assistant response using OpenAI API.

        Returns:
            The assistant's response text
        """
        # Build messages for API call
        messages = [self._system_message] + self.dialogue_history

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )
            assistant_content = response.choices[0].message.content

            # Add to dialogue history
            self.dialogue_history.append({"role": "assistant", "content": assistant_content})

            return assistant_content

        except Exception as e:
            error_msg = f"エラーが発生しました: {str(e)}"
            self.dialogue_history.append({"role": "assistant", "content": error_msg})
            return error_msg

    def get_session_data(self) -> dict:
        """Get the complete session data for saving.

        Returns:
            Dict with user_id, timestamp, and dialogue_history
        """
        return {
            "user_id": self.user_id,
            "timestamp": self.created_at,
            "dialogue_history": self.dialogue_history,
            "generation_model": self.model,
        }
