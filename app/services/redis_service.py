import redis
import json
from typing import List, Dict

redis_client = redis.Redis(host="localhost", port=6379,
                           db=0, decode_responses=True)


def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    history = redis_client.get(f"chat_session:{session_id}")
    if history:
        return json.loads(history)
    return []


def add_message_to_history(session_id: str, role: str, content: str):
    history = get_chat_history(session_id)
    history.append({"role": role, "content": content})

    if len(history) > 10:
        history = history[-10:]

    redis_client.set(f"chat_session:{session_id}",
                     json.dumps(history), ex=1800)
