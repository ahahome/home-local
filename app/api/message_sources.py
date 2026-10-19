"""Read only the source snapshot persisted with a specific assistant message."""
import json

from fastapi import APIRouter, HTTPException

from app.agent.memory_sources import source_state
from app.store.db import get_conn

router = APIRouter(prefix='/api')


@router.get('/conversations/{cid}/messages/{mid}/sources/{sid}')
def message_source(cid: str, mid: str, sid: str):
    conn = get_conn()
    message = conn.execute("SELECT tool_calls FROM message WHERE id=? AND conversation_id=? AND role='assistant'",
                           (mid, cid)).fetchone()
    if message:
        try:
            metadata = json.loads(message['tool_calls'])
        except (json.JSONDecodeError, TypeError):
            metadata = None
        sources = metadata.get('sources') if isinstance(metadata, dict) else None
        if isinstance(sources, list):
            for source in sources:
                if isinstance(source, dict) and source.get('id') == sid and \
                        source.get('type') in {'memory', 'field', 'interaction'} and \
                        isinstance(source.get('entity_id'), str):
                    return {'source': source, 'state': source_state(conn, source)}
    raise HTTPException(404, '该消息的引用来源不存在')
