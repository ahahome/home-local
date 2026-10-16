"""Communication preparation and explicit outcome confirmation endpoints."""
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from app.agent import followup
from app.store.db import get_conn

router = APIRouter(prefix='/api')


class Input(BaseModel):
    model_config = ConfigDict(extra='forbid')


class PrepareIn(Input):
    goal: str = Field(min_length=1, max_length=1000)
    request_id: str = Field(min_length=1, max_length=128)


class DraftIn(Input):
    draft: str = Field(min_length=1, max_length=12000)
    expected_revision: StrictInt = Field(ge=1)


class PromiseIn(Input):
    who: Literal['me', 'them']
    content: str = Field(min_length=1, max_length=2000)
    due_date: str | None = None


class OutcomeIn(Input):
    request_id: str = Field(min_length=1, max_length=128)
    expected_revision: StrictInt = Field(ge=1)
    contacted_at: StrictInt = Field(gt=0)
    channel: str = Field(min_length=1, max_length=80)
    notes: str = Field(min_length=1, max_length=10000)
    state: Literal['awaiting_reply', 'follow_up', 'completed']
    next_followup_at: StrictInt | None = None
    complete_promise_ids: list[str] = Field(default_factory=list, max_length=100)
    new_promises: list[PromiseIn] = Field(default_factory=list, max_length=50)


def _call(fn, *args):
    try:
        return fn(get_conn(), *args)
    except followup.FollowupError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.post('/persons/{person_id}/followups', status_code=202)
def create(person_id: str, body: PrepareIn):
    return _call(followup.create_followup, person_id, body.goal, body.request_id)


@router.get('/persons/{person_id}/followups')
def listing(person_id: str):
    return {'followups': _call(followup.list_followups, person_id)}


@router.get('/followups/{fid}')
def detail(fid: str):
    return _call(followup.read_followup, fid)


@router.patch('/followups/{fid}/draft')
def edit_draft(fid: str, body: DraftIn):
    return _call(followup.update_draft, fid, body.draft, body.expected_revision)


@router.post('/followups/{fid}/outcomes')
def outcome(fid: str, body: OutcomeIn):
    return _call(followup.record_outcome, fid, body.model_dump())
