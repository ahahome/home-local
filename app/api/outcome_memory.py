"""Review communication-derived memory proposals before changing a person's record."""
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from app.agent import outcome_memory
from app.store.db import get_conn

router = APIRouter(prefix='/api')


class Input(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class AnalyzeIn(Input):
    pass


class ResolveIn(Input):
    action: Literal['confirm', 'discard', 'rebase']
    expected_revision: StrictInt = Field(ge=0)
    content: StrictStr | None = Field(default=None, min_length=1, max_length=2000)
    valid_from: StrictStr | None = None

    @model_validator(mode='after')
    def edits_only_on_confirm(self):
        if self.action != 'confirm' and self.model_fields_set & {'content', 'valid_from'}:
            raise ValueError('修改内容和生效日期只能随确认提交')
        if any(getattr(self, key) is None for key in self.model_fields_set & {'content', 'valid_from'}):
            raise ValueError('修改内容和生效日期不能为null')
        return self


def _call(fn, *args):
    try:
        return fn(get_conn(), *args)
    except outcome_memory.OutcomeMemoryError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


@router.post('/followup-outcomes/{oid}/memory-analysis', status_code=202)
def analyze(oid: str, body: AnalyzeIn):
    return _call(outcome_memory.create_analysis, oid)


@router.get('/followups/{fid}/memory-updates')
def listing(fid: str):
    return _call(outcome_memory.list_updates, fid)


@router.get('/memory-updates/{proposal_id}')
def detail(proposal_id: str):
    return _call(outcome_memory.read_proposal, proposal_id)


@router.post('/memory-updates/{proposal_id}/resolve')
def resolve(proposal_id: str, body: ResolveIn):
    return _call(outcome_memory.resolve_proposal, proposal_id,
                 body.model_dump(exclude_unset=True))
