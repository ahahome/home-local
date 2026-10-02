UPDATE task SET status='interrupted', error='重复的对话任务已中断，可在任务中心查看'
WHERE kind='chat' AND status IN ('queued','running') AND rowid NOT IN (
 SELECT MAX(rowid) FROM task WHERE kind='chat' AND status IN ('queued','running')
 GROUP BY json_extract(payload,'$.conversation_id')
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_chat
ON task(json_extract(payload,'$.conversation_id'))
WHERE kind='chat' AND status IN ('queued','running');
