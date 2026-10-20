// Pure rendering and action tests use the shipped functions, without a server.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');

function runtime(){
  const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
  const startup=source.indexOf("document.addEventListener('submit'");
  assert.ok(startup>0);
  const context=vm.createContext({localStorage:{getItem:()=>null},document:{addEventListener:()=>{}}});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../web/person-workspace.js'),'utf8'),context);
  vm.runInContext(source.slice(0,startup),context);
  return context;
}
const query={intent:'query',status:'classified',allow_writes:false,record_spans:[]};
function review(intent=query){
  return {turn_intent:intent,answer_review:{status:'unavailable',method:'server_record_revision'},
    conversation_scope:{conversation_id:'cid-a',user_message_id:'user-a',blocked:false}};
}
function render(context,fn,...args){return vm.runInContext(`${fn}(...${JSON.stringify(args)})`,context);}

test('only a query shows a read-only generation button anchored to its own answer',()=>{
  const context=runtime(),html=render(context,'chatReviewStatus',review(),'answer-a','cid-a');
  assert.match(html,/重新生成回答（仅查询）/);
  assert.match(html,/data-action="chat-regenerate" data-cid="cid-a" data-mid="answer-a"/);
  assert.match(html,/未展示未经核对的草稿/);
  for(const intent of [undefined,{...query,intent:'unknown'},{...query,intent:'mixed',allow_writes:true},
    {...query,status:'failed'},{...query,record_spans:['record']}]){
    const data=review();data.turn_intent=intent;
    assert.doesNotMatch(render(context,'chatReviewStatus',data,'answer-a','cid-a'),/data-action="chat-regenerate"/);
  }
  assert.doesNotMatch(render(context,'chatReviewStatus',review(),'answer-a','cid-b'),/data-action="chat-regenerate"/);
});

test('historical failed reviews retain their original source-check wording',()=>{
  const context=runtime(),data=review();data.answer_review.method='single_model_revision';
  assert.match(render(context,'chatReviewStatus',data,'answer-a','cid-a'),/请对照引用来源核实当前回答/);
  data.answer_review.status='completed';
  assert.equal(render(context,'chatReviewStatus',data,'answer-a','cid-a'),'');
});

test('review status owns its warning while other source warnings remain visible',()=>{
  const context=runtime(),data=review();
  data.source_warnings=[{code:'review_unavailable',message:'duplicate review warning'},
    {code:'changed',message:'source changed'}];
  const html=render(context,'chatSourceWarnings',data);
  assert.doesNotMatch(html,/duplicate review warning/);
  assert.match(html,/source changed/);
  delete data.answer_review;
  assert.match(render(context,'chatSourceWarnings',data),/duplicate review warning/);
});

test('explicit reanalysis warns that it can save and ordinary retries reuse the decision',()=>{
  const context=runtime();
  const html=render(context,'chatIntentStatus',{turn_intent:{...query,intent:'unknown'}},'user-a','cid-a');
  assert.match(html,/重新分析原消息（可能保存）/);
  assert.match(html,/如包含明确记事，将保存或执行操作/);
  assert.match(html,/普通重试会沿用本次记事判断/);
});

test('regeneration labels show the original question and selected people with escaped text',()=>{
  const context=runtime(),data={regeneration:{source_assistant_message_id:'answer-a',source_user_message_id:'user-a',
    question:'他喜欢 <script>alert(1)</script> 吗？'},conversation_scope:{user_message_id:'user-a',
      person_ids:['person-a'],candidates:[{person_id:'person-a',display_name:'顾念'},{person_id:'person-b',display_name:'许栩'}]}};
  const html=render(context,'chatRegenerationLabel',data);
  assert.match(html,/重新回答原问题 · 顾念/);
  assert.doesNotMatch(html,/<script>|许栩/);
  assert.match(html,/&lt;script&gt;/);
  data.regeneration.source_user_message_id='other';
  assert.equal(render(context,'chatRegenerationLabel',data),'');
});

test('stale conversation buttons cannot dispatch and active ones use their exact answer ID',async()=>{
  const context=runtime(),calls=[];
  Object.assign(context,{document:{querySelector:()=>null},CSS:{escape:text=>text},
    recordCall:value=>calls.push(value)});
  vm.runInContext("state.page='chat';state.cid='cid-a'; currentStream=()=>null; loadMessages=async()=>{}; watch=(task,cid)=>recordCall({watched:task.id,cid}); api=async(path)=>{recordCall({path});return{id:'task-a',status:'queued'}};",context);
  await render(context,'regenerateChat',{cid:'cid-b',mid:'answer-b'});
  assert.equal(calls.length,0);
  await render(context,'regenerateChat',{cid:'cid-a',mid:'answer-a'});
  assert.equal(calls[0].path,'/conversations/cid-a/messages/answer-a/regenerate');
  assert.equal(calls[1].watched,'task-a');
  assert.equal(calls[1].cid,'cid-a');
});
