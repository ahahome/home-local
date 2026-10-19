'use strict';

const personWorkspaceTabs={overview:'概览',communication:'沟通',memory:'记忆',history:'历史'};

function personWorkspaceRouteId(){
  const match=/^#\/persons\/([^/?]+)(?:[/?]|$)/.exec(location.hash);
  if(!match)return null;
  try{return decodeURIComponent(match[1]);}catch{return null;}
}
function personWorkspaceCurrent(view=state.personWorkspaceView){
  return !!view&&state.page==='persons'&&state.personWorkspaceView===view&&personWorkspaceRouteId()===view.id;
}
function personWorkspaceSelectedTab(){
  const tab=new URLSearchParams(location.hash.split('?')[1]||'').get('tab');
  return Object.hasOwn(personWorkspaceTabs,tab)?tab:'overview';
}
async function openPersonWorkspace(id){
  if(personWorkspaceRouteId()===id&&state.page==='persons'){
    close();
    return personWorkspace(id);
  }
  return go('persons/'+encodeURIComponent(id));
}
async function personWorkspace(id){
  const request={};state.personWorkspaceRequest=request;
  const previous=state.personWorkspaceView;
  let data,history,followups;
  try{
    [data,history,followups]=await Promise.all([
      api(`/persons/${encodeURIComponent(id)}`),
      api(`/persons/${encodeURIComponent(id)}/history`).catch(error=>({history:[],corrections:[],error:error.message})),
      api(`/persons/${encodeURIComponent(id)}/followups`).catch(error=>({followups:[],error:error.message}))
    ]);
  }catch(error){
    if(state.personWorkspaceRequest!==request||state.page!=='persons'||personWorkspaceRouteId()!==id)return;
    throw error;
  }
  if(state.personWorkspaceRequest!==request||state.page!=='persons'||personWorkspaceRouteId()!==id)return;
  const view={id,data,history,followups,tab:personWorkspaceSelectedTab(),historyAt:'',memoryHtml:''};
  if(previous?.id===id&&previous.historyAt)view.historyAt=previous.historyAt;
  state.personWorkspaceView=view;state.detail=data;state.fieldHistory=history.history||[];
  const breadcrumb=$('#breadcrumb');if(breadcrumb)breadcrumb.textContent='人物档案 / '+data.person.display_name;
  paintPersonWorkspace(view);
  if(view.tab==='memory')await loadPersonWorkspaceMemories(view);
  if(view.tab==='history'&&view.historyAt)await loadPersonWorkspaceHistory({id,value:view.historyAt});
}
async function refreshPersonWorkspaceAfterSave(view){
  if(!personWorkspaceCurrent(view))return null;
  try{await personWorkspace(view.id);return personWorkspaceCurrent()&&state.personWorkspaceView.id===view.id?state.personWorkspaceView:null;}
  catch(error){if(personWorkspaceCurrent(view))toast('已保存，档案暂时无法刷新：'+error.message);}
  return null;
}
function personWorkspaceAttrs(id){return `data-id="${esc(id)}"`;}
function personWorkspaceEmpty(text){return `<p class="person-workspace-empty">${esc(text)}</p>`;}
function personWorkspaceHeading(title,description='',action=''){
  return `<div class="person-workspace-section-head"><div><h2>${esc(title)}</h2>${description?`<p>${esc(description)}</p>`:''}</div>${action}</div>`;
}
function personWorkspacePromises(view){
  return (view.data.open_promises||[]).map(p=>`<article class="person-workspace-promise"><div><span class="person-workspace-kicker">${p.who==='me'?'我答应的':'对方答应的'}</span><h3>${esc(p.content)}</h3><p>${p.due?'约定日期 · '+esc(p.due):'尚未约定日期'}</p></div>${btn('标记完成','complete-promise',personWorkspaceAttrs(p.id),'small-button')}</article>`).join('')||personWorkspaceEmpty('没有未完成的承诺。');
}
function personWorkspaceFields(view){
  const fields=Object.entries(view.data.fields||{});
  return fields.length?`<div class="person-workspace-fields">${fields.map(([key,value])=>`<article class="person-workspace-field"><div><h3>${esc(fieldLabel(key))}</h3><p>${esc(value.value)}</p>${value.since?`<small>生效于 ${esc(value.since)}</small>`:''}</div><div class="person-workspace-field-actions">${btn('修改','edit-field',`${personWorkspaceAttrs(view.id)} data-field="${esc(key)}"`,'ghost small-button')}${btn('纠错','correct-field',`${personWorkspaceAttrs(view.id)} data-field="${esc(key)}"`,'ghost small-button')}</div></article>`).join('')}</div>`:personWorkspaceEmpty('还没有档案信息，可以先补充公司、城市或联系方式。');
}
function personWorkspaceOverview(view){
  return `<section class="person-workspace-panel">${personWorkspaceHeading('当前档案','已确认的信息会在这里持续积累。',btn('补充信息','edit-field',personWorkspaceAttrs(view.id),'small-button'))}${personWorkspaceFields(view)}</section><section class="person-workspace-panel">${personWorkspaceHeading('未完成的承诺',`${(view.data.open_promises||[]).length} 项待完成`)}${personWorkspacePromises(view)}</section>`;
}
function personWorkspaceCommunication(view){
  const d=view.data,f=view.followups;
  const records=(d.interactions||[]).map(item=>{
    const contact=item.kind==='contact'&&item.person_role==='participant';
    return `<article class="person-workspace-interaction"><div class="person-workspace-record-meta">${tag(contact?'实际沟通':'资料记录',contact?'good':'')}<time>${esc(item.ts)}</time>${contact&&item.channel?`<span>${esc(followupChannel(item.channel))}</span>`:''}</div><p>${esc(item.content)}</p></article>`;
  }).join('');
  return `<section class="person-workspace-panel">${personWorkspaceHeading('近期跟进','从准备到记录沟通结果，接续每一次联系。',btn('准备沟通','prepare-person',`${personWorkspaceAttrs(view.id)} ${d.person.deleted_at?'disabled':''}`,'primary small-button'))}${f.error?`<p class="person-workspace-error" role="alert">跟进记录暂时无法读取：${esc(f.error)}</p>${btn('重新读取','person-workspace-reload',personWorkspaceAttrs(view.id),'small-button')}`:followupRows(f.followups)}</section><section class="person-workspace-panel">${personWorkspaceHeading('沟通与资料记录','最近 10 条记录；实际沟通与资料整理分别标记。')}${records||personWorkspaceEmpty('还没有沟通或资料记录。')}</section>`;
}
function personWorkspaceMemory(view){
  return `<section class="person-workspace-panel person-workspace-memory-panel">${personWorkspaceHeading('记忆与偏好','查看已保存的背景、经历与偏好，核对来源并进行纠错。')}<div id="person-memories" data-person-workspace-id="${esc(view.id)}" aria-live="polite">${view.memoryHtml||'<p class="memory-page-status" role="status">正在读取记忆与偏好…</p>'}</div></section><aside class="person-workspace-memory-note"><div><span class="person-workspace-kicker">待审阅提案</span><p>沟通结果中建议保存的记忆，经你确认后进入档案。</p></div><a href="${esc(memoriesHref({person:view.id,tab:'proposals',status:'pending'}))}">审阅这个人的提案 <span aria-hidden="true">→</span></a></aside>`;
}
function personWorkspaceHistory(view){
  const history=view.history;
  return `<section class="person-workspace-panel">${personWorkspaceHeading('字段历史与来源','按日期回看当时有效的信息；历史版本与原始来源持续保留。')}<div class="person-workspace-history-filter"><label for="person-workspace-history-date">查看日期</label><input id="person-workspace-history-date" type="date" value="${esc(view.historyAt)}" data-id="${esc(view.id)}"><span>留空查看全部版本</span></div><div id="field-history" aria-live="polite">${history.error?`<p class="person-workspace-error" role="alert">历史暂时无法读取：${esc(history.error)}</p>${btn('重新读取历史','person-history-retry',personWorkspaceAttrs(view.id),'small-button')}`:historyRows(history.history||[])}</div></section><section class="person-workspace-panel">${personWorkspaceHeading('手动纠错记录','修改、删除、恢复与转移归属的处理记录。')}<div id="person-workspace-corrections">${history.error?personWorkspaceEmpty('历史读取成功后显示纠错记录。'):correctionRows(history.corrections||[])}</div></section>`;
}
function personWorkspaceContent(view){
  return ({overview:personWorkspaceOverview,communication:personWorkspaceCommunication,memory:personWorkspaceMemory,history:personWorkspaceHistory})[view.tab](view);
}
function personWorkspaceSidebar(view){
  const d=view.data,p=d.person,attrs=personWorkspaceAttrs(view.id),totals=d.ledger?.totals||{};
  const aliases=(d.aliases||[]).filter(alias=>alias!==p.display_name);
  const context=['company','title','city'].filter(key=>d.fields?.[key]).map(key=>`<div><dt>${esc(fieldLabel(key))}</dt><dd>${esc(d.fields[key].value)}</dd></div>`).join('');
  return `<aside class="person-workspace-sidebar"><section class="person-workspace-identity"><div class="person-workspace-profile">${avatar(p.display_name)}<div><span class="person-workspace-kicker">人物档案</span><h1>${esc(p.display_name)}</h1></div></div>${p.deleted_at?`<p class="person-workspace-archive">${tag('已归档','warning')} 已从日常检索中收起，可在更多操作中恢复。</p>`:''}${aliases.length?`<p class="person-workspace-aliases">${aliases.map(alias=>esc(alias)).join(' · ')}</p>`:''}${p.value_read?`<p class="person-workspace-summary">${esc(p.value_read)}</p>`:''}${context?`<dl class="person-workspace-context">${context}</dl>`:''}<div class="person-workspace-last-contact"><span>上次实际联系</span><strong>${d.last_contact?esc(fmt(d.last_contact,true,true)):'尚未记录'}</strong></div><div class="person-workspace-primary-actions">${btn('准备沟通','prepare-person',`${attrs} ${p.deleted_at?'disabled':''}`,'primary')}${btn('联系周期','contact-policy',`${attrs} ${p.deleted_at?'disabled':''}`)}</div></section><section class="person-workspace-ledger"><span class="person-workspace-kicker">资金往来</span><div><p>收到<strong>¥${esc((Number(totals.in_cents||0)/100).toFixed(2))}</strong></p><p>支出<strong>¥${esc((Number(totals.out_cents||0)/100).toFixed(2))}</strong></p></div><a href="#/ledger">查看 ${esc(totals.count||0)} 笔确认账目 <span aria-hidden="true">↗</span></a></section><details class="person-workspace-maintenance"><summary>更多操作</summary><p>整理档案与数据维护</p><div>${btn('研究这个人','research-person',attrs,'ghost')}${p.deleted_at?btn('恢复人物','restore-person',attrs):btn('归档人物','archive-person',attrs,'ghost')}${btn('合并重复档案','merge-person',attrs,'ghost')}${btn('彻底清除','purge-person',attrs,'ghost danger')}</div></details></aside>`;
}
function paintPersonWorkspace(view=state.personWorkspaceView){
  if(!personWorkspaceCurrent(view))return;
  render('persons',`<div class="person-workspace"><div class="person-workspace-breadcrumb"><a href="#/persons"><span aria-hidden="true">←</span> 全部人物</a><span> / </span><span>${esc(view.data.person.display_name)}</span></div><div class="person-workspace-layout">${personWorkspaceSidebar(view)}<div class="person-workspace-main"><div class="person-workspace-tabs" role="tablist" aria-label="人物档案分区">${Object.entries(personWorkspaceTabs).map(([key,label])=>`<button id="person-workspace-tab-${key}" role="tab" aria-selected="${view.tab===key}" aria-controls="person-workspace-content" tabindex="${view.tab===key?'0':'-1'}" class="${view.tab===key?'active':''}" data-action="person-workspace-tab" data-tab="${key}" data-id="${esc(view.id)}">${label}${key==='overview'&&view.data.open_promises?.length?`<span>${esc(view.data.open_promises.length)}</span>`:''}</button>`).join('')}</div><div id="person-workspace-content" role="tabpanel" aria-labelledby="person-workspace-tab-${view.tab}" tabindex="0">${personWorkspaceContent(view)}</div></div></div></div>`);
}
async function personWorkspaceTab(data){
  const view=state.personWorkspaceView;
  if(!personWorkspaceCurrent(view)||view.id!==data.id||!Object.hasOwn(personWorkspaceTabs,data.tab))return;
  view.tab=data.tab;
  history.replaceState(null,'',`#/persons/${encodeURIComponent(view.id)}${view.tab==='overview'?'':'?tab='+view.tab}`);
  paintPersonWorkspace(view);
  $(`#person-workspace-tab-${view.tab}`)?.focus();
  if(view.tab==='memory')await loadPersonWorkspaceMemories(view);
  if(view.tab==='history'&&view.historyAt)await loadPersonWorkspaceHistory({id:view.id,value:view.historyAt});
}
async function loadPersonWorkspaceMemories(view=state.personWorkspaceView){
  if(!personWorkspaceCurrent(view)||view.tab!=='memory')return;
  const host=$('#person-memories');if(!host||host.dataset.personWorkspaceId!==view.id)return;
  const request={};view.memoryRequest=request;
  if(!view.memoryHtml)host.innerHTML='<p class="memory-page-status" role="status">正在读取记忆与偏好…</p>';
  const query=new URLSearchParams({person_id:view.id,status:'active',limit:'6',offset:'0'}),pendingQuery=new URLSearchParams({person_id:view.id,status:'pending',limit:'1',offset:'0'});
  const [saved,proposals]=await Promise.allSettled([api(`/memories?${query}`),api(`/memory-updates?${pendingQuery}`)]);
  if(!personWorkspaceCurrent(view)||view.memoryRequest!==request||view.tab!=='memory'||$('#person-memories')!==host)return;
  const rows=saved.status==='fulfilled'?(saved.value.memories||[]):[];
  view.memoryHtml=`${saved.status==='fulfilled'?rows.map(memory=>`<article class="memory-list-row"><div class="memory-row-content"><div class="memory-row-meta">${tag(memoryKinds[memory.kind]||memory.kind)}<span>${esc(memoryDate(memory.ts))}</span></div><p class="memory-list-text">${esc(memoryPreview(memory.text,180))}</p><p class="muted memory-row-meta">${memory.source?'来源 · '+esc(memorySourceName(memory.source)):'来源未记录'}</p></div>${btn('查看详情','memory-detail',`data-id="${esc(memory.id)}"`,'small-button')}</article>`).join('')||personWorkspaceEmpty('还没有保存这个人的记忆与偏好。'): `<p class="person-workspace-error" role="alert">记忆暂时无法读取：${esc(saved.reason.message)}</p>`}<div class="memory-person-links"><a href="${esc(memoriesHref({person:view.id}))}">全部记忆${saved.status==='fulfilled'?'（'+esc(saved.value.total)+'）':''} →</a>${proposals.status==='fulfilled'?`<a href="${esc(memoriesHref({person:view.id,tab:'proposals',status:'pending'}))}">待审阅提案（${esc(proposals.value.total)}） →</a>`:'<span class="memory-list-error">待审阅提案计数暂时无法读取</span>'}${saved.status==='rejected'||proposals.status==='rejected'?btn('重新读取','person-memories-retry','','ghost small-button'):''}</div>`;
  host.innerHTML=view.memoryHtml;
}
async function loadPersonWorkspaceHistory(input){
  const view=state.personWorkspaceView;
  if(!personWorkspaceCurrent(view)||view.tab!=='history')return;
  const id=input?.dataset?.id||input?.id||view.id;
  if(id!==view.id)return;
  const value=input?.value??view.historyAt,host=$('#field-history');if(!host)return;
  const request={};view.historyRequest=request;view.historyAt=value;
  host.innerHTML='<p class="memory-page-status" role="status">正在读取字段历史…</p>';
  try{
    const data=await api(`/persons/${encodeURIComponent(view.id)}/history${value?'?at='+encodeURIComponent(value):''}`);
    if(!personWorkspaceCurrent(view)||view.historyRequest!==request||view.tab!=='history'||$('#field-history')!==host)return;
    view.history={...view.history,...data,error:''};state.fieldHistory=data.history||[];
    host.innerHTML=historyRows(data.history||[]);
    const corrections=$('#person-workspace-corrections');if(corrections)corrections.innerHTML=correctionRows(view.history.corrections||[]);
  }catch(error){
    if(!personWorkspaceCurrent(view)||view.historyRequest!==request||view.tab!=='history'||$('#field-history')!==host)return;
    host.innerHTML=`<p class="person-workspace-error" role="alert">历史暂时无法读取：${esc(error.message)}</p>${btn('重新读取历史','person-history-retry',personWorkspaceAttrs(view.id),'small-button')}`;
  }
}

document.addEventListener('change',event=>{
  if(event.target.id==='person-workspace-history-date')loadPersonWorkspaceHistory(event.target).catch(error=>toast(error.message));
});
document.addEventListener('keydown',event=>{
  if(!event.target.closest('.person-workspace-tabs')||!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
  event.preventDefault();
  const keys=Object.keys(personWorkspaceTabs),current=keys.indexOf(event.target.dataset.tab);
  const index=event.key==='Home'?0:event.key==='End'?keys.length-1:(current+(event.key==='ArrowLeft'?-1:1)+keys.length)%keys.length;
  personWorkspaceTab({id:event.target.dataset.id,tab:keys[index]}).catch(error=>toast(error.message));
});
