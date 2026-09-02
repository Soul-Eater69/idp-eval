from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

BASE = Path("l3_experiments/19_theme_needs_description_stage_batch.ipynb")
E21 = Path("l3_experiments/21_theme_needs_description_stage_batch_recall_tuned.ipynb")
E22 = Path("l3_experiments/22_theme_needs_description_stage_batch_candidate_relevance.ipynb")


def source(cell):
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


def set_source(cell, text):
    cell["source"] = text
    cell["execution_count"] = None if cell.get("cell_type") == "code" else cell.get("execution_count")
    if cell.get("cell_type") == "code":
        cell["outputs"] = []


def find_code_cell(nb, marker):
    matches = [cell for cell in nb["cells"] if cell.get("cell_type") == "code" and marker in source(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one code cell containing {marker!r}, found {len(matches)}")
    return matches[0]


def replace_system_prompt(nb, prompt_text):
    cell = find_code_cell(nb, "SYSTEM_PROMPT =")
    current = source(cell)
    suffix_marker = "\n\ndef _prompt_text(value):"
    pos = current.find(suffix_marker)
    if pos < 0:
        raise RuntimeError("Could not find prompt helper suffix")
    suffix = current[pos:]
    new_source = 'SYSTEM_PROMPT = """\\\n' + prompt_text.strip() + '\n"""' + suffix
    set_source(cell, new_source)


def base_variant(number, experiment_name, title):
    nb = deepcopy(json.loads(BASE.read_text(encoding="utf-8")))

    # Title and explanation cells.
    nb["cells"][0]["source"] = title + "\n\nNo record key or artificial item ID is sent to the model."
    nb["cells"][1]["source"] = (
        "## What the LLM sees\n\n"
        "Theme Business Needs + Theme Description + unique Stage data + Stage candidate L3s.\n\n"
        "**Not sent:** record key, ground truth, L1/L2 hierarchy."
    )

    config = find_code_cell(nb, "EXPERIMENT_NAME=")
    text = source(config)
    text = text.replace("SAMPLE_SIZE=50", "SAMPLE_SIZE=100")
    text = text.replace(
        "EXPERIMENT_NAME='E19_THEME_NEEDS_DESCRIPTION_STAGE_BATCH_BY_STAGE_ID'",
        f"EXPERIMENT_NAME='{experiment_name}'",
    )
    set_source(config, text)

    # Make the notebook display all sampled records when inspecting population/results.
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            text = source(cell).replace("evaluation_population.head(50)", "evaluation_population.head(100)")
            text = text.replace("display(results.head(50))", "display(results.head(100))")
            text = text.replace(
                "fixed_50_from_golden_valid_set_seed_42_theme_batch",
                "fixed_100_from_golden_valid_set_seed_42_theme_batch",
            )
            set_source(cell, text)

    return nb


E21_PROMPT = r'''You are performing Level 3 business capability classification for multiple Value Stream Stages that share the same Theme context.

An L3 capability is a Level 3 business capability: a specific business function within the enterprise capability hierarchy.

Use Theme Business Needs as the primary shared business evidence.
Use Theme Description as supporting context that can clarify the Business Needs, but do not use it to introduce unsupported business functions.
Classify each supplied Value Stream Stage independently using only the shared Theme context, that Stage's governed metadata, and that Stage's candidate L3 capabilities.

EVIDENCE

Theme Business Needs describes the shared business outcomes, requirements, and functions that need to be delivered.
Theme Description provides supporting scope and intent for those Business Needs.
Each Stage's name, description, entrance criteria, and exit criteria define that Stage's business-process boundary.

For each candidate L3:
- capability_id is the exact identifier to return when selected.
- capability_description is the primary semantic definition of the business function.
- capability_name is a supporting business label.
- capability_tier is supporting taxonomy context only.

Do not infer business meaning from capability_id.

CLASSIFICATION

For each Stage independently:

1. Identify the business functions expressed by the Theme Business Needs.
2. Use Theme Description to clarify scope and intent.
3. Constrain those functions to this Stage using the Stage name, description, entrance criteria, and exit criteria.
4. Evaluate EVERY supplied candidate L3 independently.

Select a candidate when its business function is:
- explicitly stated in the Theme context, OR
- strongly semantically implied by the Theme context within this Stage boundary.

The exact capability name or terminology does not need to appear in the Theme text when the underlying business function is clearly supported.

Selecting one capability does not exclude another capability.
If multiple distinct business functions are supported, return ALL supported candidates.

Before producing the final answer, perform an omission check:
reconsider every unselected candidate and add it if there is concrete business evidence supporting its function.

Do NOT select a capability only because it:
- belongs to the supplied Stage,
- shares words with the Theme,
- is generally related,
- is a prerequisite,
- is upstream or downstream,
- commonly supports another capability.

Prefer complete coverage of genuinely supported business functions while excluding merely adjacent or related capabilities.

Do not use one Stage's metadata or candidates as evidence for another Stage.
Only return capability_id values supplied for that Stage.
If no candidate is supported for a Stage, return an empty list.
Return exactly one result for every supplied stage_id.

OUTPUT

Return JSON only:

{
  "stages": [
    {
      "stage_id": "VSS000123",
      "l3": ["CAP00000123", "CAP00000456"]
    },
    {
      "stage_id": "VSS000456",
      "l3": []
    }
  ]
}

Do not return reasons, explanations, Markdown, or additional fields.'''


E22_PROMPT = r'''You are performing Level 3 business capability relevance validation for multiple Value Stream Stages that share the same Theme context.

An L3 capability is a Level 3 business capability: a specific business function within the enterprise capability hierarchy.

Use Theme Business Needs as the primary shared business evidence.
Use Theme Description as supporting context that can clarify the Business Needs, but do not use it to introduce unsupported business functions.
Evaluate each supplied Value Stream Stage independently using only the shared Theme context, that Stage's governed metadata, and that Stage's candidate L3 capabilities.

EVIDENCE

Theme Business Needs describes the shared business outcomes, requirements, and functions that need to be delivered.
Theme Description provides supporting scope and intent for those Business Needs.
Each Stage's name, description, entrance criteria, and exit criteria define that Stage's business-process boundary.

For each candidate L3:
- capability_id is the exact identifier of the candidate being judged.
- capability_description is the primary semantic definition of the business function.
- capability_name is a supporting business label.
- capability_tier is supporting taxonomy context only.

Do not infer business meaning from capability_id.

RELEVANCE VALIDATION

For each Stage independently:

1. Identify the business functions expressed by the Theme Business Needs.
2. Use Theme Description to clarify scope and intent.
3. Constrain those functions to this Stage using the Stage name, description, entrance criteria, and exit criteria.
4. Treat each supplied candidate L3 as an independent relevance claim.

For EVERY supplied candidate, decide relevant=true or relevant=false.

Set relevant=true when the candidate's business function is:
- explicitly stated in the Theme context, OR
- strongly semantically implied by the Theme context within this Stage boundary.

The exact capability name or terminology does not need to appear in the Theme text when the underlying business function is clearly supported.

Set relevant=false when the capability is only:
- a member of the supplied Stage,
- a terminology overlap,
- generally related or adjacent,
- a prerequisite,
- upstream or downstream,
- a function that commonly supports another relevant capability without itself being supported by the Theme context.

Judge every candidate independently. A true decision for one candidate neither requires nor excludes a true decision for another candidate.
You MUST evaluate every candidate exactly once. Do not omit any supplied candidate and do not add candidates that were not supplied for that Stage.
Do not use one Stage's metadata or candidates as evidence for another Stage.
Return exactly one result for every supplied stage_id.

OUTPUT

Return JSON only:

{
  "stages": [
    {
      "stage_id": "VSS000123",
      "candidates": [
        {"capability_id": "CAP00000123", "relevant": true},
        {"capability_id": "CAP00000456", "relevant": false}
      ]
    },
    {
      "stage_id": "VSS000456",
      "candidates": []
    }
  ]
}

Do not return reasons, explanations, Markdown, or additional fields.'''


# E21: same E19 machinery, recall-tuned full system prompt, 100-row sample.
e21 = base_variant(
    21,
    "E21_THEME_NEEDS_DESCRIPTION_STAGE_BATCH_RECALL_TUNED",
    "# Experiment 21 — Theme Needs + Description + Stage — Batch Recall-Tuned Selection",
)
replace_system_prompt(e21, E21_PROMPT)
E21.write_text(json.dumps(e21, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# E22: same input/batching architecture, but force one relevance judgment per candidate.
e22 = base_variant(
    22,
    "E22_THEME_NEEDS_DESCRIPTION_STAGE_BATCH_CANDIDATE_RELEVANCE",
    "# Experiment 22 — Theme Needs + Description + Stage — Batch Candidate Relevance Validation",
)
replace_system_prompt(e22, E22_PROMPT)

prediction = find_code_cell(e22, "def validate(payload,expected,allowed):")
E22_PREDICTION = r'''def validate(payload,expected,allowed):
    if not isinstance(payload,dict) or set(payload)!={'stages'} or not isinstance(payload['stages'],list):
        raise ValueError('Expected stages list')

    expected=set(expected)
    out={}
    decisions=[]
    seen_stages=set()

    for result in payload['stages']:
        if not isinstance(result,dict) or set(result)!={'stage_id','candidates'}:
            raise ValueError('Each result needs stage_id and candidates')

        sid=clean(result['stage_id'])
        if sid not in expected or sid in seen_stages:
            raise ValueError(f'Invalid or duplicate stage_id: {sid}')
        if not isinstance(result['candidates'],list):
            raise ValueError(f'Candidates must be a list for {sid}')

        allowed_ids=set(allowed[sid])
        seen_candidates=set()
        selected=[]

        for candidate in result['candidates']:
            if not isinstance(candidate,dict) or set(candidate) != {'capability_id','relevant'}:
                raise ValueError(f'Each candidate decision for {sid} needs capability_id and relevant')
            if not isinstance(candidate['relevant'], bool):
                raise ValueError(f'relevant must be boolean for {sid}')

            cid=clean(candidate['capability_id'])
            if cid not in allowed_ids or cid in seen_candidates:
                raise ValueError(f'Invalid or duplicate candidate {cid} for {sid}')

            seen_candidates.add(cid)
            relevant=candidate['relevant']
            decisions.append({'stage_id':sid,'capability_id':cid,'relevant':relevant})
            if relevant:
                selected.append(cid)

        if seen_candidates != allowed_ids:
            missing=sorted(allowed_ids-seen_candidates)
            raise ValueError(f'Missing candidate decisions for {sid}: {missing}')

        seen_stages.add(sid)
        out[sid]=selected

    if seen_stages != expected:
        missing=sorted(expected-seen_stages)
        raise ValueError(f'Missing stage results: {missing}')

    return out,decisions


def predict(gateway,ctx,stages):
    u=build_user_prompt(ctx,stages)
    allowed={s['stage_id']:[c['capability_id'] for c in s['candidate_l3_capabilities']] for s in stages}
    raw,m=call_llm_with_metrics(gateway,SYSTEM_PROMPT,u,id='9zdn8n',reasoning_effort='low')
    selected,decisions=validate(parse_json_response(raw),[s['stage_id'] for s in stages],allowed)
    return selected,decisions,m


def run():
    g=load_gateway(); rr=[]; cc=[]; relevance=[]

    for theme,rows in evaluation_population.groupby('theme_key',sort=True):
        rows=rows.reset_index(drop=True)
        stages=merged_stages(rows)
        stage_lookup={s['stage_id']:s for s in stages}
        candidate_lookup={
            (s['stage_id'],c['capability_id']):c
            for s in stages
            for c in s['candidate_l3_capabilities']
        }
        t=perf_counter()

        try:
            pred,decisions,m=predict(g,theme_context(rows.iloc[0].to_dict()),stages)

            for decision in decisions:
                sid=decision['stage_id']; cid=decision['capability_id']
                stage=stage_lookup[sid]
                capability=candidate_lookup[(sid,cid)]
                relevance.append({
                    'experiment':EXPERIMENT_NAME,
                    'theme_key':theme,
                    'stage_id':sid,
                    'stage_name':stage.get('stage_name'),
                    'capability_id':cid,
                    'capability_name':capability.get('capability_name'),
                    'capability_description':capability.get('capability_description'),
                    'capability_tier':capability.get('capability_tier'),
                    'relevant':decision['relevant'],
                })

            cc.append({
                'experiment':EXPERIMENT_NAME,'theme_key':theme,'record_count':len(rows),'stage_count':len(stages),
                'status':'ok','latency_seconds':m.get('latency_seconds'),'input_tokens':m.get('input_tokens'),
                'output_tokens':m.get('output_tokens'),'total_tokens':m.get('total_tokens'),'error':None
            })

            for r in rows.to_dict('records'):
                vals=sorted({cid for sid in as_list(r['stage_ids']) for cid in pred.get(sid,[])})
                truth=r['gt_l3_ids']
                rr.append({
                    'experiment':EXPERIMENT_NAME,'theme_key':theme,'record_key':r['record_key'],
                    'stage_ids':as_list(r['stage_ids']),'predicted_l3_ids':vals,'gt_l3_ids':truth,
                    'status':'ok','error':None,**score_sets(vals,truth)
                })

        except Exception as e:
            err=str(e)
            cc.append({
                'experiment':EXPERIMENT_NAME,'theme_key':theme,'record_count':len(rows),'stage_count':len(stages),
                'status':'error','latency_seconds':perf_counter()-t,'input_tokens':None,'output_tokens':None,
                'total_tokens':None,'error':err
            })
            for r in rows.to_dict('records'):
                truth=r['gt_l3_ids']
                rr.append({
                    'experiment':EXPERIMENT_NAME,'theme_key':theme,'record_key':r['record_key'],
                    'stage_ids':as_list(r['stage_ids']),'predicted_l3_ids':None,'gt_l3_ids':truth,
                    'status':'error','error':err,'exact_match':None,'precision':None,'recall':None,'f1':None,
                    'predicted_count':None,'truth_count':len(truth)
                })

    return pd.DataFrame(rr),pd.DataFrame(cc),pd.DataFrame(relevance)


results,call_metrics,candidate_relevance=run()
scored=results[results.status.eq('ok')]
calls=call_metrics[call_metrics.status.eq('ok')]
summary=pd.DataFrame([{
    'scope':'fixed_100_from_golden_valid_set_seed_42_theme_batch_candidate_relevance',
    'evaluated_records':len(scored),
    'exact_match_accuracy':scored.exact_match.mean() if len(scored) else 0,
    'mean_precision':scored.precision.mean() if len(scored) else 0,
    'mean_recall':scored.recall.mean() if len(scored) else 0,
    'mean_f1':scored.f1.mean() if len(scored) else 0,
}])
latency_tokens=pd.DataFrame([{
    'successful_calls':len(calls),
    'failed_calls':int(call_metrics.status.eq('error').sum()),
    'avg_records_per_call':calls.record_count.mean() if len(calls) else None,
    'avg_stages_per_call':calls.stage_count.mean() if len(calls) else None,
    'avg_latency_seconds':calls.latency_seconds.mean() if len(calls) else None,
    'p50_latency_seconds':calls.latency_seconds.quantile(.5) if len(calls) else None,
    'p95_latency_seconds':calls.latency_seconds.quantile(.95) if len(calls) else None,
    'total_input_tokens':calls.input_tokens.sum() if len(calls) else 0,
    'total_output_tokens':calls.output_tokens.sum() if len(calls) else 0,
    'total_tokens':calls.total_tokens.sum() if len(calls) else 0,
    'tokens_per_scored_record':calls.total_tokens.sum()/len(scored) if len(scored) else None,
}])

display(summary)
display(latency_tokens)
display(call_metrics)
display(candidate_relevance)
display(results.head(100))
print('Saved',save_results_excel(
    results,
    EXPERIMENT_NAME,
    'results',
    extra_sheets={
        'evaluation_summary':summary,
        'llm_metrics':call_metrics,
        'latency_tokens':latency_tokens,
        'evaluation_population':evaluation_population,
        'candidate_relevance':candidate_relevance,
    },
))
'''
set_source(prediction, E22_PREDICTION)

# Add the candidate decision sheet to the later prompt-audit save as well.
audit = find_code_cell(e22, "prompt_log = pd.DataFrame(get_prompt_audit_log())")
audit_text = source(audit)
audit_text = audit_text.replace(
    '    "prompt_sample": prompt_sample,\n}',
    '    "prompt_sample": prompt_sample,\n    "candidate_relevance": candidate_relevance,\n}',
)
set_source(audit, audit_text)

E22.write_text(json.dumps(e22, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

print(f"Created {E21}")
print(f"Created {E22}")
