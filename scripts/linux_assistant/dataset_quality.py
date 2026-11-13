"""Fail-closed structural and hidden-constraint checks for rebuilt data."""
from __future__ import annotations
import json
import re


FORMAT_REFERENCE = re.compile(r"(?:上述|以上|前述|同样|相同|仍用|照旧).{0,12}(?:格式|字段|结构|模板|JSON)")
PATH_RE = re.compile(r"(?:^|\s|[=:])((?:/|\./|\.\./)[\w./-]+|[\w.-]+/[\w./-]+)")
DESTRUCTIVE = re.compile(r"(?:^|[;&|\n]\s*)(?:sudo\s+)?(?:rm\b|mv\b|rmdir\b|mkfs\b|dd\b)|\bsed\s+-i\b|\bchmod\s+0?777\b", re.I)


def conversation_issues(messages: list[dict]) -> list[str]:
    """The unified schema starts with a user and alternates complete chat turns."""
    issues = []
    for index, message in enumerate(messages):
        expected = 'user' if index % 2 == 0 else 'assistant'
        if message.get('role') != expected:
            issues.append('non_alternating_conversation')
            break
    if any(left.get('role') == right.get('role') == 'assistant' and
           left.get('content') == right.get('content')
           for left, right in zip(messages, messages[1:])):
        issues.append('duplicated_adjacent_assistant_target')
    return issues


def input_text(row: dict) -> str:
    messages = row.get('messages', [])
    if messages and messages[-1].get('role') == 'assistant':
        messages = messages[:-1]
    return '\n'.join(m.get('content', '') for m in messages if m.get('role') in ('user', 'assistant'))


def row_issues(row: dict) -> list[str]:
    """Return deterministic blocking issues; this never grants content approval."""
    issues = []
    messages = row.get('messages')
    if not isinstance(messages, list) or not messages:
        return ['missing_messages']
    if any(not isinstance(m, dict) or m.get('role') not in ('user', 'assistant') or not isinstance(m.get('content'), str) for m in messages):
        issues.append('invalid_message')
        return issues
    issues.extend(conversation_issues(messages))
    if messages[-1].get('role') == 'assistant':
        target_content = messages[-1].get('content', '')
    elif messages[-1].get('role') == 'user':
        target_content = row.get('reference', '')
    else:
        target_content = ''
    if not isinstance(target_content, str) or not target_content.strip():
        issues.append('missing_final_assistant_target')
        return issues
    if target_content != row.get('reference'):
        issues.append('reference_target_mismatch')
    prompt = input_text(row)
    if not prompt.strip():
        issues.append('empty_model_input')

    for snippet in row.get('grading', {}).get('content', {}).get('required_snippets', []):
        if not isinstance(snippet, str) or not snippet or snippet not in prompt or snippet not in target_content:
            issues.append('source_required_snippet_missing')

    if FORMAT_REFERENCE.search(prompt):
        earlier = prompt[:FORMAT_REFERENCE.search(prompt).start()]
        defined = bool(re.search(r"(?:字段|标签|格式|JSON|结构|模板).{0,50}[A-Za-z_][A-Za-z0-9_-]*|[A-Za-z_][A-Za-z0-9_-]*.{0,20}(?:字段|标签)", earlier, re.S))
        if not defined:
            issues.append('missing_prior_format_definition')

    # Exact JSON keys in the target must be specified in the visible conversation,
    # rather than appearing only in grader metadata or the reference answer.
    try:
        answer = json.loads(target_content)
    except (json.JSONDecodeError, TypeError):
        answer = None
    if isinstance(answer, dict):
        missing = [key for key in answer if str(key) not in prompt]
        if missing and re.search(r"(?:只|仅|严格).{0,12}(?:JSON|json)|JSON对象|JSON格式", prompt):
            issues.append('undefined_json_field:' + ','.join(map(str, missing)))

    # A prompt that points to a file must provide its path, or explicitly define a
    # placeholder. Do not accept a concrete invented path in the reference.
    if re.search(r"(?:这个文件|该文件|此文件|提供的文件)", prompt) and not PATH_RE.search(prompt):
        answer_paths = PATH_RE.findall(target_content)
        if answer_paths:
            issues.append('filename_missing_from_input')

    if re.search(r"(?:只读|只查看|不要.{0,4}(?:修改|删除|写入)|不得.{0,4}(?:修改|删除|写入))", prompt) and DESTRUCTIVE.search(target_content):
        issues.append('command_conflicts_with_user_constraint')

    evidence = row.get('evidence', {})
    scope = evidence.get('scope', {}) if isinstance(evidence, dict) else {}
    supported_scope = ' '.join(str(scope.get(k, '')) for k in ('board', 'soc', 'verified_board'))
    if supported_scope.strip():
        mismatch = False
        for match in re.finditer(r"(?:LubanCat[- ]?\d+|鲁班猫\s*\d+|RK\d{3,4})", target_content, re.I):
            mentioned = match.group(0)
            sentence_start = max(target_content.rfind(mark, 0, match.start()) for mark in ('。', '！', '？', '\n')) + 1
            sentence_end = min([p for p in (target_content.find(mark, match.end()) for mark in ('。', '！', '？', '\n')) if p >= 0] or [len(target_content)])
            sentence = target_content[sentence_start:sentence_end]
            normalized_scope = re.sub(r"[\s-]", '', supported_scope).lower()
            normalized_mention = re.sub(r"[\s-]", '', mentioned).lower()
            negated = bool(re.search(r"(?:未验证|没有验证|尚未验证|未测试|没有测试|不代表|不能称为|不能说|不能推断).{0,20}" + re.escape(mentioned), sentence, re.I)
                           or re.search(re.escape(mentioned) + r".{0,20}(?:未验证|尚未验证|未测试|不代表|不能称为|不能说|不能推断)", sentence, re.I))
            if normalized_mention not in normalized_scope and not negated:
                mismatch = True
        if mismatch:
            issues.append('board_scope_mismatch')
    return issues


def split_issues(splits: dict[str, list[dict]]) -> list[dict]:
    issues = []
    group_to_split = {}
    exact_input_to_split = {}
    ids = set()
    for split, rows in splits.items():
        for row in rows:
            if row.get('id') in ids:
                issues.append({'id': row.get('id'), 'issue': 'duplicate_record_id'})
            ids.add(row.get('id'))
            group = row.get('group')
            if group:
                previous = group_to_split.get(group)
                if previous and previous != split:
                    issues.append({'id': row.get('id'), 'issue': 'cross_split_case_group', 'other_split': previous})
                group_to_split[group] = split
            key = re.sub(r"\s+", " ", input_text(row)).strip().casefold()
            if key:
                previous = exact_input_to_split.get(key)
                if previous:
                    issue = 'cross_split_exact_input' if previous != split else 'duplicate_exact_input'
                    issues.append({'id': row.get('id'), 'issue': issue, 'other_split': previous})
                exact_input_to_split[key] = split
            issues.extend({'id': row.get('id'), 'issue': issue} for issue in row_issues(row))
    return issues

