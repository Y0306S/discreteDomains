"""Check rationale completion without silently shortening model responses."""

VALIDATION_VERSION = 2
GENERATION_LIMITS = (192, 384, 512)


def rationale_text(text):
    return text.strip().removeprefix('Rationale:').strip()


def text_error(text):
    rationale = rationale_text(text)
    if not rationale:
        return 'empty_rationale'
    if any(mark in rationale for mark in ('{', '}', '<think>', '</think>', '```')):
        return 'invalid_format'
    if len(rationale.splitlines()) != 1:
        return 'multiline_response'
    if not rationale.rstrip('\"\'`)]').endswith(('.', '!', '?')):
        return 'incomplete_sentence'
    return None


def generation_details(tokenizer, token_ids, max_new_tokens, eos_ids):
    eos_ids = list(eos_ids)
    token_ids = list(token_ids)
    stop = next((i for i, token in enumerate(token_ids) if token in eos_ids), None)
    ended_with_eos = stop is not None
    if ended_with_eos:
        token_ids = token_ids[:stop + 1]
    return dict(text=tokenizer.decode(token_ids, skip_special_tokens=True),
                token_ids=token_ids, generated_tokens=len(token_ids),
                max_new_tokens=max_new_tokens, eos_ids=eos_ids,
                ended_with_eos=ended_with_eos,
                finish_reason='eos' if ended_with_eos else
                    ('length' if len(token_ids) >= max_new_tokens else 'unknown'),
                stop_reason_verified=True)


def generation_error(details):
    if details['generated_tokens'] > details['max_new_tokens']:
        return 'generation_exceeds_limit'
    if not details['ended_with_eos']:
        return 'generation_' + details['finish_reason']
    return text_error(details['text'])


def record_error(record, tokenizer):
    attempts = record.get('attempts', [])
    if not attempts or rationale_text(attempts[-1]) != record.get('rationale'):
        return 'stored_rationale_differs_from_response'
    if record.get('validation_version') == VALIDATION_VERSION:
        generations = record.get('generation_attempts', [])
        if len(generations) != len(attempts):
            return 'missing_generation_evidence'
        final = generations[-1]
        try:
            checked = generation_details(tokenizer, final['token_ids'],
                                         final['max_new_tokens'], final['eos_ids'])
        except (KeyError, TypeError):
            return 'invalid_generation_evidence'
        if checked['text'] != attempts[-1] or any(final.get(key) != value for key, value in checked.items()):
            return 'generation_evidence_mismatch'
        return generation_error(checked)
    error = text_error(attempts[-1])
    if error:
        return error
    # Older records discarded EOS IDs. Preserve complete short replies, but mark
    # their stopping evidence as inferred in the audit rather than inventing EOS.
    tokens = tokenizer.encode(attempts[-1], add_special_tokens=False)
    if len(tokens) >= record.get('max_new_tokens', 192):
        return 'legacy_response_at_token_limit'
    return None
