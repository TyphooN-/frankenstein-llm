#!/bin/bash
watch -n1 "curl -s 'http://127.0.0.1:8080/slots?model=heretic' |
jq '.[0] | {
  processing: .is_processing,
  task: .id_task,
  prompt_done: .n_prompt_tokens_processed,
  decoded: .next_token[0].n_decoded,
  next: .next_token[0].has_next_token
}'"
