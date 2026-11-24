"""Allow vLLM 0.31.0's NIXL pull connector to count embedding-only prompts."""

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import NixlPullConnector


class LatentNixlPullConnector(NixlPullConnector):
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params
        if params and params.get('do_remote_prefill'):
            assert self.connector_scheduler is not None
            remote = params.get('remote_num_tokens') or request.num_prompt_tokens
            positions = min(remote, request.num_prompt_tokens)
            available = self.connector_scheduler._get_remote_prefill_token_count(positions)
            count = available - num_computed_tokens
            return max(0, count), count > 0
        return super().get_num_new_matched_tokens(request, num_computed_tokens)
