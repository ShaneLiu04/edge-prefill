#include "edge_prefill/attention_sparsity.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace edge_prefill {

Status AttentionSparsityController::build(const Selection& selection,
                                          const AttentionSparsityOptions& opts,
                                          std::vector<int32_t>* attention_indices,
                                          std::string* reason) {
    if (!attention_indices) {
        return Status::error(ErrorCode::InvalidArgument, "null attention indices output");
    }
    if (opts.block_size == 0) {
        return Status::error(ErrorCode::InvalidArgument, "block size must be > 0");
    }
    if (opts.per_layer && opts.layer_top_k.empty()) {
        return Status::error(ErrorCode::InvalidArgument,
                             "per_layer requires a non-empty layer_top_k");
    }

    attention_indices->clear();
    if (reason) reason->clear();

    if (selection.dense) {
        // Dense prefill computes full attention; gathering is pure overhead.
        if (reason) *reason = "dense";
        return Status::success();
    }

    if (selection.token_indices.empty()) {
        return Status::error(ErrorCode::InvalidArgument,
                             "non-dense selection has no token indices");
    }
    const uint32_t n_tokens = selection.total_tokens;
    if (n_tokens == 0) {
        return Status::error(ErrorCode::InvalidArgument, "selection has no tokens");
    }
    const uint32_t block_size = opts.block_size;
    const uint32_t n_blocks = (n_tokens + block_size - 1) / block_size;
    if (selection.signals.size() != n_blocks) {
        return Status::error(ErrorCode::InvalidArgument, "selection signals size mismatch");
    }

    for (uint32_t t : selection.token_indices) {
        if (t >= n_tokens) {
            return Status::error(ErrorCode::InvalidArgument, "token index out of range");
        }
    }
    for (size_t i = 1; i < selection.token_indices.size(); ++i) {
        if (selection.token_indices[i] <= selection.token_indices[i - 1]) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "token indices must be strictly ascending");
        }
    }

    if (opts.top_k_blocks == 0) {
        if (reason) *reason = "attn-disabled";
        return Status::success();
    }

    std::set<uint32_t> mandatory(selection.mandatory_blocks.begin(),
                                 selection.mandatory_blocks.end());
    std::set<uint32_t> selected(selection.block_indices.begin(), selection.block_indices.end());
    for (uint32_t b : mandatory) {
        if (b >= n_blocks) {
            return Status::error(ErrorCode::InvalidArgument, "mandatory block out of range");
        }
    }
    for (uint32_t b : selected) {
        if (b >= n_blocks) {
            return Status::error(ErrorCode::InvalidArgument, "selected block out of range");
        }
    }

    // Rank selected blocks by fused score (descending, stable on index) and
    // keep the top_k_blocks; mandatory blocks are always kept.
    std::vector<uint32_t> order(selected.begin(), selected.end());
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (std::abs(selection.signals[a].fused - selection.signals[b].fused) > 1e-12) {
            return selection.signals[a].fused > selection.signals[b].fused;
        }
        return a < b;
    });

    std::set<uint32_t> kept(mandatory.begin(), mandatory.end());
    for (uint32_t b : order) {
        if (kept.size() >= opts.top_k_blocks) break;
        kept.insert(b);
    }
    // Mandatory blocks may exceed the cap; that is intentional: anchors are
    // never sacrificed to meet top_k_blocks.

    if (kept.size() >= selected.size()) {
        // Gathering every selected block is equivalent to the unpatched
        // path, so skip the patch entirely.
        if (reason) *reason = "attn-top-k-covers-all";
        return Status::success();
    }

    // Only tokens that actually get prefilled can be gathered.
    std::vector<bool> in_kept_block(n_blocks, false);
    for (uint32_t b : kept) in_kept_block[b] = true;
    for (uint32_t t : selection.token_indices) {
        if (in_kept_block[t / block_size]) {
            attention_indices->push_back(static_cast<int32_t>(t));
        }
    }

    // The last prompt token must stay gatherable: it produces the
    // first-token logits. It lives in a mandatory (recent) block, but the
    // selector may have trimmed it below block granularity, so re-check.
    const int32_t last = static_cast<int32_t>(n_tokens - 1);
    if (attention_indices->empty() || attention_indices->back() != last) {
        auto it = std::lower_bound(attention_indices->begin(), attention_indices->end(), last);
        if (it == attention_indices->end() || *it != last) {
            attention_indices->insert(it, last);
        }
    }

    if (reason) {
        *reason = "attn-top-k-blocks:" + std::to_string(kept.size()) + "/" +
                  std::to_string(selected.size());
    }
    return Status::success();
}

} // namespace edge_prefill
