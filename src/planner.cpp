#include "edge_prefill/planner.h"

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "edge_prefill/budget_curve.h"
#include "edge_prefill/ffn_skip.h"

namespace edge_prefill {
namespace {

void add_reason(Plan* plan, const std::string& reason) {
    if (std::find(plan->reasons.begin(), plan->reasons.end(), reason) == plan->reasons.end()) {
        plan->reasons.push_back(reason);
    }
}

// AR007 FR-3: the curve is read once per (path, mtime) pair. A missing or
// broken file degrades to the mode default with a reason, never a failure:
// budget selection is an optimisation, not a correctness requirement.
struct LoadedCurve {
    std::string path;
    std::filesystem::file_time_type mtime{};
    bool valid = false;
    BudgetCurve curve;
};

const BudgetCurve* cached_curve(const std::string& path) {
    static LoadedCurve cache;
    if (path.empty()) return nullptr;
    std::error_code ec;
    const auto mtime = std::filesystem::last_write_time(path, ec);
    if (ec) return nullptr;
    if (cache.valid && cache.path == path && cache.mtime == mtime) return &cache.curve;
    cache.path = path;
    cache.mtime = mtime;
    cache.valid = false;
    BudgetCurve curve;
    if (load_budget_curve(path, &curve).ok()) {
        cache.curve = std::move(curve);
        cache.valid = true;
    }
    return cache.valid ? &cache.curve : nullptr;
}

} // namespace

Status Planner::make_plan(const PlannerInput& in, const Config& cfg, Plan* out) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null plan output");
    if (in.n_ubatch == 0) return Status::error(ErrorCode::InvalidArgument, "n_ubatch must be > 0");
    if (in.prefix_hit > in.n_tokens) {
        return Status::error(ErrorCode::InvalidArgument, "prefix_hit exceeds n_tokens");
    }

    switch (in.mode) {
        case Mode::Latency:
        case Mode::Balanced:
        case Mode::Quality:
        case Mode::Reuse:
        case Mode::Dense:
            break;
        default:
            return Status::error(ErrorCode::InvalidArgument, "unknown planner mode");
    }

    Plan plan;
    plan.block_size = cfg.block_size;
    plan.chunk_size = cfg.chunk_size;

    if (in.n_tokens == 0) {
        plan.dense_fallback = true;
        plan.budget = 1.0;
        plan.budget_source = "default";
        add_reason(&plan, "empty");
        *out = std::move(plan);
        return Status::success();
    }

    if (in.n_tokens < 512 || cfg.budget >= 0.95) {
        plan.dense_fallback = true;
        plan.budget = 1.0;
        plan.budget_source = cfg.budget_overridden ? "override" : "default";
        add_reason(&plan, in.n_tokens < 512 ? "short-prompt" : "budget>=0.95");
        *out = std::move(plan);
        return Status::success();
    }

    if (in.n_tokens < 2048 && in.costs.dense_ms_per_token > 0.0) {
        const double c_prefill = in.costs.dense_ms_per_token * in.n_tokens;
        const double c_select = std::max(in.costs.select_base_ms,
                                         in.costs.select_ms_per_token * in.n_tokens);
        const double ffn_saving = in.n_layer > 0
                                      ? 0.66 * (static_cast<double>(cfg.skip_ffn_mask != 0 ? 1 : 0))
                                      : 0.0;
        const double t_opt = c_select + cfg.budget * c_prefill * (1.0 - ffn_saving);
        if (t_opt >= 0.98 * c_prefill) {
            plan.dense_fallback = true;
            plan.budget = 1.0;
            plan.budget_source = "default";
            add_reason(&plan, "negative-estimated-gain");
            *out = std::move(plan);
            return Status::success();
        }
    }

    bool prefix_reuse = in.prefix_hit >= in.n_tokens / 2 && in.prefix_hit > 0;
    double budget = 0.5;
    uint64_t mask = 0;
    KvPrecision kv = KvPrecision::Q8_0;
    bool skip = false;

    if (prefix_reuse) {
        budget = 0.7;
        kv = KvPrecision::Q8_0;
        add_reason(&plan, "prefix-reuse");
    } else if (in.mode == Mode::Latency) {
        budget = 0.4;
        kv = KvPrecision::Q8_0;
        skip = true;
    } else if (in.mode == Mode::Quality) {
        budget = 0.7;
        kv = KvPrecision::F16;
        skip = false;
    } else if (in.mode == Mode::Reuse) {
        budget = 0.7;
        kv = KvPrecision::Q8_0;
        prefix_reuse = true;
        add_reason(&plan, "prefix-reuse");
    } else if (in.mode == Mode::Dense) {
        plan.dense_fallback = true;
        plan.budget = 1.0;
        plan.budget_source = "default";
        add_reason(&plan, "mode-dense");
        *out = std::move(plan);
        return Status::success();
    } else {
        // Balanced keeps FFN fully dense by default: skipping 25% of layers
        // measurably degrades long-context retrieval. Users can still opt in
        // via --skip-ffn or the latency mode.
        budget = 0.5;
        kv = KvPrecision::Q8_0;
        skip = false;
    }

    if (cfg.budget_overridden) {
        budget = cfg.budget;
        add_reason(&plan, "cli-override");
    }
    if (cfg.kv_overridden) {
        kv = cfg.kv_type;
        add_reason(&plan, "cli-override");
    }
    if (cfg.skip_ffn_overridden) {
        skip = cfg.skip_ffn_mask != 0;
        mask = cfg.skip_ffn_mask;
        add_reason(&plan, "cli-override");
    }

    // AR007 FR-3: the measured curve picks the admission budget when the
    // user did not override it. The runner clears budget_curve when the
    // model fingerprint does not match, so reaching this branch means the
    // numbers came from this model.
    plan.budget_source = cfg.budget_overridden ? "override" : "default";
    if (!cfg.budget_overridden && !cfg.budget_curve.empty()) {
        if (const BudgetCurve* curve = cached_curve(cfg.budget_curve)) {
            double recommended = 0.0;
            if (curve->recommend(in.n_tokens, &recommended)) {
                budget = recommended;
                plan.budget_source = "model";
                add_reason(&plan, "budget-curve");
            } else {
                add_reason(&plan, "budget-curve-empty");
            }
        } else {
            add_reason(&plan, "budget-curve-unavailable");
        }
    }

    if (in.n_tokens > 8192) {
        // The curve may admit more than the legacy 4096-token ceiling: its
        // recommendation was measured to hold quality. Explicit overrides
        // keep the legacy clamp so user-visible numbers do not shift.
        const double ceiling = plan.budget_source == "model" ? 8192.0 : 4096.0;
        const double clamped = std::min(budget, ceiling / static_cast<double>(in.n_tokens));
        if (clamped < budget) {
            budget = clamped;
            if (plan.budget_source != "model") plan.budget_source = "clamp";
            add_reason(&plan, "budget-clamped");
        }
    }

    if (!in.arch_supports_ffn_skip || in.n_layer <= 0 || in.n_layer > 64) {
        skip = false;
        mask = 0;
        add_reason(&plan, "pafs-disabled");
    } else if (skip && mask == 0) {
        uint64_t generated = 0;
        auto st = FfnSkipController::auto_mask(in.n_layer, 0.25, &generated);
        if (!st.ok()) {
            add_reason(&plan, "pafs-disabled");
            skip = false;
        } else {
            mask = generated;
        }
    }

    plan.budget = budget;
    plan.kv_type = kv;
    plan.skip_ffn_mask = skip ? mask : 0;
    plan.dense_fallback = false;

    // Attention-level top-k only applies to the sparse path; dense fallback
    // plans keep attention_top_k = 0 (their early returns never set it).
    if (cfg.attn_top_k > 0) {
        plan.attention_top_k = cfg.attn_top_k;
        add_reason(&plan, "attn-top-k:" + std::to_string(cfg.attn_top_k));
        if (!cfg.attn_top_k_layers.empty()) {
            std::vector<uint32_t> layers;
            const auto st = parse_u32_list(cfg.attn_top_k_layers, &layers);
            if (!st.ok()) return st;
            add_reason(&plan, "attn-top-k-per-layer");
        }
    }

    plan.passes.push_back("sparse_prefill");
    if (prefix_reuse) plan.passes.push_back("prefix_reuse");
    if (cfg.enable_chunking) plan.passes.push_back("chunked_prefill");
    if (plan.skip_ffn_mask != 0) plan.passes.push_back("ffn_skip");
    if (plan.attention_top_k > 0) plan.passes.push_back("attn_top_k");
    plan.passes.push_back("kv_precision");

    *out = std::move(plan);
    return Status::success();
}

} // namespace edge_prefill
