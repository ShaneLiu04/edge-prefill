#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <llama.h>

#include "edge_prefill/status.h"

namespace edge_prefill {

// AR014 FR-1: one conversation's append-only context.
//
// The log is the *only* thing that decides what the model sees: turns append
// their newly admitted evidence followed by the question, and the session arm
// decodes just that tail while the full-log arm decodes the whole thing. Both
// arms therefore run exactly the same token sequence at exactly the same
// positions, which is what makes "reuse must not change quality" a testable
// invariant instead of a hope.
struct SessionState {
    std::string id;
    // Document the session was opened on. A turn that points somewhere else is
    // an error: reusing KV across documents would silently corrupt answers.
    std::string context_path;
    std::vector<llama_token> log;
    // Original document token indices already inside `log`, ascending. The
    // incremental admission is the set difference against this.
    std::vector<uint32_t> admitted;
    // Next free position; always equal to log.size().
    uint32_t next_pos = 0;
    // Tokens that close the previous assistant turn (`generated` + the chat
    // template's turn terminator). They are appended at the start of the next
    // turn so the conversation stays well formed while remaining append-only.
    std::string pending_tail;
    uint32_t turns = 0;
    uint32_t resets = 0;
    std::string reset_reason;
};

// Process-local table of sessions (one llama sequence each).
class SessionStore {
public:
    // Returns the session for `id`, creating it on first use. Fails when an
    // existing session was opened on a different document.
    Status begin(const std::string& id, const std::string& context_path,
                 SessionState** out);

    // Drops the context but keeps the session (used when an append fails or the
    // caller explicitly restarts a conversation).
    static void reset(SessionState* session, std::string reason);

    size_t size() const { return sessions_.size(); }

    const std::vector<SessionState>& all() const { return sessions_; }

private:
    std::vector<SessionState> sessions_;
};

} // namespace edge_prefill
