#include "edge_prefill/session.h"

#include <algorithm>

namespace edge_prefill {

Status SessionStore::begin(const std::string& id, const std::string& context_path,
                           SessionState** out) {
    if (!out) return Status::error(ErrorCode::InvalidArgument, "null session output");
    if (id.empty()) return Status::error(ErrorCode::InvalidArgument, "empty session id");
    if (context_path.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "empty session context path");
    }
    for (auto& session : sessions_) {
        if (session.id != id) continue;
        if (session.context_path != context_path) {
            return Status::error(
                    ErrorCode::InvalidArgument,
                    "session " + id + " was opened on " + session.context_path +
                            " and cannot continue on " + context_path);
        }
        *out = &session;
        return Status::success();
    }
    sessions_.push_back(SessionState{});
    SessionState& session = sessions_.back();
    session.id = id;
    session.context_path = context_path;
    *out = &session;
    return Status::success();
}

void SessionStore::reset(SessionState* session, std::string reason) {
    if (!session) return;
    session->log.clear();
    session->pending_tail.clear();
    session->admitted.clear();
    session->next_pos = 0;
    session->turns = 0;
    ++session->resets;
    session->reset_reason = std::move(reason);
}

} // namespace edge_prefill
