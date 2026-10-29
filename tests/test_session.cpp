#include "test_framework.h"

#include <vector>

#include "edge_prefill/session.h"

using namespace edge_prefill;

TEST_CASE(session_begin_creates_once) {
    SessionStore store;
    SessionState* first = nullptr;
    CHECK_OK(store.begin("s1", "contexts/a.txt", &first));
    CHECK(first != nullptr);
    CHECK_EQ(store.size(), 1u);
    CHECK_EQ(first->id, std::string("s1"));
    CHECK_EQ(first->next_pos, 0u);

    SessionState* again = nullptr;
    CHECK_OK(store.begin("s1", "contexts/a.txt", &again));
    CHECK(again == first);
    CHECK_EQ(store.size(), 1u);
}

TEST_CASE(session_begin_rejects_other_document) {
    SessionStore store;
    SessionState* session = nullptr;
    CHECK_OK(store.begin("s1", "contexts/a.txt", &session));
    // Reusing KV across documents would silently corrupt answers.
    CHECK(!store.begin("s1", "contexts/b.txt", &session).ok());
    CHECK_EQ(store.size(), 1u);
}

TEST_CASE(session_begin_rejects_empty) {
    SessionStore store;
    SessionState* session = nullptr;
    CHECK(!store.begin("", "contexts/a.txt", &session).ok());
    CHECK(!store.begin("s1", "", &session).ok());
    CHECK(!store.begin("s1", "contexts/a.txt", nullptr).ok());
    CHECK_EQ(store.size(), 0u);
}

TEST_CASE(session_reset_clears_log_keeps_id) {
    SessionStore store;
    SessionState* session = nullptr;
    CHECK_OK(store.begin("s1", "contexts/a.txt", &session));
    session->log = {1, 2, 3, 4};
    session->admitted = {10, 20};
    session->next_pos = 4;
    session->pending_tail = "tail";
    session->turns = 2;

    SessionStore::reset(session, "document-restart");
    CHECK(session->log.empty());
    CHECK(session->admitted.empty());
    CHECK(session->pending_tail.empty());
    CHECK_EQ(session->next_pos, 0u);
    CHECK_EQ(session->turns, 0u);
    CHECK_EQ(session->resets, 1u);
    CHECK_EQ(session->reset_reason, std::string("document-restart"));
    CHECK_EQ(session->id, std::string("s1"));
    CHECK_EQ(session->context_path, std::string("contexts/a.txt"));

    SessionStore::reset(session, "again");
    CHECK_EQ(session->resets, 2u);
}

TEST_CASE(session_reset_null_is_noop) { SessionStore::reset(nullptr, "x"); }

TEST_CASE(session_all_lists_sessions) {
    SessionStore store;
    SessionState* a = nullptr;
    SessionState* b = nullptr;
    CHECK_OK(store.begin("s1", "contexts/a.txt", &a));
    CHECK_OK(store.begin("s2", "contexts/a.txt", &b));
    CHECK_EQ(store.size(), 2u);
    CHECK_EQ(store.all().size(), 2u);
    CHECK_EQ(store.all()[1].id, std::string("s2"));
    CHECK(a != b);
}
