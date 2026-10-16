// Browser side: call only your own website backend. No API keys here.
// Keep conversationId in component state, clear on logout/user change.
/** Create a user-owned chat after your website backend authenticates the request. */
export async function createConversation() {
  const r = await fetch('/api/rag/conversations', { method: 'POST', credentials: 'same-origin' });
  if (!r.ok) throw new Error('Could not create conversation');
  return (await r.json()).conversation_id;
}

// Create requestId ONCE per message, reuse when retrying that exact message.
/** Send a message; reuse requestId for retries of this exact text. */
export async function sendMessage(conversationId, message, requestId = crypto.randomUUID()) {
  const r = await fetch('/api/rag/chat', {
    method: 'POST', credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conversation_id: conversationId, request_id: requestId, message }),
  });
  const data = await r.json();
  if (!r.ok) throw Object.assign(new Error(data.detail || data.error || 'Chat failed'), { requestId, status: r.status });
  return data;
}

// Render answer as plain text, or use a sanitized Markdown renderer.
// Do not use innerHTML. Render data.sources separately as links after validating
// URL protocol === 'https:'. [S1] in answer corresponds to sources[i].id.
// Persist/load history with GET /api/rag/conversations/:id/messages?after=0.
// Never render one account's cached history after another account logs in.


/** Load this signed-in user's prior chats; clear cached lists on logout. */
export async function listConversations(cursor = '') {
  const query = new URLSearchParams();
  if (cursor) query.set('before', cursor);
  const response = await fetch(`/api/rag/conversations?${query}`, { credentials: 'same-origin' });
  if (!response.ok) throw new Error('Could not load conversations');
  return response.json();
}
