/** Node 20+. Mount AFTER your existing session/auth middleware.
 * The browser NEVER chooses user_id and NEVER sees BACKEND_API_KEY.
 * For anonymous visitors, use a server-generated signed session identity instead
 * of req.user.id. Do not trust a user_id submitted in JSON or HTTP headers.
 */
import express from 'express';

export function createRagRouter() {
  const router = express.Router();
  router.use(express.json({ limit: '8kb' }));
  router.use((req, res, next) => {
    if (!req.user?.id) return res.status(401).json({ error: 'Login required' });
    next();
  });
  const root = process.env.RAG_URL || 'http://127.0.0.1:8000';
  const secret = process.env.BACKEND_API_KEY;
  if (!secret) throw new Error('Set BACKEND_API_KEY on the website backend');

  async function forward(req, res, path, method, body) {
    try {
      const upstream = await fetch(`${root}${path}`, {
        method,
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${secret}`,
          'X-User-ID': String(req.user.id),
        },
        body: body ? JSON.stringify(body) : undefined,
        signal: AbortSignal.timeout(180000),
      });
      const data = await upstream.json();
      if (upstream.headers.get('Retry-After')) {
        res.set('Retry-After', upstream.headers.get('Retry-After'));
      }
      res.status(upstream.status).json(data);
    } catch {
      res.status(503).json({ error: 'Chat temporarily unavailable. Retry with the same request_id.' });
    }
  }
  router.post('/conversations', (req, res) => forward(req,res,'/conversations','POST'));
  router.get('/conversations', (req, res) => {
    const cursor = String(req.query.before || '');
    if (cursor.length > 256 || !/^[A-Za-z0-9_=-]*$/.test(cursor)) {
      return res.status(400).json({ error: 'Invalid conversation cursor' });
    }
    const query = new URLSearchParams({ limit: '20' });
    if (cursor) query.set('before', cursor);
    forward(req,res,`/conversations?${query.toString()}`,'GET');
  });
  router.post('/chat', (req, res) => forward(req,res,'/chat','POST', {
    conversation_id: req.body?.conversation_id,
    request_id: req.body?.request_id,
    message: req.body?.message,
  }));
  // UUID validation prevents user-controlled URL/path traversal.
  const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  router.get('/conversations/:id/messages', (req,res) => {
    if (!uuid.test(req.params.id)) return res.status(400).json({ error:'Invalid conversation ID' });
    const after = Number(req.query.after || 0);
    if (!Number.isSafeInteger(after) || after < 0) return res.status(400).json({ error:'Invalid cursor' });
    forward(req,res,`/conversations/${req.params.id}/messages?after=${after}&limit=50`,'GET');
  });
  router.delete('/conversations/:id', (req,res) => {
    if (!uuid.test(req.params.id)) return res.status(400).json({ error:'Invalid conversation ID' });
    forward(req,res,`/conversations/${req.params.id}`,'DELETE');
  });
  return router;
}
// In your existing Express server:
// import { createRagRouter } from './express-proxy.mjs';
// app.use('/api/rag', yourAuthMiddleware, createRagRouter());
// Keep your existing CSRF protection if your authentication uses cookies.
