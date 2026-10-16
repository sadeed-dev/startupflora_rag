// Optional presentation layer. Existing API and sendMessage callers stay compatible.
import { sendMessage } from './browser-client.js';

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

// A deliberately small Markdown subset. All model text uses textContent;
// HTML, images and arbitrary Markdown links are never executed.
function inline(parent, text, sources) {
  for (const part of text.split(/(\*\*[^*]+\*\*|\[S\d+\])/g)) {
    if (part.startsWith('**') && part.endsWith('**')) {
      parent.append(element('strong', part.slice(2, -2)));
    } else if (/^\[S\d+\]$/.test(part)) {
      const source = sources.find(item => item.id === part.slice(1, -1));
      const link = sourceLink(source, part);
      parent.append(link || document.createTextNode(part));
    } else parent.append(document.createTextNode(part));
  }
}

function sourceLink(source, label) {
  try {
    const url = new URL(source.url);
    if (url.protocol !== 'https:') return null;
    const link = element('a', label, 'rag-source');
    link.href = url.href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    return link;
  } catch { return null; }
}

const cells = line => line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(x => x.trim());
const divider = line => line.includes('|') && cells(line).every(x => /^:?-{3,}:?$/.test(x));

/** Render a complete API answer, including safe inline citations and source links. */
export function renderAnswer(container, data) {
  container.replaceChildren();
  container.classList.add('rag-answer');
  const sources = Array.isArray(data.sources) ? data.sources : [];
  const lines = String(data.answer || '').replace(/\r\n?/g, '\n').split('\n');
  let list = null;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) { list = null; continue; }
    if (line.includes('|') && i + 1 < lines.length && divider(lines[i + 1])) {
      const wrapper = element('div', undefined, 'rag-table-scroll');
      wrapper.tabIndex = 0;
      wrapper.setAttribute('role', 'region');
      wrapper.setAttribute('aria-label', 'Answer table');
      const table = element('table');
      const head = element('thead');
      const row = element('tr');
      const headers = cells(line);
      for (const value of headers) {
        const cell = element('th');
        cell.scope = 'col';
        inline(cell, value, sources);
        row.append(cell);
      }
      head.append(row);
      table.append(head);
      const body = element('tbody');
      i += 2;
      while (i < lines.length && lines[i].trim() && lines[i].includes('|')) {
        const values = cells(lines[i]);
        const tr = element('tr');
        headers.forEach((_, column) => {
          const td = element('td');
          inline(td, values[column] || '', sources);
          tr.append(td);
        });
        body.append(tr);
        i++;
      }
      i--;
      table.append(body);
      wrapper.append(table);
      container.append(wrapper);
      list = null;
      continue;
    }
    const heading = line.match(/^#{1,6}\s+(.+)$/);
    const item = line.match(/^(?:([-*])|\d+[.)])\s+(.+)$/);
    if (item) {
      const tag = item[1] ? 'ul' : 'ol';
      if (!list || list.localName !== tag) {
        list = element(tag);
        if (tag === 'ol') list.start = Number.parseInt(line, 10);
        container.append(list);
      }
      const li = element('li');
      inline(li, item[2], sources);
      list.append(li);
    } else {
      list = null;
      const block = element(heading ? 'h3' : 'p');
      inline(block, heading ? heading[1] : line, sources);
      container.append(block);
    }
  }
  const links = element('div', undefined, 'rag-sources');
  for (const source of sources) {
    const link = sourceLink(source, `${source.id} · ${source.title || 'Source'}`);
    if (link) links.append(link);
  }
  if (links.childElementCount) {
    const nav = element('nav');
    nav.setAttribute('aria-label', 'Answer sources');
    nav.append(element('h3', 'Sources'), links);
    container.append(nav);
  }
}

/** Show loading until the real request completes; preserve requestId for retries. */
export async function sendMessageWithPresentation({ container, conversationId, message,
  requestId = crypto.randomUUID(), submitButton }) {
  if (container.getAttribute('aria-busy') === 'true') {
    throw new Error('A reply is already being prepared in this container');
  }
  const wasDisabled = submitButton?.disabled;
  if (submitButton) submitButton.disabled = true;
  container.classList.add('rag-answer');
  container.setAttribute('aria-busy', 'true');
  const loading = element('div', 'Thinking…', 'rag-thinking');
  loading.setAttribute('role', 'status');
  container.replaceChildren(loading);
  try {
    const data = await sendMessage(conversationId, message, requestId);
    renderAnswer(container, data);
    return data;
  } catch (error) {
    const alert = element('p', 'Could not load the reply. Please retry your message.', 'rag-error');
    alert.setAttribute('role', 'alert');
    container.replaceChildren(alert);
    throw error;
  } finally {
    container.setAttribute('aria-busy', 'false');
    if (submitButton) submitButton.disabled = wasDisabled;
  }
}
