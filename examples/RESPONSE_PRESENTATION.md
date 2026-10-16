# Response presentation

The backend now requests readable Markdown: direct answers, short sections,
numbered procedures, bullet lists, and comparison tables when useful. The JSON
response shape, source validation, authentication, and persistence are unchanged.
Existing saved replies retain their original text.

This repository does not contain the deployed website frontend. To enable the
loading indicator and styled answers there, copy `browser-client.js`,
`chat-response.js`, and `chat-response.css` into your website's static assets and
use the optional component in place of rendering `data.answer` as plain text:

```html
<link rel="stylesheet" href="/assets/chat-response.css" />
<div id="assistant-reply"></div>
```

```js
import {
  sendMessageWithPresentation,
  renderAnswer,
} from "/assets/chat-response.js";

// In your existing authenticated chat form's submit handler:
// Create a separate reply container for each message.
const requestId = crypto.randomUUID(); // retain this ID if this message fails
try {
  await sendMessageWithPresentation({
    container: document.querySelector("#assistant-reply"),
    conversationId, // from your existing createConversation() flow
    message, // the submitted text
    requestId,
    submitButton: document.querySelector('button[type="submit"]'),
  });
} catch (error) {
  // Connect to your existing retry handler, reusing requestId and message.
  console.error("Reply failed", error.status);
}

// For persisted assistant messages, pass the saved answer and source metadata:
// renderAnswer(replyContainer, { answer: savedText, sources: savedSources });
```

The indicator stays visible for the actual request duration. It is a loading
state, not streamed model reasoning. Replies appear when the existing API returns;
there is no artificial typing delay. Markdown support includes headings, bold,
flat lists, paragraphs and simple pipe tables. Raw HTML and arbitrary Markdown
links display as text. Only validated HTTPS source metadata becomes a link.
Tables scroll on narrow screens; reduced-motion preferences disable the pulse.
Serve the JavaScript as ES modules through your website, not `file://`.
