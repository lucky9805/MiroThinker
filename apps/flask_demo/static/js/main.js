document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatContainer = document.getElementById('chat-container');
    const welcomeMessage = document.getElementById('welcome-message');
    const sendBtn = document.getElementById('send-btn');
    const stopBtn = document.getElementById('stop-btn');
    const newChatBtn = document.getElementById('new-chat-btn');
    const modelSelect = document.getElementById('model-select');
    const agentSelect = document.getElementById('agent-select');

    let currentController = null;
    let isGenerating = false;
    let conversationHistory = []; // Stores {role, content} objects
    let currentAIResponseText = ""; // Accumulates streaming text for history

    // Auto-resize textarea
    userInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value === '') this.style.height = '48px';
    });

    // Handle Enter key (Shift+Enter for new line)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isGenerating && userInput.value.trim()) {
                submitForm();
            }
        }
    });

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submitForm();
    });

    stopBtn.addEventListener('click', () => {
        if (currentController) {
            currentController.abort();
            currentController = null;
            finishGeneration();
            appendSystemMessage("Generation stopped by user.");
        }
    });

    newChatBtn.addEventListener('click', () => {
        if (isGenerating && currentController) {
            currentController.abort();
        }
        chatContainer.innerHTML = '';
        chatContainer.appendChild(welcomeMessage);
        welcomeMessage.style.display = 'flex';
        userInput.value = '';
        userInput.style.height = '48px';
    });

    async function submitForm() {
        const query = userInput.value.trim();
        if (!query) return;

        // UI Updates
        userInput.value = '';
        userInput.style.height = '48px';
        welcomeMessage.style.display = 'none';

        appendUserMessage(query);

        // Add user message to history
        conversationHistory.push({ role: 'user', content: query });

        startGeneration();

        // Reset AI response accumulator
        currentAIResponseText = "";

        const model = modelSelect.value;
        const agent = agentSelect.value;

        currentController = new AbortController();
        const signal = currentController.signal;
        let responseId = null;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query,
                    model,
                    agent,
                    history: conversationHistory // Send full history
                }),
                signal
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            // Create a placeholder for the agent response
            responseId = 'msg-' + Date.now();
            createAgentMessageBubbles(responseId);

            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                buffer += chunk;

                const lines = buffer.split('\n\n');
                buffer = lines.pop(); // Keep incomplete last line

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6);
                        try {
                            const data = JSON.parse(dataStr);
                            handleStreamEvent(data, responseId);
                        } catch (e) {
                            console.error('JSON parse error', e);
                        }
                    }
                }
            }

        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Fetch aborted');
            } else {
                console.error('Fetch error:', error);
                appendSystemMessage(`Error: ${error.message}`);
            }
        } finally {
            finishGeneration(responseId);
            currentController = null;
        }
    }

    function startGeneration() {
        isGenerating = true;
        userInput.disabled = true;
        sendBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function finishGeneration(id) {
        isGenerating = false;
        userInput.disabled = false;
        userInput.focus();
        sendBtn.classList.remove('hidden');
        stopBtn.classList.add('hidden');
        newChatBtn.disabled = false; // Re-enable new chat button

        // Hide loading indicator if id provided
        if (id) {
            const loadingDiv = document.getElementById(`${id}-loading`);
            if (loadingDiv) {
                loadingDiv.remove();
            }
        }

        // Save to history (if we have accumulated text)
        if (currentAIResponseText) {
            conversationHistory.push({ role: 'assistant', content: currentAIResponseText });
        }
    }

    function appendUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'flex justify-end message-enter';
        div.innerHTML = `
            <div class="max-w-[85%] bg-brand-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 shadow-sm text-sm">
                ${escapeHtml(text)}
            </div>
        `;
        chatContainer.appendChild(div);
        scrollToBottom();
    }

    function appendSystemMessage(text) {
        const div = document.createElement('div');
        div.className = 'flex justify-center my-4 message-enter';
        div.innerHTML = `
            <div class="bg-gray-100 text-gray-500 rounded-full px-4 py-1 text-xs">
                ${escapeHtml(text)}
            </div>
        `;
        chatContainer.appendChild(div);
        scrollToBottom();
    }

    // State for tracking tools and content
    let agentState = {
        currentAgentId: null,
        agents: {} // agentId -> { name, tools: { id -> { name, content, input, output } } }
    };

    // We need to mirror the logic from _update_state_with_event in main.py
    // roughly, but since we are receiving granular events, we can update the DOM incrementally.
    // However, it's easier to maintain a state object and re-render the active message 
    // or append chunks to the active text node.

    function createAgentMessageBubbles(id) {
        const div = document.createElement('div');
        div.id = id;
        div.className = 'flex justify-start gap-3 message-enter w-full';
        div.innerHTML = `
            <div class="w-8 h-8 rounded-full bg-gradient-to-br from-brand-500 to-indigo-600 flex-shrink-0 flex items-center justify-center text-white text-xs font-bold shadow-sm mt-1">
                AI
            </div>
            <div class="flex-1 max-w-3xl space-y-2">
                <div id="${id}-content" class="bg-white border border-gray-100 rounded-2xl rounded-tl-sm px-6 py-5 shadow-sm text-sm markdown-body min-h-[60px]">
                    <div id="${id}-thinking" class="hidden"></div>
                    <div id="${id}-loading" class="flex gap-1 items-center text-gray-400 text-xs py-2">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            </div>
        `;
        chatContainer.appendChild(div);
        scrollToBottom();

        // Reset state for new response
        agentState = {
            currentAgentId: null,
            agents: {}
        };
    }

    function handleStreamEvent(message, responseId) {
        const contentDiv = document.getElementById(`${responseId}-content`);
        const loadingDiv = document.getElementById(`${responseId}-loading`);

        if (!contentDiv) return;

        const event = message.event;
        const data = message.data || {};

        if (event === 'start_of_agent') {
            const agentId = data.agent_id;
            const agentName = data.agent_name || 'unknown';
            agentState.agents[agentId] = { name: agentName, tools: {} };
            agentState.currentAgentId = agentId;
            loadingDiv.style.display = 'flex'; // Show loading when agent starts
        }
        else if (event === 'end_of_agent') {
            agentState.currentAgentId = null;
        }
        else if (event === 'agent_thought') {
            // Handle explicit thinking events
            const thoughtContent = data.thought || data.content;
            if (thoughtContent) {
                const thinkingDiv = document.getElementById(`${responseId}-thinking`);
                if (thinkingDiv) {
                    if (thinkingDiv.classList.contains('hidden')) {
                        thinkingDiv.classList.remove('hidden');
                        thinkingDiv.className = 'bg-gray-50 border border-gray-100 rounded-lg p-3 mb-4 text-xs text-gray-500 font-mono whitespace-pre-wrap';
                        thinkingDiv.innerHTML = '<div class="font-bold mb-1 text-gray-400">🤔 Thinking Process:</div>';
                    }
                    // Append thought content
                    const thoughtSpan = document.createElement('span');
                    thoughtSpan.textContent = thoughtContent + "\n";
                    thinkingDiv.appendChild(thoughtSpan);
                    scrollToBottom();
                }
            }
        }
        else if (event === 'tool_call') {
            const agentId = agentState.currentAgentId;
            if (!agentId) return;

            const toolCallId = data.tool_call_id;
            const toolName = data.tool_name;
            const agent = agentState.agents[agentId];

            if (!agent.tools[toolCallId]) {
                agent.tools[toolCallId] = {
                    name: toolName,
                    content: '',
                    input: null,
                    output: null,
                    rendered: false
                };
            }

            const toolEntry = agent.tools[toolCallId];

            // Handle show_text (streaming text response)
            if (toolName === 'show_text' || toolName === 'message') {
                let delta = '';
                if (data.delta_input && data.delta_input.text) {
                    delta = data.delta_input.text;
                } else if (data.tool_input && data.tool_input.text) {
                    delta = data.tool_input.text;
                }

                // Format think tags
                if (delta) {
                    // Check for <think> tags
                    // We will simple append raw text for now and let markdown parser handle it?
                    // Or we can parse it out.
                    // Let's just append to content and re-render markdown of that specific block?
                    // Better: accumulate valid markdown text.

                    toolEntry.content += delta;
                    currentAIResponseText += delta; // Accumulate for history

                    // Update main content area
                    // We treat the whole chat response as one markdown block for simplicity of "show_text"
                    // Tools are inserted as separate blocks.

                    // Actually, let's keep it simple: 
                    // If it's show_text, we append to a "text-accumulator".
                    // If it's a tool, we append a "tool-card".
                    // But order matters.

                    // Simplification: We will just append HTML to the contentDiv immediately
                    // instead of complex state tracking recalculation.
                }
            } else {
                // Other tools
                // We update input/output in the state
                if (data.tool_input) toolEntry.input = data.tool_input;
                if (data.tool_output) toolEntry.output = data.tool_output;
            }

            renderUpdate(responseId, toolName, toolEntry, data);
        }
    }

    function renderUpdate(responseId, toolName, toolEntry, data) {
        const contentDiv = document.getElementById(`${responseId}-content`);
        const loadingDiv = document.getElementById(`${responseId}-loading`);

        // Remove loading div temporarily to append at end, then add back
        if (loadingDiv.parentNode === contentDiv) {
            contentDiv.removeChild(loadingDiv);
        }

        if (toolName === 'show_text' || toolName === 'message') {
            // Find or create the current text block
            // We need a stable ID for the current streaming text block
            let textBlock = document.getElementById(`${responseId}-text-current`);
            if (!textBlock) {
                textBlock = document.createElement('div');
                textBlock.id = `${responseId}-text-current`;
                textBlock.className = 'markdown-content';
                contentDiv.appendChild(textBlock);
            }

            // Re-render the markdown for this block
            // Handle <think> tags specially
            let rawContent = toolEntry.content;

            // Basic think tag handling
            rawContent = rawContent.replace(/<think>/g, '\n> **Thinking:**\n> ');
            rawContent = rawContent.replace(/<\/think>/g, '\n\n');

            textBlock.innerHTML = marked.parse(rawContent);

            // Highlight code blocks
            textBlock.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });

        } else {
            // It's a tool (google_search, scrape, etc)
            // If it's the first time we see this tool call, create a container
            // We use toolEntry in state to only create once.

            // We need to identify if we should update an existing card or create new
            // Since we don't have unique stable IDs for DOM elements mapped to tool_id easily without clutter,
            // we'll try to find it.

            const toolDomId = `${responseId}-tool-${data.tool_call_id}`;
            let toolDiv = document.getElementById(toolDomId);

            if (!toolDiv) {
                toolDiv = document.createElement('div');
                toolDiv.id = toolDomId;
                contentDiv.appendChild(toolDiv);

                // If we switched from text to tool, the next text should be a new block
                const currentText = document.getElementById(`${responseId}-text-current`);
                if (currentText) currentText.removeAttribute('id'); // Retire current text block
            }

            // Render the tool card based on current input/output state
            toolDiv.innerHTML = renderToolCard(toolName, toolEntry.input, toolEntry.output);
        }

        // Add loading back at bottom
        contentDiv.appendChild(loadingDiv);
        scrollToBottom();
    }

    function renderToolCard(name, input, output) {
        // Helper to check if empty
        const isEmpty = (obj) => !obj || (typeof obj === 'object' && Object.keys(obj).length === 0);
        const hasInput = !isEmpty(input);
        const hasOutput = !isEmpty(output);

        if (!hasInput && !hasOutput) return '';

        // Google Search Special
        if (name === 'google_search') {
            const query = input ? (input.q || input.query || '') : '';
            let results = [];

            if (output) {
                if (output.organic) results = output.organic;
                else if (output.result) {
                    try {
                        const parsed = JSON.parse(output.result);
                        if (parsed.organic) results = parsed.organic;
                    } catch (e) { }
                }
            }

            if (!query && results.length === 0) return '';

            let html = `<div class="search-card">`;
            if (query) {
                html += `
                    <div class="search-header">
                        <span class="mr-2">🔍</span> Google Search: "${escapeHtml(query)}"
                    </div>
                `;
            }

            if (results.length > 0) {
                html += `<div class="search-count">≡ Found ${results.length} results</div>`;
                html += `<div class="search-results">`;
                results.slice(0, 50).forEach(item => {
                    html += `
                        <a href="${item.link}" target="_blank" class="search-result-item">
                            <span class="mr-2">🌐</span>
                            <span class="truncate">${escapeHtml(item.title)}</span>
                        </a>
                     `;
                });
                html += `</div>`;
            } else if (hasInput && !hasOutput) {
                html += `<div class="p-2 text-xs text-gray-400 italic">Searching...</div>`;
            }
            html += `</div>`;
            return html;
        }

        // Sogou Search Special
        if (name === 'sogou_search') {
            const query = input ? (input.q || input.query || '') : '';
            let results = [];

            if (output) {
                // Handle various potential output formats (parsed object or JSON string)
                let parsedOutput = output;
                if (typeof output === 'string') {
                    try { parsedOutput = JSON.parse(output); } catch (e) { }
                }

                if (parsedOutput.Pages) {
                    results = parsedOutput.Pages;
                } else if (parsedOutput.result) {
                    // Sometimes wrapped in result
                    try {
                        const nested = JSON.parse(parsedOutput.result);
                        if (nested.Pages) results = nested.Pages;
                    } catch (e) { }
                }
            }

            if (!query && results.length === 0) return '';

            let html = `<div class="search-card">`;
            if (query) {
                html += `
                    <div class="search-header">
                        <span class="mr-2">🔍</span> Sogou Search: "${escapeHtml(query)}"
                    </div>
                `;
            }

            if (results.length > 0) {
                html += `<div class="search-count">≡ Found ${results.length} results</div>`;
                html += `<div class="search-results">`;
                results.forEach(item => {
                    html += `
                        <a href="${item.url}" target="_blank" class="search-result-item">
                            <span class="mr-2">🌐</span>
                            <span class="truncate">${escapeHtml(item.title)}</span>
                        </a>
                     `;
                });
                html += `</div>`;
            } else if (hasInput && !hasOutput) {
                html += `<div class="p-2 text-xs text-gray-400 italic">Searching...</div>`;
            }
            html += `</div>`;
            return html;
        }

        // Scrape Special
        if (name.includes('scrape') || name.includes('reading')) {
            let url = '';
            if (input) {
                url = input.url || input.link || input.website || input.q || '';
            }

            // If URL is still empty but input is a string, maybe it's just the URL?
            if (!url && typeof input === 'string') {
                url = input;
            }

            const isError = output && output.error;
            const isDone = hasOutput && !isError;

            // If we still can't find a URL, show generic "Web Content"
            const displayUrl = url ? url : 'Web Content';

            return `
                <div class="scrape-card ${isError ? 'scrape-error' : ''}">
                    <div class="flex items-center overflow-hidden">
                        <span class="scrape-icon mr-2">🌐</span>
                        <a href="${escapeHtml(url)}" target="_blank" class="scrape-url truncate max-w-[300px] hover:underline cursor-pointer text-blue-600 block">
                            ${escapeHtml(displayUrl)}
                        </a>
                    </div>
                    <div class="scrape-status ${isError ? 'error' : (isDone ? 'success' : 'text-gray-400')}">
                        ${isError ? '❌ Failed' : (isDone ? '✓ Scraped' : 'Reading...')}
                    </div>
                </div>
            `;
        }

        // Generic Tool
        let inputStr = '';
        if (hasInput) {
            inputStr = Object.entries(input).map(([k, v]) => `${k}: ${String(v).slice(0, 30)}`).join(', ');
        }

        return `
            <div class="tool-card">
                <div class="tool-header">
                    <span>🔧 ${name}</span>
                </div>
                ${inputStr ? `<div class="tool-brief">${escapeHtml(inputStr)}</div>` : ''}
                ${hasOutput ? '<div class="tool-status">✓ Done</div>' : ''}
            </div>
        `;
    }

    function escapeHtml(text) {
        if (typeof text !== 'string') return text;
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});
