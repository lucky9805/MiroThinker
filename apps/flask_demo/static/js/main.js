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

    // Polyfill or simple UUID generator for non-secure contexts
    function generateUUID() {
        if (typeof crypto !== 'undefined' && crypto.randomUUID) {
            return crypto.randomUUID();
        }
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
            var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    let currentSessionId = generateUUID(); // Unique ID for current chat

    // Load initial history list
    loadSessionList();

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
        currentSessionId = generateUUID(); // New ID for new chat
        conversationHistory = [];
        agentState = { currentAgentId: null, agents: {} }; // Reset agent state too

        chatContainer.innerHTML = '';
        chatContainer.appendChild(welcomeMessage);
        welcomeMessage.style.display = 'flex';
        userInput.value = '';
        userInput.style.height = '48px';

        // Refresh list to remove active highlight if any
        loadSessionList();
    });

    // Load Session List function
    async function loadSessionList() {
        try {
            const res = await fetch('/api/sessions');
            const sessions = await res.json();

            const historyContainer = document.getElementById('history-list');
            if (!historyContainer) return; // Must be added to HTML first

            historyContainer.innerHTML = '';

            sessions.forEach(session => {
                const item = document.createElement('div');
                item.className = `p-3 rounded-lg cursor-pointer hover:bg-gray-50 group flex items-center justify-between transition-colors ${session.id === currentSessionId ? 'bg-gray-50' : ''}`;
                item.onclick = () => loadSession(session.id);

                item.innerHTML = `
                    <div class="flex-1 min-w-0 pr-2">
                        <div class="text-sm font-medium text-gray-700 truncate" title="${escapeHtml(session.title)}">${escapeHtml(session.title)}</div>
                        <div class="text-xs text-gray-400 mt-0.5">${new Date(session.timestamp).toLocaleDateString()}</div>
                    </div>
                `;
                // Add delete button
                const delBtn = document.createElement('button');
                delBtn.className = 'opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 transition-opacity';
                delBtn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                `;
                delBtn.onclick = (e) => {
                    e.stopPropagation();
                    if (confirm('Delete this chat?')) deleteSession(session.id);
                };
                item.appendChild(delBtn);

                historyContainer.appendChild(item);
            });
        } catch (e) {
            console.error("Failed to load history:", e);
        }
    }

    async function loadSession(id) {
        if (isGenerating) return;
        try {
            const res = await fetch(`/api/sessions/${id}`);
            if (!res.ok) return;
            const data = await res.json();

            currentSessionId = id;
            conversationHistory = data.history || [];

            // Clear UI and Re-render
            chatContainer.innerHTML = '';
            welcomeMessage.style.display = 'none';

            // Re-play history
            conversationHistory.forEach(msg => {
                if (msg.role === 'user') appendUserMessage(msg.content);
                else if (msg.role === 'assistant') {
                    if (msg.thinking_data) {
                        restoreFullAIMessage(msg);
                    } else {
                        appendSimpleAIMessage(msg.content);
                    }
                }
            });

            loadSessionList(); // Update active state
            scrollToBottom();

        } catch (e) {
            console.error("Failed to load session:", e);
        }
    }

    async function saveSession() {
        if (!conversationHistory.length) return;

        // Generate title if needed (first user message)
        let title = "New Chat";
        const firstUserMsg = conversationHistory.find(m => m.role === 'user');
        if (firstUserMsg) {
            title = firstUserMsg.content.slice(0, 30) + (firstUserMsg.content.length > 30 ? "..." : "");
        }

        const data = {
            id: currentSessionId,
            title: title,
            timestamp: Date.now(),
            history: conversationHistory
        };

        try {
            await fetch(`/api/sessions/${currentSessionId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            loadSessionList(); // Refresh list to show new/updated chat
        } catch (e) {
            console.error("Failed to save session:", e);
        }
    }

    async function deleteSession(id) {
        try {
            await fetch(`/api/sessions/${id}`, { method: 'DELETE' });
            if (currentSessionId === id) {
                newChatBtn.click(); // Reset if deleted active
            } else {
                loadSessionList();
            }
        } catch (e) {
            console.error("Failed to delete session:", e);
        }
    }

    function appendSimpleAIMessage(markdownText) {
        // Re-use createAgentMessageBubbles style but simpler
        const id = 'hist-' + Math.random().toString(36).substr(2, 9);
        createAgentMessageBubbles(id);
        const contentDiv = document.getElementById(`${id}-content`);
        const loadingDiv = document.getElementById(`${id}-loading`);
        if (loadingDiv) loadingDiv.remove();

        if (contentDiv) {
            contentDiv.innerHTML = marked.parse(markdownText || "");
            contentDiv.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }
    }

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
        saveSession(); // Save immediately with user msg

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

            // Auto-collapse thinking process on finish
            const thinkBody = document.getElementById(`${id}-thinking-body`);
            const thinkHeaderIcon = document.querySelector(`#${id}-thinking-container .transform`);
            if (thinkBody) {
                thinkBody.style.display = 'none'; // Collapse
                if (thinkHeaderIcon) {
                    thinkHeaderIcon.classList.remove('rotate-180');
                }

                // Optional: Update header text to show Done?
                // const thinkTitle = document.querySelector(`#${id}-thinking-container span.font-medium`);
                // if (thinkTitle) thinkTitle.textContent = "Thinking Process (Completed)";
            }
        }

        // Save to history (if we have accumulated text)
        if (currentAIResponseText) {
            // Include thinking data
            const historyEntry = {
                role: 'assistant',
                content: currentAIResponseText,
                thinking_data: {
                    log: agentState ? agentState.log : []
                }
            };
            conversationHistory.push(historyEntry);
            saveSession(); // Save with AI response
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
            agents: {},
            thoughts: [], // Track thinking steps for history
            tools: []     // Track tools linearly for history
        };
    }

    function restoreFullAIMessage(msg) {
        const id = 'hist-' + Math.random().toString(36).substr(2, 9);
        createAgentMessageBubbles(id);
        const loadingDiv = document.getElementById(`${id}-loading`);
        if (loadingDiv) loadingDiv.remove();

        // 1. Restore Thinking Process
        if (msg.thinking_data) {
            // We need to replay events roughly. 
            // Or just render the list of items (thoughts and tools) in order if we saved them linearly.
            // If we saved them separately, we just render all thoughts then all tools? No, order matters.

            // If we have a linear log:
            if (msg.thinking_data.log) {
                msg.thinking_data.log.forEach(item => {
                    if (item.type === 'thought') {
                        updateThinkingProcess(id, item.content, true);
                    } else if (item.type === 'tool') {
                        // Create a fake tool entry to render
                        // We need to inject it into thinking body
                        if (!document.getElementById(`${id}-thinking-container`)) {
                            updateThinkingProcess(id, "", true);
                        }
                        const thinkingBody = document.getElementById(`${id}-thinking-body`);
                        const toolDiv = document.createElement('div');
                        toolDiv.className = 'mb-2';
                        toolDiv.innerHTML = renderToolCard(item.name, item.input, item.output);
                        thinkingBody.appendChild(toolDiv);
                    }
                });
            } else {
                // Fallback for older saves? or if we stick to split arrays
                if (msg.thinking_data.thoughts) {
                    msg.thinking_data.thoughts.forEach(t => updateThinkingProcess(id, t, true));
                }
                // Tools might be lost or separate.
            }

            // Auto collapse
            const thinkBody = document.getElementById(`${id}-thinking-body`);
            const thinkIcon = document.getElementById(`${id}-thinking-icon`);
            if (thinkBody) {
                thinkBody.style.display = 'none';
                if (thinkIcon) thinkIcon.classList.remove('rotate-180');
            }
        }

        // 2. Restore Final Content
        const contentDiv = document.getElementById(`${id}-content`);
        // We need to make sure we append AFTER thinking container
        // current text block logic handles this?
        // simple innerHTML might kill thinking container if we are not careful.
        // appendSimpleAIMessage used innerHTML = marked.parse...
        // We should append a text block.

        let textBlock = document.createElement('div');
        textBlock.className = 'markdown-content';
        textBlock.innerHTML = marked.parse(msg.content || "");
        contentDiv.appendChild(textBlock);

        contentDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
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
                updateThinkingProcess(responseId, thoughtContent, true);
            }
        }
        else if (event === 'tool_call') {
            const agentId = agentState.currentAgentId;
            if (!agentId) return;

            const toolCallId = data.tool_call_id;
            let toolName = data.tool_name;
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
            if (!toolName && toolEntry) toolName = toolEntry.name;

            // Handle show_text (streaming text response)
            if (toolName === 'show_text' || toolName === 'message') {
                let delta = '';
                if (data.delta_input && data.delta_input.text) {
                    delta = data.delta_input.text;
                } else if (data.tool_input && data.tool_input.text) {
                    delta = data.tool_input.text;
                }

                if (delta) {
                    toolEntry.content += delta;
                    currentAIResponseText += delta; // Accumulate for history
                }
            } else {
                // Other tools
                // We update input/output in the state

                if (!toolEntry.input) toolEntry.input = {};

                if (data.tool_input) {
                    // 1. Preserve critical fields from current state
                    const preserved = {
                        code: toolEntry.input.code,
                        code_block: toolEntry.input.code_block,
                        command: toolEntry.input.command
                    };

                    // 2. Merge new data (potentially overwriting with empty junk)
                    Object.assign(toolEntry.input, data.tool_input);

                    // 3. Restore critical fields IF they became empty/missing AND were valid before
                    if (preserved.code && !toolEntry.input.code) toolEntry.input.code = preserved.code;
                    if (preserved.code_block && !toolEntry.input.code_block) toolEntry.input.code_block = preserved.code_block;
                    if (preserved.command && !toolEntry.input.command) toolEntry.input.command = preserved.command;
                }

                if (data.tool_output) toolEntry.output = data.tool_output;
            }

            renderUpdate(responseId, toolName, toolEntry, data);
        }
    }

    function updateThinkingProcess(responseId, content, append = true) {
        // ... (DOM creation logic) ...
        // Ensure agentState.log exists if we use it
        if (agentState && !agentState.log) agentState.log = [];

        // Log the thought
        // Check if last log item is a thought, append to it to avoid fragmentation?
        if (agentState.log) {
            const lastLog = agentState.log[agentState.log.length - 1];
            if (lastLog && lastLog.type === 'thought' && append) {
                lastLog.content += content;
            } else {
                agentState.log.push({ type: 'thought', content: content });
            }
        }

        // ... (Rest of existing DOM logic) ...
        const contentDiv = document.getElementById(`${responseId}-content`);
        if (!contentDiv) return;

        let thinkingContainer = document.getElementById(`${responseId}-thinking-container`);
        if (!thinkingContainer) {
            thinkingContainer = document.createElement('div');
            thinkingContainer.id = `${responseId}-thinking-container`;
            thinkingContainer.className = 'mb-4 border border-gray-200 rounded-lg overflow-hidden';

            // Header
            const header = document.createElement('div');
            header.className = 'bg-gray-50 px-3 py-2 text-xs font-medium text-gray-500 cursor-pointer flex items-center justify-between hover:bg-gray-100 transition-colors select-none';
            header.innerHTML = `
                <div class="flex items-center gap-2">
                    <span>🤔 Thinking Process</span>
                    <span id="${responseId}-thinking-status" class="animate-pulse text-blue-500 hidden">●</span>
                </div>
                <svg id="${responseId}-thinking-icon" class="w-4 h-4 transform rotate-180 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            `;
            header.onclick = () => {
                const body = document.getElementById(`${responseId}-thinking-body`);
                const icon = document.getElementById(`${responseId}-thinking-icon`);
                if (body.style.display === 'none') {
                    body.style.display = 'block';
                    icon.classList.add('rotate-180');
                } else {
                    body.style.display = 'none';
                    icon.classList.remove('rotate-180');
                }
            };
            thinkingContainer.appendChild(header);

            // Body
            const body = document.createElement('div');
            body.id = `${responseId}-thinking-body`;
            body.className = 'bg-gray-50/50 p-3 text-xs text-gray-600 font-mono whitespace-pre-wrap border-t border-gray-200';
            body.style.display = 'block'; // Default open
            thinkingContainer.appendChild(body);

            // Insert at top of content div
            if (contentDiv.firstChild) {
                contentDiv.insertBefore(thinkingContainer, contentDiv.firstChild);
            } else {
                contentDiv.appendChild(thinkingContainer);
            }
        }

        const body = document.getElementById(`${responseId}-thinking-body`);
        const statusDot = document.getElementById(`${responseId}-thinking-status`);

        // Show status dot if active
        if (statusDot) statusDot.classList.remove('hidden');

        // Append logic: Find the LAST element. If it's a text container, append. Else create new.
        const lastEl = body.lastElementChild;
        let textContainer;

        if (lastEl && lastEl.classList.contains('thinking-text')) {
            textContainer = lastEl;
        } else {
            textContainer = document.createElement('div');
            textContainer.className = 'thinking-text mb-2'; // Add spacing between blocks
            body.appendChild(textContainer);
        }

        if (append) {
            textContainer.textContent += content;
        } else {
            textContainer.textContent = content;
        }

        scrollToBottom();
    }

    function renderUpdate(responseId, toolName, toolEntry, data) {
        const contentDiv = document.getElementById(`${responseId}-content`);
        const loadingDiv = document.getElementById(`${responseId}-loading`);

        // Remove loading div temporarily
        if (loadingDiv && loadingDiv.parentNode === contentDiv) {
            contentDiv.removeChild(loadingDiv);
        }

        if (toolName === 'show_text' || toolName === 'message') {
            // Find or create the current text block
            let textBlock = document.getElementById(`${responseId}-text-current`);
            if (!textBlock) {
                textBlock = document.createElement('div');
                textBlock.id = `${responseId}-text-current`;
                textBlock.className = 'markdown-content';
                contentDiv.appendChild(textBlock);
            }

            // Extract <think> content
            let rawContent = toolEntry.content;
            const thinkPattern = /<think>([\s\S]*?)<\/think>/g;
            let displayContent = rawContent;
            let foundThinking = false;
            let accumulatedThought = "";

            displayContent = displayContent.replace(thinkPattern, (match, content) => {
                accumulatedThought += content;
                foundThinking = true;
                return "";
            });

            // Handle partial <think>
            const partialThink = displayContent.match(/<think>([\s\S]*)$/);
            if (partialThink) {
                accumulatedThought += partialThink[1];
                foundThinking = true;
                displayContent = displayContent.substring(0, partialThink.index);
            }

            if (foundThinking) {
                // Use updateThinkingProcess to correctly append interleaved thoughts
                // We use append=false if we want to replace, but here we are re-parsing the whole chunk?
                // Actually, since this runs every chunk on full content, we might duplicate text if we just append?

                // CRITICAL CORRECTION: If we are extracting from full content stream, we shouldn't use "append += content" naively.
                // We need to manage the text block.
                // But updateThinkingProcess is designed for 'agent_thought' events which are incremental segments.
                // Here we have the FULL thought string extracted from full text.

                // Let's create a specialized updater for extracted thought that Replaces the content of the "Current" thought block.
                updateExtractThinking(responseId, accumulatedThought);
            }

            textBlock.innerHTML = marked.parse(displayContent);

            // Highlight
            textBlock.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });

        } else {
            // TOOL CARDS -> MOVE TO THINKING BODY

            // Ensure thinking container exists
            // calling updateThinkingProcess with empty content ensures creation
            if (!document.getElementById(`${responseId}-thinking-container`)) {
                updateThinkingProcess(responseId, "", true);
            }

            const thinkingBody = document.getElementById(`${responseId}-thinking-body`);

            // Identify tool div
            const toolDomId = `${responseId}-tool-${data.tool_call_id}`;
            let toolDiv = document.getElementById(toolDomId);

            if (!toolDiv) {
                toolDiv = document.createElement('div');
                toolDiv.id = toolDomId;
                toolDiv.className = 'mb-2';
                thinkingBody.appendChild(toolDiv); // Append to THINKING BODY

                // Log the tool call in sequence
                if (agentState && !agentState.log) agentState.log = [];
                // Check if already in log (by reference or ID?)
                // Since this runs only when creating DIV, it runs once per tool usually.
                // We add a reference to the toolEntry so updates reflect in the log automatically.
                if (agentState.log) {
                    toolEntry.type = 'tool'; // Mark as tool
                    agentState.log.push(toolEntry);
                }
            }

            toolDiv.innerHTML = renderToolCard(toolName, toolEntry.input, toolEntry.output);
        }

        // Add loading back at bottom
        if (loadingDiv) contentDiv.appendChild(loadingDiv);
        scrollToBottom();
    }

    function updateExtractThinking(responseId, content) {
        // Helper to update specific extracted thought part only
        const contentDiv = document.getElementById(`${responseId}-content`);
        if (!contentDiv) return;

        // Ensure container exists (reuse logic or call updateThinkingProcess with empty)
        if (!document.getElementById(`${responseId}-thinking-container`)) {
            updateThinkingProcess(responseId, "", true);
        }

        const body = document.getElementById(`${responseId}-thinking-body`);

        // Find the "extracted" container or create one at the end if it doesn't exist?
        // To avoid duplication during streaming re-render, we need a stable ID for THIS thought block.
        // We can attach it to the text-current?
        // Simplified: Just use one 'extracted-thought' block at the end.

        let extractedContainer = body.querySelector('.extracted-thinking-current');
        if (!extractedContainer) {
            extractedContainer = document.createElement('div');
            extractedContainer.className = 'thinking-text extracted-thinking-current mb-2';
            body.appendChild(extractedContainer);
        }
        extractedContainer.textContent = content;
    }

    function renderToolCard(name, input, output) {
        // Helper to check if empty
        const isEmpty = (obj) => !obj || (typeof obj === 'object' && Object.keys(obj).length === 0);
        const hasInput = !isEmpty(input);
        const hasOutput = !isEmpty(output);

        if (!hasInput && !hasOutput) return '';




        if (name.toLowerCase().includes('search')) {
            // Fallback: If output is missing but input has result (legacy/history mode), use it
            if ((!output || output === "") && input && input.result) {
                output = input.result;
            }

            const effectivelyHasOutput = output !== null && output !== undefined && output !== '';
            const query = input ? (input.q || input.query || input.queries || '') : '';
            let results = [];

            if (output) {
                let raw = output;
                // If output itself wraps result
                if (output.result) raw = output.result;
                else if (output.organic) raw = output.organic;

                if (typeof raw === 'string') {
                    try {
                        const parsed = JSON.parse(raw);
                        if (Array.isArray(parsed)) results = parsed;
                        else if (parsed.organic) results = parsed.organic;
                        else if (parsed.results) results = parsed.results;
                        else if (parsed.Pages) results = parsed.Pages; // Sogou
                    } catch (e) {
                        // If not JSON, treat as single text result if not empty
                        if (raw.trim().length > 0) results = [{ title: "Result", snippet: raw, link: "" }];
                    }
                } else if (Array.isArray(raw)) {
                    results = raw;
                } else if (typeof raw === 'object') {
                    // Handle object that contains results list
                    if (raw.organic && Array.isArray(raw.organic)) results = raw.organic;
                    else if (raw.results && Array.isArray(raw.results)) results = raw.results;
                    else if (raw.Pages && Array.isArray(raw.Pages)) results = raw.Pages; // Sogou
                }
            }

            let html = `<div class="bg-white rounded-md border border-gray-200 overflow-hidden my-2 tool-card search-card">`;

            // Header
            html += `
                <div class="bg-gray-50 px-3 py-2 border-b border-gray-100 flex items-center justify-between">
                    <div class="flex items-center truncate">
                        <span class="mr-2 text-base">🔍</span> 
                        <span class="font-medium text-gray-700 text-sm truncate max-w-[200px]" title="${escapeHtml(query)}">${escapeHtml(query)}</span>
                    </div>
                    <div class="text-xs text-gray-400 whitespace-nowrap ml-2">
                        ${effectivelyHasOutput ? results.length + ' results' : 'Searching...'}
                    </div>
                </div>
            `;

            if (results.length > 0) {
                html += `<div class="divide-y divide-gray-100 max-h-60 overflow-y-auto">`;
                results.forEach(item => {
                    const title = item.title || item.name || "Untitled";
                    const link = item.link || item.url || "#";

                    html += `
                        <div class="px-3 py-1.5 hover:bg-blue-50 transition-colors duration-150 group">
                            <a href="${escapeHtml(link)}" target="_blank" class="flex items-center text-sm text-blue-600 hover:text-blue-800 hover:underline truncate">
                                <span class="mr-2 text-xs opacity-70">🌐</span>
                                <span class="truncate" title="${title}">${escapeHtml(title)}</span>
                            </a>
                        </div>
                     `;
                });
                html += `</div>`;
            } else if (effectivelyHasOutput) {
                html += `<div class="p-3 text-xs text-gray-500 italic">No structure parsed, see raw output below.</div>`;
            } else {
                html += `<div class="p-3 text-xs text-gray-400 italic">Searching...</div>`;
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

        // Python Tool Special
        if (name === 'tool-python' || name === 'run_python_code' || name === 'python_interpreter' || name === 'create_sandbox' || name === 'run_command') {

            // Handle create_sandbox specially
            if (name === 'create_sandbox') {
                return `
                        < div class="tool-card" >
                        <div class="tool-header">
                            <span>📦 Create Sandbox</span>
                        </div>
                        <div class="tool-status text-xs text-gray-500 mt-1">
                            ${hasOutput ? (output.result || 'Sandbox Ready') : 'Initializing environment...'}
                        </div>
                    </div >
                        `;
            }

            let code = '';
            if (input && input.code) {
                code = input.code;
            } else if (input && input.code_block) {
                code = input.code_block;
            } else if (input && input.command) {
                code = input.command; // For run_command
            } else if (typeof input === 'string') {
                code = input;
            }

            const isError = output && output.error;
            const isDone = hasOutput;

            let outputDisplay = '';
            if (output && output.result) {
                outputDisplay = output.result;
            } else if (output && output.stdout) {
                outputDisplay = output.stdout;
            } else if (output && typeof output === 'string') {
                outputDisplay = output;
            }

            const lang = name === 'run_command' ? 'bash' : 'python';
            const title = name === 'run_command' ? '💻 Shell Command' : '🐍 Python Code';

            return `
                        < div class="tool-card python-card" >
                    <div class="tool-header flex justify-between items-center mb-2">
                        <span class="font-bold text-gray-700">${title}</span>
                        <span class="text-xs ${isError ? 'text-red-500' : 'text-gray-400'}">${isError ? 'Error' : (isDone ? 'Executed' : 'Running...')}</span>
                    </div>
                    <div class="bg-gray-800 rounded-md p-3 overflow-x-auto mb-2">
                        <pre><code class="language-${lang} text-xs text-gray-100">${escapeHtml(code)}</code></pre>
                    </div>
                    ${outputDisplay ? `
                    <div class="mt-2 border-t border-gray-200 pt-2">
                        <div class="text-xs font-semibold text-gray-500 mb-1">Output:</div>
                        <pre class="bg-gray-50 p-2 rounded text-xs text-gray-700 whitespace-pre-wrap font-mono max-h-60 overflow-y-auto">${escapeHtml(outputDisplay)}</pre>
                    </div>
                    ` : ''
                }
                </div >
                        `;
        }

        // Generic Tool
        return `
                        < div class="tool-card custom-tool-card" >
                 <div class="tool-header mb-2 font-bold text-gray-700">🔧 ${escapeHtml(name)}</div>
                 <div class="text-xs bg-gray-50 p-2 rounded border border-gray-100 font-mono">
                    <div class="text-gray-500 mb-1">Input:</div>
                    <div class="mb-2 whitespace-pre-wrap">${escapeHtml(JSON.stringify(input, null, 2))}</div>
                    ${hasOutput ? `
                    <div class="text-gray-500 mb-1 border-t border-gray-100 pt-2">Output:</div>
                    <div class="whitespace-pre-wrap">${escapeHtml(typeof output === 'string' ? output : JSON.stringify(output, null, 2))}</div>
                    ` : ''}
                 </div>
            </div >
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
