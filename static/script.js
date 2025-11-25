


'use strict';
// --- State and Constants ---
if (!window.appState) {
    window.appState = {
        ws: null,
        isGenerating: false,
        currentMessageContainer: null,
        fullResponseText: '',
        activeChatId: 'default_chat',
        isInitialized: false,
        isStreamInitialized: false,
        awaitingAssistantTimestamp: false
    };
}
const SETTINGS_KEY = 'aiRoleplaySettings_v3';
const DOM = {
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    sendBtn: document.getElementById('send-btn'),
    stopBtn: document.getElementById('stop-btn'),
    statusText: document.getElementById('status-text'),
    statusDot: document.getElementById('status-dot'),
    chatTitle: document.getElementById('chat-title'),
    leftPanel: document.getElementById('left-panel'),
    rightPanel: document.getElementById('right-panel'),
    leftPanelToggle: document.getElementById('left-panel-toggle'),
    rightPanelToggle: document.getElementById('right-panel-toggle'),
    mobileMenuLeft: document.getElementById('mobile-menu-left'),
    mobileMenuRight: document.getElementById('mobile-menu-right'),
    appContainer: document.getElementById('app-container'),
    chatList: document.getElementById('chat-list'),
    newChatName: document.getElementById('new-chat-name'),
    addChatBtn: document.getElementById('add-chat-btn'),
    reloadChatBtn: document.getElementById('reload-chat'),
    newDayBtn: document.getElementById('new-day'),
    checkSummaryBtn: document.getElementById('check-summary'),
    forceSummaryBtn: document.getElementById('force-summary-btn'),
    clearMemoryBtn: document.getElementById('clear-memory'),
    modelSelect: document.getElementById('model-select'),
    embeddingModelSelect: document.getElementById('embedding-model-select'),
    persistentStatsToggle: document.getElementById('persistent-stats-toggle'),
    enableMemoryToggle: document.getElementById('enable-memory-toggle'),
    tempSlider: document.getElementById('temperature-slider'),
    tempValue: document.getElementById('temp-value'),
    tokensSlider: document.getElementById('tokens-slider'),
    tokensValue: document.getElementById('tokens-value'),
    thoughtSlider: document.getElementById('thought-ratio-slider'),
    thoughtValue: document.getElementById('thought-ratio-value'),
    talkSlider: document.getElementById('talkativeness-slider'),
    talkValue: document.getElementById('talkativeness-value'),
    personaAvatar: document.getElementById('persona-avatar'),
    sidePanelPersonaPreset: document.getElementById('side-panel-persona-preset'),
    sidePanelLoadBtn: document.getElementById('side-panel-load-btn'),
    openPersonaModalBtn: document.getElementById('open-persona-modal'),
    worldEventInput: document.getElementById('world-event-input'),
    eventTypeSelect: document.getElementById('event-type-select'),
    eventValueInput: document.getElementById('event-value-input'),
    injectEventBtn: document.getElementById('inject-event-btn'),
    testTextModelBtn: document.getElementById('test-text-model'),
    testEmbedBtn: document.getElementById('test-embed'),
    openSysInfoModalBtn: document.getElementById('open-sys-info-modal'),
    personaModal: document.getElementById('persona-modal'),
    personaModalClose: document.getElementById('persona-modal-close'),
    generatePersonaBtn: document.getElementById('generate-persona-btn'),
    personaPrompt: document.getElementById('persona-prompt'),
    personaEditor: document.getElementById('persona-editor'),
    savedPersonasList: document.getElementById('saved-personas-list'),
    loadPersonaBtn: document.getElementById('load-persona-btn'),
    savePersonaName: document.getElementById('save-persona-name'),
    savePersonaBtn: document.getElementById('save-persona-btn'),
    sysInfoModal: document.getElementById('sys-info-modal'),
    sysInfoModalClose: document.getElementById('sys-info-modal-close'),
    summaryModal: document.getElementById('summary-modal'),
    summaryModalClose: document.getElementById('summary-modal-close'),
    summaryModalBody: document.getElementById('summary-modal-body'),
    memoryPanel: document.getElementById('memory-panel'),
    memoryContent: document.getElementById('memory-content')
};

// --- Core Utility Functions (defined first to prevent ReferenceError) ---
function updateStatus(status) {
  DOM.statusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);
  DOM.statusDot.className = 'status-dot ' + status;
  toggleSendStopButtons(status === 'generating');
}

function toggleSendStopButtons(showStop) {
  DOM.sendBtn.style.display = showStop ? 'none' : 'flex';
  DOM.stopBtn.style.display = showStop ? 'flex' : 'none';
}

function renderMarkdown(text) {
    if (typeof text !== 'string') return '';
    if (window.marked && window.DOMPurify) {
        const rawHtml = marked.parse(text, { gfm: true, breaks: true });
        return DOMPurify.sanitize(rawHtml);
    }
    // Fallback parser for environments without the libraries
    return text
        .replace(/^######\s*(.*)$/gm, '<h6>$1</h6>')
        .replace(/^#####\s*(.*)$/gm, '<h5>$1</h5>')
        .replace(/^####\s*(.*)$/gm, '<h4>$1</h4>')
        .replace(/^###\s*(.*)$/gm, '<h3>$1</h3>')
        .replace(/^##\s*(.*)$/gm, '<h2>$1</h2>')
        .replace(/^#\s*(.*)$/gm, '<h1>$1</h1>')
        .replace(/^\s*[-*]\s+(.*)$/gm, '<ul><li>$1</li></ul>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
}

function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function stripThinkTags(text) {
    if (!text) return '';
    return text
        .replace(/<think\b[^>]*>/gi, '')   // Usuń <think> (globalnie, case-insensitive)
        .replace(/<\/think\s*>/gi, '')      // Usuń </think>
        .replace(/<BEGIN_THOUGHTS>/gi, '')  // Usuń tagi BEGIN jeśli zostały
        .replace(/<\/BEGIN_THOUGHTS>/gi, '')
        .trim();
}

// --- NOWE FUNKCJE POMOCNICZE (Wklej pod stripThinkTags) ---

function formatStatsContent(text) {
    if (!text) return '';
    
    let clean = text.replace(/(\*\*\[\[Stats\]\]\*\*|\[\[Stats\]\]|Stats:)/gi, '').trim();
    clean = clean.replace(/,\s/g, '\n');

    return clean.split('\n').map(line => {
        return line
            .replace(/^[\*\-•]\s*/, '') // Usuń kropki listy
            .replace(/^Stat:\s*/i, '')  // NOWE: Usuń słowo "Stat:" na początku linii (case insensitive)
            .replace(/\*/g, '')
            .replace(/[\(\)]/g, '')
            .replace(/_/g, '')
            .trim();
    }).filter(line => line.length > 0)
      .join('\n');
}



function extractWrappedSection(text, startTag, endTag) {
    if (!text) return { section: '', remaining: text };
    
    const startEsc = escapeRegex(startTag);
    const endEsc = escapeRegex(endTag);
    
    // 1. Kompletny blok
    const completePattern = new RegExp(`${startEsc}([\\s\\S]*?)${endEsc}`, 'i');
    const completeMatch = text.match(completePattern);
    
    if (completeMatch) {
        const section = (completeMatch[1] || '').trim();
        const remaining = (text.slice(0, completeMatch.index) + text.slice(completeMatch.index + completeMatch[0].length)).trim();
        return { section, remaining };
    }
    
    // 2. Strumieniowanie (tylko otwarcie)
    const startPattern = new RegExp(`${startEsc}([\\s\\S]*)`, 'i');
    const startMatch = text.match(startPattern);
    
    if (startMatch) {
        const section = (startMatch[1] || '').trim();
        const remaining = text.slice(0, startMatch.index).trim();
        return { section, remaining };
    }

    return { section: '', remaining: text };
}

function normalizeLabeledSection(text, header) {
    if (!text) return '';
    const trimmed = text.trim();
    const headerToken = `**[[${header}]]**`;
    if (!trimmed.toLowerCase().includes(headerToken.toLowerCase())) {
        return `${headerToken}\n${trimmed}`;
    }
    return trimmed;
}

function extractPrefixedSection(text, label) {
    if (!text) return { section: '', remaining: text };

    const pattern = new RegExp(`^\\s*${label}\\s*:\\s*(.+?)(?:\\n{2,}|\\r?\\n\\r?\\n|$)`, 'is');
    const match = text.match(pattern);
    if (!match) return { section: '', remaining: text };

    const section = (match[1] || '').trim();
    const remaining = (text.slice(0, match.index) + text.slice(match.index + match[0].length)).trim();
    return { section, remaining };
}



function cleanFinalThoughts(text) {
    if (!text) return '';
    // To wyrażenie regularne usuwa: **[[Final Thoughts]]**, [[Final Thoughts]], spacje i gwiazdki wokół
    return text.replace(/(\*\*|)\s*\[\[Final Thoughts\]\]\s*(\*\*|:|)/gi, '').trim();
}

function parseFullResponse(fullText) {
    let tempText = fullText || '';
    let thoughts = '', stats = '', finalThoughts = '', cleanContent = '';

    const extractSectionFromEnd = (markerRegex) => {
        const match = tempText.match(markerRegex);
        if (match) {
            const fullSection = match[0];
            const contentOnly = fullSection.replace(/(\*\*|)\s*\[\[(Stats|Final Thoughts)\]\]\s*(\*\*|:|)/gi, '').trim();
            
            // FIX: Usuwanie "Relevant memories" z Final Thoughts na frontendzie
            const cleanContent = contentOnly.split(/(Relevant memories|\[user\]|User:|Relevant context):/i)[0].trim();
            
            tempText = tempText.substring(0, match.index).trim();
            return cleanContent;
        }
        return '';
    };

    // 1. Final Thoughts
    finalThoughts = extractSectionFromEnd(/(\*\*\[\[Final Thoughts\]\]\*\*|\[\[Final Thoughts\]\])([\s\S]*)$/i);

    // 2. Stats
    stats = extractSectionFromEnd(/(\*\*\[\[Stats\]\]\*\*|\[\[Stats\]\])([\s\S]*)$/i);

    // 3. Thoughts & Response (Nowa logika z [[Response]])
    
    // Najpierw szukamy splitera **[[Response]]**
    const responseMatch = tempText.match(/(\*\*\[\[Response\]\]\*\*|\[\[Response\]\])/i);
    
    if (responseMatch) {
        // Wszystko PRZED znacznikiem to myśli
        thoughts = tempText.substring(0, responseMatch.index).trim();
        // Wszystko PO znaczniku to odpowiedź
        cleanContent = tempText.substring(responseMatch.index + responseMatch[0].length).trim();
    } else {
        // Brak znacznika Response. Próbujemy po staremu (szukamy </think>)
        const closingThinkMatch = tempText.match(/<\/think>/i);
        if (closingThinkMatch) {
            const splitIndex = closingThinkMatch.index;
            thoughts = tempText.substring(0, splitIndex);
            cleanContent = tempText.substring(splitIndex + closingThinkMatch[0].length).trim();
        } else {
            // Brak </think> i brak [[Response]]. 
            // Streaming: Zakładamy Force Think (wszystko to myśli, dopóki nie pojawi się znacznik)
            thoughts = tempText;
            cleanContent = ''; 
        }
    }

    if (thoughts) thoughts = stripThinkTags(thoughts);

    return { content: cleanContent, thoughts, stats, final_thoughts: finalThoughts };
}

function normalizeMessageDataForRender(msgData = {}) {
    const normalized = {
        content: msgData.content || '',
        thoughts: msgData.thoughts || '',
        stats: msgData.stats || '',
        final_thoughts: msgData.final_thoughts || msgData.finalThoughts || ''
    };

    const hasInlineStructuredBlocks = /<think\b[^>]*>|\*\*\[\[Thoughts\]\]\*\*/i.test(normalized.content);
    if (hasInlineStructuredBlocks) {
        const parsed = parseFullResponse(normalized.content);
        normalized.content = parsed.content || normalized.content;
        normalized.thoughts = normalized.thoughts || parsed.thoughts;
        normalized.stats = normalized.stats || parsed.stats;
        normalized.final_thoughts = normalized.final_thoughts || parsed.final_thoughts;
    }

    // USUŃ TE LINIE (lub zakomentuj), aby nie dodawało **[[Stats]]** do czystych danych
    // normalized.stats = normalizeLabeledSection(normalized.stats, 'Stats');
    // normalized.final_thoughts = normalizeLabeledSection(normalized.final_thoughts, 'Final Thoughts');

    return normalized;
}

// Dodaj nową funkcję do transformacji nagłówków [[]] na ładny HTML
function replaceRawHeaders(htmlText) {
    if (!htmlText) return '';
    
    // ZMIANA: Regex teraz ignoruje WSZYSTKIE otaczające tagi HTML i białe znaki.
    // Łapie: <p><strong>[[Stats]]</strong></p>, [[Stats]], **[[Stats]]** itd.
    
    // Stats Header
    htmlText = htmlText.replace(
        /(?:<[^>]+>|\s|\*)*\[\[Stats\]\](?:<[^>]+>|\s|\*|:)*/gi, 
        '<div class="stats-header">📊 Stats</div>'
    );

    // Final Thoughts Header
    htmlText = htmlText.replace(
        /(?:<[^>]+>|\s|\*)*\[\[Final Thoughts\]\](?:<[^>]+>|\s|\*|:)*/gi, 
        '<div class="final-thoughts-header">💭 Final Thoughts</div>'
    );

    return htmlText;
}

function updateOrCreateElement(parent, selector, content, position = 'append') {
    if (!content || content.trim() === '') {
        const el = parent.querySelector(selector);
        if (el) el.style.display = 'none';
        return;
    }
    
    let element = parent.querySelector(selector);
    if (!element) {
        element = document.createElement('div');
        element.className = selector.substring(1); 
        if (position === 'prepend') parent.prepend(element);
        else parent.appendChild(element);
    }

    element.style.display = 'block';
    
    // Jeśli treść zawiera jeszcze jakieś stare nagłówki tekstowe (np. podczas streamingu), usuń je
    let cleanContent = content.replace(/(\*\*|)\s*\[\[(Stats|Final Thoughts)\]\]\s*(\*\*|:|)/gi, '').trim();
    
    // Renderujemy Markdown
    let htmlContent = renderMarkdown(cleanContent);

    // --- DOKLEJAMY ŁADNE NAGŁÓWKI HTML ---
    if (selector === '.stats-container') {
        htmlContent = `<div class="stats-header">📊 Stats</div>` + htmlContent;
    } 
    else if (selector === '.final-thoughts-container') {
        htmlContent = `<div class="final-thoughts-header">💭 Final Thoughts</div>` + htmlContent;
    }

    // Obsługa Thoughts (bez zmian)
    if (selector === '.thought-container') {
        const wasExpanded = element.classList.contains('expanded');
        if (!element.innerHTML.trim()) {
            element.innerHTML = `
                <div class="thought-header" role="button" aria-expanded="false">
                    <span class="chevron">▼</span>
                    <span>Thoughts</span>
                </div>
                <div class="thought-content"></div>`;
        }
        const contentEl = element.querySelector('.thought-content');
        if (contentEl && contentEl.innerHTML !== htmlContent) {
            contentEl.innerHTML = htmlContent;
        }
        
        const contentDiv = element.querySelector('.thought-content');
        if(contentDiv && !element.classList.contains('expanded')) {
            requestAnimationFrame(() => contentDiv.scrollTop = contentDiv.scrollHeight);
        }
        
        if (!element.hasToggleListener) {
             element.querySelector('.thought-header').addEventListener('click', () => element.classList.toggle('expanded'));
             element.hasToggleListener = true;
        }
    } else {
        // Aktualizacja HTML dla reszty
        if (element.innerHTML !== htmlContent) {
            element.innerHTML = htmlContent;
        }
    }
}

function renderMessage(msgWrapper, msgData) {
    const msgBody = msgWrapper.querySelector('.message-body');
    if (!msgBody) return;
    msgBody.innerHTML = ''; // Clear for final, clean render

    const normalized = normalizeMessageDataForRender(msgData);

    updateOrCreateElement(msgBody, '.thought-container', normalized.thoughts, 'prepend');
    updateOrCreateElement(msgBody, '.message-content', normalized.content, 'append');
    updateOrCreateElement(msgBody, '.stats-container', normalized.stats, 'append');
    updateOrCreateElement(msgBody, '.final-thoughts-container', normalized.final_thoughts, 'append');
}

// --- Main Application Logic ---
function connectWebSocket() {
    return new Promise((resolve, reject) => {
        if (appState.ws && appState.ws.readyState === WebSocket.OPEN) {
            appState.ws.send(JSON.stringify({ type: 'init', chat_id: appState.activeChatId }));
            resolve();
            return;
        }
        if (appState.ws && (appState.ws.readyState === WebSocket.OPEN || appState.ws.readyState === WebSocket.CONNECTING)) {
    try { appState.ws.send(JSON.stringify({ type: 'init', chat_id: (appState.activeChatId || 'default_chat') })); } catch(e){}
} else {
    const wsUrl = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
    appState.ws = new WebSocket(wsUrl);
}

        appState.ws.onopen = () => {
            updateStatus('connected');
            DOM.sendBtn.disabled = false;
            appState.ws.send(JSON.stringify({ type: 'init', chat_id: appState.activeChatId }));
            resolve();
        };
        appState.ws.onclose = () => {
            updateStatus('disconnected');
            DOM.sendBtn.disabled = true;
            appState.isGenerating = false;
            setTimeout(connectWebSocket, 3000);
        };
        appState.ws.onerror = (error) => {
            updateStatus('disconnected');
            appState.isGenerating = false;
            console.error("WebSocket Error:", error);
            reject(error);
        };
        appState.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            } catch (e) {
                console.error("Failed to parse WebSocket message:", event.data, e);
            }
        };
    });
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'start': // This is now mainly a server-side confirmation, UI is handled by sendMessage
            appState.isGenerating = true;
            updateStatus('generating');
            appState.fullResponseText = '';
            appState.isStreamInitialized = false;
            appState.awaitingAssistantTimestamp = false;
            if(data.old_ts) {
                document.querySelector(`.message[data-ts="${data.old_ts}"]`)?.remove();
            }
            break;
        case 'partial':
            handlePartialMessage(data.chunk);
            break;
        case 'done':
        case 'stopped':
            appState.isGenerating = false;
            updateStatus('connected');
            appState.awaitingAssistantTimestamp = true;
            // Niektóre dostawcy zwracają końcowy fragment tylko w zdarzeniu "done"
            if (typeof data.chunk === 'string' && data.chunk.trim()) {
                appState.fullResponseText += data.chunk;
            }
            // Zapewnienie, że zawsze mamy kontener, nawet jeśli strumień nigdy nie wystartował
            if (!appState.currentMessageContainer) {
                appState.currentMessageContainer = addMessage('assistant');
            }
            
            // UWAGA: Pełne renderowanie jest teraz celowo przesunięte do 'assistant_ts' 
            // w celu zsynchronizowania z zapisem na dysku.
            
            // Użyj pełnego tekstu odpowiedzi, który został zgromadzony podczas strumieniowania.
            if (appState.currentMessageContainer && appState.fullResponseText) {
                // Renderujemy to, co mamy, jeśli strumień się zatrzymał.
                renderMessage(appState.currentMessageContainer, parseFullResponse(appState.fullResponseText));
            }

            if (data.type === 'stopped') {
                // Jeśli strumień został zatrzymany ręcznie, renderujemy to, co mamy, i czyścimy stan.
                appState.awaitingAssistantTimestamp = false;
                appState.currentMessageContainer = null;
                appState.fullResponseText = ''; 
            }
            break;
        case 'error':
            appState.isGenerating = false;
            updateStatus('connected');
            addMessage('system', `[SERVER ERROR] ${data.message}`);
            break;
        case 'user_ts':
            const el = Array.from(document.querySelectorAll('.message.user')).pop();
            if (el && !el.dataset.ts) {
                el.dataset.ts = data.ts;
                addMessageFooter(el, 'user');
            }
            break;
        case 'assistant_ts':
            if (appState.currentMessageContainer) {
                appState.currentMessageContainer.dataset.ts = data.ts;
                addMessageFooter(appState.currentMessageContainer, 'assistant');
                
                // POPRAWKA: Pobieramy pewną, zapisaną wersję wiadomości z serwera
                fetch(`/${appState.activeChatId}/messages`)
                    .then(r => r.json())
                    .then(msgs => {
                        const saved = msgs.find(m => m.ts === data.ts);
                        if (saved) renderMessage(appState.currentMessageContainer, saved);
                        else renderMessage(appState.currentMessageContainer, parseFullResponse(appState.fullResponseText));
                    })
                    .catch(() => renderMessage(appState.currentMessageContainer, parseFullResponse(appState.fullResponseText)));
                
                appState.awaitingAssistantTimestamp = false;
                appState.currentMessageContainer = null;
                appState.fullResponseText = '';
            }
            break;
    }
}

function handlePartialMessage(chunk){
  if(!appState.currentMessageContainer) return;
  const body = appState.currentMessageContainer.querySelector('.message-body');
  if(!body) return;
  if(!appState.isStreamInitialized){
    body.querySelector('.responding-indicator')?.remove();
    appState.isStreamInitialized = true;
  }
  appState.fullResponseText += (chunk || '');
  const parsed = parseFullResponse(appState.fullResponseText);

  // 1. Myśli (prepend - na samej górze)
  updateOrCreateElement(body, '.thought-container', parsed.thoughts, 'prepend');
  
  // 2. Odpowiedź główna (message-content - append)
  updateOrCreateElement(body, '.message-content', parsed.content, 'append'); 
  
  // 3. Statystyki (append)
  updateOrCreateElement(body, '.stats-container', parsed.stats, 'append');
  
  // 4. Final Thoughts (append - na samym dole)
  updateOrCreateElement(body, '.final-thoughts-container', parsed.final_thoughts, 'append');
  
  scrollToBottom();
}



function addMessage(role, content = '', ts = '', thoughts = '', stats = '', final_thoughts = '') {
    const msgWrapper = document.createElement('div');
    msgWrapper.className = `message ${role}`;
    if (ts) msgWrapper.dataset.ts = ts;
    const msgBody = document.createElement('div');
    msgBody.className = 'message-body';
    msgWrapper.appendChild(msgBody);
    
    if (role !== 'assistant' || (role === 'assistant' && !appState.isGenerating)) {
        renderMessage(msgWrapper, { content, thoughts, stats, final_thoughts });
    }

    if (role !== 'system' && ts) {
        addMessageFooter(msgWrapper, role);
    }
    DOM.chatMessages.appendChild(msgWrapper);
    scrollToBottom();
    return msgWrapper;
}

function addMessageFooter(msgWrapper, role) {
    msgWrapper.querySelector('.message-footer')?.remove();
    const footer = document.createElement('div');
    footer.className = 'message-footer';
    const timestamp = document.createElement('span');
    timestamp.className = 'message-timestamp';
    timestamp.textContent = new Date(msgWrapper.dataset.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const actionsContainer = document.createElement('div');
    actionsContainer.className = 'message-actions';
    actionsContainer.innerHTML = `<button title="Edit">📝</button>${role === 'assistant' ? '<button title="Regenerate">🔄</button>' : ''}<button title="Delete">🗑️</button>`;
    footer.appendChild(timestamp);
    footer.appendChild(actionsContainer);
    msgWrapper.appendChild(footer);
    actionsContainer.querySelector('[title="Delete"]').onclick = () => deleteMessage(msgWrapper.dataset.ts, msgWrapper);
    actionsContainer.querySelector('[title="Edit"]').onclick = () => editMessage(msgWrapper, role);
    if (role === 'assistant') {
        actionsContainer.querySelector('[title="Regenerate"]').onclick = () => regenerateMessage(msgWrapper.dataset.ts);
    }
}

async function deleteMessage(ts, element) {
  if (!ts) return;
  if (confirm('Delete this message?')) {
    await api(`/${appState.activeChatId}/delete_message`, 'POST', { ts });
    element.remove();
  }
}

function editMessage(msgWrapper, role) {
    const msgBody = msgWrapper.querySelector('.message-body');
    if (!msgBody || msgBody.querySelector('textarea')) return;
    const getTextFromContainer = (selector) => {
        const el = msgWrapper.querySelector(selector);
        if (!el || el.style.display === 'none') return '';
        const tempDiv = document.createElement('div');
        const contentSource = el.querySelector('.thought-content') || el;
        tempDiv.innerHTML = contentSource.innerHTML.replace(/<br\s*[\/]?>/gi, "\n");
        return tempDiv.textContent || tempDiv.innerText || "";
    };
    const thoughts = getTextFromContainer('.thought-container');
    const content = getTextFromContainer('.message-content');
    const stats = getTextFromContainer('.stats-container');
    const finalThoughts = getTextFromContainer('.final-thoughts-container');
    let fullRawText = '';
    if (thoughts) fullRawText += `<think>${thoughts.trim()}</think>\n\n`;
    fullRawText += content.trim();
    if (stats) fullRawText += `\n\n${stats.trim()}`;
    if (finalThoughts) fullRawText += `\n\n${finalThoughts.trim()}`;
    const originalHTML = msgBody.innerHTML;
    const editor = document.createElement('textarea');
    editor.className = 'chat-input';
    editor.style.width = '100%';
    editor.value = fullRawText.trim();
    const btnContainer = document.createElement('div');
    btnContainer.style.marginTop = '10px'; btnContainer.style.display = 'flex'; btnContainer.style.gap = '8px';
    const saveBtn = document.createElement('button');
    saveBtn.className = 'btn'; saveBtn.textContent = 'Save';
    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'btn danger'; cancelBtn.textContent = 'Cancel';
    btnContainer.appendChild(saveBtn); btnContainer.appendChild(cancelBtn);
    msgBody.innerHTML = ''; msgBody.appendChild(editor); msgBody.appendChild(btnContainer);
    editor.focus(); editor.style.height = 'auto'; editor.style.height = `${editor.scrollHeight}px`;
    cancelBtn.onclick = () => { msgBody.innerHTML = originalHTML; };
    saveBtn.onclick = async () => {
        const newRawContent = editor.value;
        const result = await api(`/${appState.activeChatId}/edit_message`, 'POST', { ts: msgWrapper.dataset.ts, raw_content: newRawContent });
        renderMessage(msgWrapper, result.updated_message);
        addMessageFooter(msgWrapper, role);
    };
}

async function regenerateMessage(ts) {
    if (!ts || appState.isGenerating) return;
    if (!confirm('Regenerate this response? The current one will be deleted.')) return;
    appState.ws.send(JSON.stringify({ type: 'regenerate', ts, chat_id: appState.activeChatId, settings: getSettings() }));
    // Immediately show visual feedback for regeneration
    appState.isGenerating = true;
    updateStatus('generating');
    const oldMessage = document.querySelector(`.message[data-ts="${ts}"]`);
    if(oldMessage) {
        appState.currentMessageContainer = oldMessage;
        const body = oldMessage.querySelector('.message-body');
        if(body) body.innerHTML = '<div class="responding-indicator"><span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>';
    }
}

function getSettings() {
    return {
      model: DOM.modelSelect.value,
      embedding_model: DOM.embeddingModelSelect.value,
      temperature: parseFloat(DOM.tempSlider.value),
      max_tokens: parseInt(DOM.tokensSlider.value),
      thought_ratio: parseFloat(DOM.thoughtSlider.value),
      talkativeness: parseFloat(DOM.talkSlider.value),
      persistent_stats: DOM.persistentStatsToggle.checked,
      enable_memory: DOM.enableMemoryToggle.checked,
    };
}

function saveSettings() {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(getSettings()));
}

function loadSettings() {
    const savedSettings = localStorage.getItem(SETTINGS_KEY);
    if (savedSettings) {
        try {
            const settings = JSON.parse(savedSettings);
            if(settings.model) DOM.modelSelect.value = settings.model;
            if(settings.embedding_model) DOM.embeddingModelSelect.value = settings.embedding_model;
            DOM.tempSlider.value = settings.temperature || 1.0;
            DOM.tokensSlider.value = settings.max_tokens || 1024;
            DOM.thoughtSlider.value = settings.thought_ratio || 0.5;
            DOM.talkSlider.value = settings.talkativeness || 0.5;
            DOM.persistentStatsToggle.checked = settings.persistent_stats === true;
            DOM.enableMemoryToggle.checked = settings.enable_memory !== false;
            DOM.tempValue.textContent = DOM.tempSlider.value;
            DOM.tokensValue.textContent = DOM.tokensSlider.value;
            DOM.thoughtValue.textContent = DOM.thoughtSlider.value;
            DOM.talkValue.textContent = DOM.talkSlider.value;
        } catch (e) { console.error("Failed to load settings", e); }
    }
}

function scrollToBottom() {
    setTimeout(() => {
        DOM.chatMessages.scrollTo({ top: DOM.chatMessages.scrollHeight, behavior: 'smooth' });
    }, 100);
}

async function sendMessage() {
    const message = DOM.chatInput.value.trim();
    if (!message || appState.isGenerating) return;
    appState.fullResponseText = '';
    appState.isStreamInitialized = false;
    DOM.memoryPanel.style.display = 'none';
    DOM.memoryContent.innerHTML = '';
    DOM.chatInput.value = '';
    DOM.chatInput.style.height = 'auto';
    addMessage('user', message, new Date().toISOString());
    try {
        await connectWebSocket();
        appState.ws.send(JSON.stringify({ type: 'message', message, settings: getSettings(), chat_id: appState.activeChatId }));
        // Start "responding" animation immediately for better UX
        appState.isGenerating = true;
        updateStatus('generating');
        appState.currentMessageContainer = addMessage('assistant');
        const body = appState.currentMessageContainer.querySelector('.message-body');
        // Show a minimal animated placeholder instead of the full template
        if (body) {
            body.innerHTML = '<div class="responding-indicator generating-text">Generating<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>';
        }
    } catch (error) {
        addMessage('system', '[ERROR] Connection failed. Please check the server.');
    }
}

function stopGeneration() {
  if (appState.ws && appState.ws.readyState === WebSocket.OPEN && appState.isGenerating) {
    appState.ws.send(JSON.stringify({ type: 'stop', chat_id: appState.activeChatId }));
  }
}

async function api(path, method = 'GET', body = null) {
  try {
    const opts = { method };
    if (body) {
        opts.headers = { 'Content-Type': 'application/json' };
        opts.body = JSON.stringify(body);
    }
    const response = await fetch(path, opts);
    const contentType = response.headers.get("content-type");
    const isJson = contentType && contentType.includes("application/json");
    if (!response.ok) {
        const errorData = isJson ? await response.json() : await response.text();
        const errorText = isJson ? errorData.error || JSON.stringify(errorData) : errorData;
        const errorMsg = `[API ERROR] ${response.status}: ${errorText}`;
        addMessage('system', errorMsg);
        console.error("API Error Response:", errorData);
        throw new Error(`API Error: ${response.status} ${errorText}`);
    }
    return isJson ? await response.json() : await response.text();
  } catch (error) {
      console.error("API call failed:", error);
      addMessage('system', `[API ERROR] Failed to fetch from ${path}. Check server console & .env config.`);
      throw error;
  }
}

async function clearMemory() {
  if (!confirm('Clear all memory for this chat? This deletes conversation history, summaries, stats and events.')) return;
  await api(`/${appState.activeChatId}/clear_memory`, 'POST');
  await reloadChat();
}

async function testEmbeddings() {
    const selectedModel = DOM.embeddingModelSelect.value;
    if (!selectedModel) {
        alert("Please select an embedding model to test.");
        return;
    }
    addMessage('system', `Testing embedding model: ${selectedModel}...`);
    try {
        const res = await api('/test_embeddings', 'POST', { model: selectedModel });
        addMessage('system', res.success ? `✅ Embedding test successful!` : `❌ Embedding test failed: ${res.error}`);
    } catch(e) {
        console.error("Failed to test embeddings:", e);
    }
}

async function loadAvailableModels() {
    try {
        const data = await api('/models');
        DOM.modelSelect.innerHTML = '';
        DOM.embeddingModelSelect.innerHTML = '';
        if (data.models && data.models.length > 0) {
            data.models.forEach(modelId => {
                const option = document.createElement('option');
                option.value = modelId;
                option.textContent = modelId;
                DOM.modelSelect.appendChild(option.cloneNode(true));
                DOM.embeddingModelSelect.appendChild(option);
            });
        } else {
             const errorHtml = '<option value="">No models found</option>';
             DOM.modelSelect.innerHTML = errorHtml;
             DOM.embeddingModelSelect.innerHTML = errorHtml;
        }
    } catch (e) {
        const errorHtml = '<option value="">Error loading models</option>';
        DOM.modelSelect.innerHTML = errorHtml;
        DOM.embeddingModelSelect.innerHTML = errorHtml;
    }
    loadSettings();
}

function setupPanelToggles() {
    document.querySelectorAll('.panel-toggle').forEach(toggle => {
        toggle.addEventListener('click', (e) => {
            const content = e.target.nextElementSibling;
            if (content && content.classList.contains('collapsible-content')) {
                content.classList.toggle('show');
                e.target.classList.toggle('collapsed');
            }
        });
    });
    const toggleLogic = (panel, container, className) => {
        const isOpening = !container.classList.contains(className);
        const otherPanelClass = className === 'left-panel-open' ? 'right-panel-open' : 'left-panel-open';
        if (isOpening && window.innerWidth <= 1024) {
            container.classList.remove(otherPanelClass);
            if(otherPanelClass === 'left-panel-open') DOM.leftPanel.classList.add('collapsed');
            else DOM.rightPanel.classList.add('collapsed');
        }
        panel.classList.toggle('collapsed');
        container.classList.toggle(className);
    };
    DOM.leftPanelToggle.addEventListener('click', () => toggleLogic(DOM.leftPanel, DOM.appContainer, 'left-panel-open'));
    DOM.rightPanelToggle.addEventListener('click', () => toggleLogic(DOM.rightPanel, DOM.appContainer, 'right-panel-open'));
    DOM.mobileMenuLeft.addEventListener('click', () => toggleLogic(DOM.leftPanel, DOM.appContainer, 'left-panel-open'));
    DOM.mobileMenuRight.addEventListener('click', () => toggleLogic(DOM.rightPanel, DOM.appContainer, 'right-panel-open'));
    document.querySelectorAll('.collapse-handle').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const panel = e.target.closest('.side-panel');
            if (panel.id === 'left-panel') DOM.appContainer.classList.remove('left-panel-open');
            else if (panel.id === 'right-panel') DOM.appContainer.classList.remove('right-panel-open');
            panel.classList.add('collapsed');
        });
    });
}
function autoResizeTextarea(el) {
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 240) + 'px';
}

async function reloadChat() {
    if (!appState.activeChatId) return;
    try {
        const messages = await api(`/${appState.activeChatId}/messages`);
        DOM.chatMessages.innerHTML = '';
        (messages || []).forEach(msg => {
            if (!msg) return;
            const finalThoughts = msg.final_thoughts || msg.finalThoughts || '';
            addMessage(
                msg.role || 'assistant',
                msg.content || '',
                msg.ts,
                msg.thoughts || '',
                msg.stats || '',
                finalThoughts
            );
        });
        scrollToBottom();
    } catch (error) {
        console.error('Failed to reload chat history:', error);
        addMessage('system', '[ERROR] Failed to load chat history.');
    }
}

async function loadSavedPersonasIntoSelect(selectElement) {
    if (!selectElement) return;
    try {
        const personas = await api('/personas');
        selectElement.innerHTML = '';
        if (personas && personas.length) {
            personas.forEach(name => {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name;
                selectElement.appendChild(option);
            });
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No saved personas';
            selectElement.appendChild(option);
        }
    } catch (error) {
        console.error('Failed to load saved personas:', error);
        selectElement.innerHTML = '<option value="">Error loading personas</option>';
    }
}

async function refreshPersonaLists() {
    await loadSavedPersonasIntoSelect(DOM.savedPersonasList);
    await loadSavedPersonasIntoSelect(DOM.sidePanelPersonaPreset);
}

async function refreshActivePersona() {
    try {
        const persona = await api(`/${appState.activeChatId}/persona`);
        DOM.personaEditor.value = JSON.stringify(persona, null, 2);
        
        // Obsługa ścieżki avatara
        // Jeśli avatar to np. "vex.png", dodajemy prefix "/avatars/"
        // Jeśli już ma http, zostawiamy (na przyszłość)
        let avatarSrc = persona.avatar || 'default.png';
        if (!avatarSrc.startsWith('http') && !avatarSrc.startsWith('/')) {
            avatarSrc = '/avatars/' + avatarSrc;
        }
        
        DOM.personaAvatar.src = avatarSrc;
        return persona;
    } catch (error) {
        console.error('Failed to load active persona:', error);
        return null;
    }
}
async function loadAndActivatePersona(name) {
    if (!name) {
        alert('Please select a persona to load.');
        return;
    }
    try {
        const persona = await api(`/personas/${encodeURIComponent(name)}`);
        await api(`/${appState.activeChatId}/persona`, 'POST', persona);
        DOM.personaEditor.value = JSON.stringify(persona, null, 2);
        DOM.personaAvatar.src = `/static/${persona.avatar || 'default_avatar.png'}`;
        addMessage('system', `Persona "${name}" loaded for this chat.`);
    } catch (error) {
        console.error('Failed to load persona:', error);
        alert('Failed to load persona. Please check the server logs.');
    }
}

async function generatePersonaFromPrompt() {
    const prompt = DOM.personaPrompt.value.trim();
    if (!prompt) {
        alert('Please provide a short description to generate a persona.');
        return;
    }
    DOM.generatePersonaBtn.disabled = true;
    DOM.generatePersonaBtn.textContent = 'Generating...';
    try {
        const result = await api('/generate_persona', 'POST', { description: prompt });
        if (result && result.persona) {
            DOM.personaEditor.value = JSON.stringify(result.persona, null, 2);
            DOM.personaAvatar.src = `/static/${result.persona.avatar || 'default_avatar.png'}`;
        }
    } catch (error) {
        console.error('Failed to generate persona:', error);
        alert('Failed to generate persona. Please try again.');
    } finally {
        DOM.generatePersonaBtn.textContent = 'Generate';
        DOM.generatePersonaBtn.disabled = false;
    }
}

async function savePersona() {
    const name = DOM.savePersonaName.value.trim();
    if (!name) {
        alert('Please enter a name to save the persona.');
        return;
    }
    let persona;
    try {
        persona = JSON.parse(DOM.personaEditor.value || '{}');
    } catch (error) {
        alert('Persona JSON is invalid. Please correct it before saving.');
        return;
    }
    try {
        await api(`/personas/${encodeURIComponent(name)}`, 'POST', persona);
        await refreshPersonaLists();
        DOM.savePersonaName.value = '';
        addMessage('system', `Persona "${name}" saved.`);
    } catch (error) {
        console.error('Failed to save persona:', error);
        alert('Failed to save persona.');
    }
}

async function injectWorldEvent() {
    const eventText = DOM.worldEventInput.value.trim();
    if (!eventText) {
        alert('Please describe the world event before injecting it.');
        return;
    }
    try {
        await api(`/${appState.activeChatId}/inject_event`, 'POST', {
            event: eventText,
            type: DOM.eventTypeSelect.value,
            value: parseInt(DOM.eventValueInput.value, 10)
        });
        DOM.worldEventInput.value = '';
        addMessage('system', `[WORLD EVENT INJECTED] ${eventText}`);
    } catch (error) {
        console.error('Failed to inject world event:', error);
        alert('Failed to inject world event.');
    }
}

async function markNewDay() {
    // 1. Zmień status na "Generating" (Żółta lampka)
    updateStatus('generating');
    addMessage('system', '🌞 Starting a new day... (Generating summary & updating context)');
    
    try {
        const response = await api(`/${appState.activeChatId}/new_day`, 'POST');
        
        // 2. Obsługa odpowiedzi
        if (response.marker) {
            addMessage('system', response.marker);
        }
        if (response.summary) {
            addMessage('system', `📝 Summary Generated:\n${response.summary}`);
        }
        
        addMessage('system', '✅ New Day Started. The bot will greet you in the next message.');
        
        // Opcjonalnie przeładuj, żeby upewnić się, że wszystko jest zsynchronizowane
        await reloadChat();
        
    } catch (error) {
        alert('Failed to mark a new day.');
    } finally {
        // 3. Przywróć status "Connected" (Zielona lampka)
        updateStatus('connected');
    }
}

async function checkSummary() {
    try {
        const result = await api(`/${appState.activeChatId}/last_summary`);
        const summary = result && result.summary ? result.summary.trim() : '';
        DOM.summaryModalBody.textContent = summary || 'No summary available yet.';
        DOM.summaryModal.style.display = 'flex';
    } catch (error) {
        console.error('Failed to fetch last summary:', error);
        alert('Failed to fetch the last summary.');
    }
}

async function testTextModel() {
    const selectedModel = DOM.modelSelect.value;
    if (!selectedModel) {
        alert('Please select a text model to test.');
        return;
    }
    addMessage('system', `Testing text model: ${selectedModel}...`);
    try {
        const res = await api('/test_text_model', 'POST', { model: selectedModel });
        addMessage('system', res.success ? '✅ Text model test successful!' : `❌ Text model test failed: ${res.error}`);
    } catch (error) {
        console.error('Failed to test text model:', error);
    }
}

function openModal(modal) {
    if (modal) modal.style.display = 'flex';
}

function closeModal(modal) {
    if (modal) modal.style.display = 'none';
}

function setupEventListeners() {
    DOM.sendBtn.addEventListener('click', sendMessage);
    DOM.stopBtn.addEventListener('click', stopGeneration);

    DOM.chatInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
    DOM.chatInput.addEventListener('input', () => {
        const hasText = DOM.chatInput.value.trim().length > 0;
        DOM.sendBtn.disabled = !hasText || appState.isGenerating;
        autoResizeTextarea(DOM.chatInput);
    });

    DOM.addChatBtn.addEventListener('click', async () => {
        const name = DOM.newChatName.value.trim();
        if (!name) {
            alert('Please enter a chat name.');
            return;
        }
        try {
            const res = await api('/chats/create', 'POST', { name });
            DOM.newChatName.value = '';
            await loadChatList();
            if (res && res.chat_id) {
                await switchChat(res.chat_id);
            }
        } catch (error) {
            console.error('Failed to create chat:', error);
            alert('Failed to create chat.');
        }
    });

    DOM.reloadChatBtn.addEventListener('click', reloadChat);
    DOM.clearMemoryBtn.addEventListener('click', clearMemory);
    DOM.newDayBtn.addEventListener('click', markNewDay);
    DOM.checkSummaryBtn.addEventListener('click', checkSummary);
    DOM.forceSummaryBtn.addEventListener('click', forceSummarize);

    DOM.injectEventBtn.addEventListener('click', injectWorldEvent);

    DOM.testTextModelBtn.addEventListener('click', testTextModel);
    DOM.testEmbedBtn.addEventListener('click', testEmbeddings);

    DOM.openPersonaModalBtn.addEventListener('click', async () => {
        openModal(DOM.personaModal);
        await refreshPersonaLists();
    });
    DOM.personaModalClose.addEventListener('click', () => closeModal(DOM.personaModal));
    // Otwieranie modala - pobiera AKTUALNĄ personę czatu
    DOM.openPersonaModalBtn.addEventListener('click', async () => {
        openModal(DOM.personaModal);
        await refreshPersonaLists();
        // Pobierz aktualną personę z serwera
        const currentPersona = await refreshActivePersona();
        if (currentPersona) {
            DOM.personaEditor.value = JSON.stringify(currentPersona, null, 2);
        }
    });

    // Przycisk: Zapisz jako Preset (Stary przycisk)
    DOM.savePersonaBtn.addEventListener('click', savePersona); // To zostaje bez zmian

    // NOWY: Nadpisz istniejący preset
    const overwriteBtn = document.getElementById('overwrite-persona-btn');
    if (overwriteBtn) {
        overwriteBtn.addEventListener('click', async () => {
            const selectedPreset = DOM.savedPersonasList.value;
            if (!selectedPreset) { alert('Please select a preset to overwrite.'); return; }
            if (!confirm(`Overwrite preset "${selectedPreset}"?`)) return;
            try {
                const personaData = JSON.parse(DOM.personaEditor.value);
                await api(`/personas/${encodeURIComponent(selectedPreset)}`, 'POST', personaData);
                alert(`Preset "${selectedPreset}" updated!`);
                await refreshPersonaLists();
                DOM.savedPersonasList.value = selectedPreset;
            } catch (e) { alert('Error saving preset.'); }
        });
    }

    // NOWY: Zastosuj do bieżącego czatu
    const applyBtn = document.getElementById('save-current-persona-btn');
    if (applyBtn) {
        applyBtn.addEventListener('click', async () => {
            try {
                const personaData = JSON.parse(DOM.personaEditor.value);
                await api(`/${appState.activeChatId}/persona`, 'POST', personaData);
                await refreshActivePersona();
                alert('Persona applied to chat!');
                closeModal(DOM.personaModal);
            } catch (e) { alert('Invalid JSON.'); }
        });
    }

    DOM.sidePanelLoadBtn.addEventListener('click', () => loadAndActivatePersona(DOM.sidePanelPersonaPreset.value));
    DOM.loadPersonaBtn.addEventListener('click', () => loadAndActivatePersona(DOM.savedPersonasList.value));
    DOM.generatePersonaBtn.addEventListener('click', generatePersonaFromPrompt);
    DOM.savePersonaBtn.addEventListener('click', savePersona);

    DOM.openSysInfoModalBtn.addEventListener('click', async () => {
        openModal(DOM.sysInfoModal);
        try {
            const info = await api('/system_info');
            const versionEl = document.getElementById('sys-info-version');
            const modelEl = document.getElementById('sys-info-model');
            if (versionEl) versionEl.textContent = info.version || 'unknown';
            if (modelEl) modelEl.textContent = info.model_name || 'unknown';
        } catch (error) {
            console.error('Failed to fetch system info:', error);
        }
    });
    DOM.sysInfoModalClose.addEventListener('click', () => closeModal(DOM.sysInfoModal));

    DOM.summaryModalClose.addEventListener('click', () => closeModal(DOM.summaryModal));

    const settingsControls = [
        DOM.modelSelect,
        DOM.embeddingModelSelect,
        DOM.tempSlider,
        DOM.tokensSlider,
        DOM.thoughtSlider,
        DOM.talkSlider,
        DOM.persistentStatsToggle,
        DOM.enableMemoryToggle
    ];
    const sliderLabels = {
        [DOM.tempSlider.id]: DOM.tempValue,
        [DOM.tokensSlider.id]: DOM.tokensValue,
        [DOM.thoughtSlider.id]: DOM.thoughtValue,
        [DOM.talkSlider.id]: DOM.talkValue
    };
    settingsControls.forEach(control => {
        control.addEventListener('input', (event) => {
            if (event.target.type === 'range' && sliderLabels[event.target.id]) {
                sliderLabels[event.target.id].textContent = event.target.value;
            }
            saveSettings();
        });
    });
    autoResizeTextarea(DOM.chatInput);
    DOM.chatInput.dispatchEvent(new Event('input'));
}

async function loadChatList() {
    const chats = await api('/chats');
    DOM.chatList.innerHTML = '';
    if (chats.length === 0) {
        await api('/chats/create', 'POST', { name: 'default_chat' });
        return loadChatList();
    }
    chats.forEach(chatId => {
        const li = document.createElement('li');
        li.className = 'chat-list-item'; li.dataset.chatId = chatId;
        const nameSpan = document.createElement('span');
        nameSpan.textContent = chatId; nameSpan.style.flexGrow = '1'; nameSpan.style.overflow = 'hidden'; nameSpan.style.textOverflow = 'ellipsis';
        const deleteBtn = document.createElement('span');
        deleteBtn.className = 'delete-chat-btn'; deleteBtn.innerHTML = '&times;'; deleteBtn.title = `Delete chat "${chatId}"`;
        li.appendChild(nameSpan); li.appendChild(deleteBtn);
        if (chatId === appState.activeChatId) li.classList.add('active');
        li.addEventListener('click', async (e) => {
            if (e.target !== deleteBtn) {
                try {
                    await switchChat(chatId);
                } catch (error) {
                    console.error('Failed to switch chat:', error);
                }
            }
        });
        deleteBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            try {
                await deleteChat(chatId);
            } catch (error) {
                console.error('Failed to delete chat:', error);
            }
        });
        DOM.chatList.appendChild(li);
    });
}

async function deleteChat(chatId) {
    if (appState.isGenerating) { alert("Cannot delete a chat while a response is being generated."); return; }
    if (!confirm(`Are you sure you want to permanently delete the chat "${chatId}"? This cannot be undone.`)) return;
    await api(`/chats/${chatId}`, 'DELETE');
    if (appState.activeChatId === chatId) {
        const chats = await api('/chats');
        const newActiveChat = chats.length > 0 ? chats[0] : 'default_chat';
        await switchChat(newActiveChat);
    }
    await loadChatList();
}

async function switchChat(chatId) {
    if (!chatId) return;
    if (appState.activeChatId === chatId && appState.ws && appState.ws.readyState === WebSocket.OPEN) return;
    appState.activeChatId = chatId;
    DOM.chatTitle.textContent = `Chat: ${chatId}`;
    console.log(`Switching to chat: ${appState.activeChatId}`);
    document.querySelectorAll('.chat-list-item.active').forEach(el => el.classList.remove('active'));
    document.querySelector(`.chat-list-item[data-chat-id="${chatId}"]`)?.classList.add('active');
    appState.isGenerating = false;
    stopGeneration();
    await connectWebSocket();
    await reloadChat();
    await refreshActivePersona();
    DOM.chatInput.value = '';
    DOM.chatInput.dispatchEvent(new Event('input'));
}

async function forceSummarize() {
    // 1. Zmień status na "Generating"
    updateStatus('generating');
    const loadingMsg = addMessage('system', 'Generating summary using LLM...');
    
    // Dodajmy klasę animacji do ostatniej wiadomości, żeby kropki "żyły"
    // (Możesz dodać klasę .responding-indicator wewnątrz dymku systemowego jeśli chcesz)
    
    try {
        const response = await api(`/${appState.activeChatId}/force_summarize`, 'POST');
        // Usuwamy wiadomość "Generating..." i wstawiamy gotowe podsumowanie
        loadingMsg.remove(); 
        addMessage('system', `📝 Summary Generated:\n${response.summary}`);
    } catch (error) {
        console.error(error);
        addMessage('system', '❌ Failed to generate summary.');
    } finally {
        // 2. Przywróć status
        updateStatus('connected');
    }
}

async function initializeApp() {
    if (window.appState.isInitialized) return;
    window.appState.isInitialized = true;
    try {
        setupEventListeners();
        setupPanelToggles();
        await loadAvailableModels();
        await loadChatList();
        if (!document.querySelector('.chat-list-item.active')) {
            const firstChat = document.querySelector('.chat-list-item');
            if(firstChat) await switchChat(firstChat.dataset.chatId);
            else await switchChat('default_chat');
        }
        DOM.chatTitle.textContent = `Chat: ${appState.activeChatId}`;
        
        // ZMIANA: Tutaj korzystamy ze zmiennej globalnej zdefiniowanej w HTML
        if (window.SERVER_CONFIG) {
            DOM.persistentStatsToggle.checked = window.SERVER_CONFIG.persistentStats;
        }
        
        await connectWebSocket();
        await reloadChat();
        await refreshPersonaLists();
        await refreshActivePersona();
    } catch (error) {
        console.error("Init failed:", error);
        addMessage('system', 'Initialization error. Check console.');
    }
}

document.addEventListener('DOMContentLoaded', initializeApp);