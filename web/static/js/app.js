// Chat Interface JavaScript

let chatHistory = [];
let currentConversationId = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    const messageInput = document.getElementById('messageInput');
    messageInput.focus();
    
    // Load conversations list
    loadConversations();
    
    // Start new conversation if none exists
    startNewConversation();
});

// Load chat history from server
async function loadChatHistory() {
    try {
        // Show loading indicator
        const chatMessages = document.getElementById('chatMessages');
        const loadingId = addMessage('assistant', 'Loading chat history...', true);
        
        const response = await fetch('/api/chat/history?limit=50');
        const data = await response.json();
        
        // Remove loading indicator
        removeMessage(loadingId);
        
        if (data.success && data.history && data.history.length > 0) {
            // Clear existing messages (except welcome)
            const existingMessages = chatMessages.querySelectorAll('.message');
            existingMessages.forEach(msg => {
                if (!msg.closest('.welcome-message')) {
                    msg.remove();
                }
            });
            
            // Clear welcome message if we have history
            const welcomeMsg = document.querySelector('.welcome-message');
            if (welcomeMsg) {
                welcomeMsg.remove();
            }
            
            // Display history (most recent first, but show oldest first)
            const reversedHistory = [...data.history].reverse();
            for (const item of reversedHistory) {
                // Add user message
                addMessage('user', item.user_message);
                
                // Add assistant response
                addMessage('assistant', formatMarkdown(item.assistant_response));
            }
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Show success message
            console.log(`Loaded ${data.history.length} previous interactions`);
        } else {
            // No history, show welcome message if not already shown
            if (!document.querySelector('.welcome-message')) {
                const welcomeMsg = document.createElement('div');
                welcomeMsg.className = 'welcome-message';
                welcomeMsg.innerHTML = `
                    <h3>👋 Welcome to Log Analyzer</h3>
                    <p>Upload a log file or paste log content to get started with AI-powered analysis.</p>
                `;
                chatMessages.appendChild(welcomeMsg);
            }
        }
    } catch (error) {
        console.error('Failed to load chat history:', error);
        const loadingMsg = document.querySelector('.message.assistant .loading');
        if (loadingMsg) {
            loadingMsg.closest('.message').remove();
        }
    }
}

// Handle Enter key
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Send message
async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) {
        return;
    }
    
    // Clear welcome message
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    // Add user message
    addMessage('user', message);
    messageInput.value = '';
    
    // Disable send button
    const sendButton = document.getElementById('sendButton');
    sendButton.disabled = true;
    sendButton.innerHTML = '<span>Analyzing...</span>';
    
    try {
        // Show loading
        const loadingId = addMessage('assistant', '', true);
        
        try {
            // Send to API
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    log_content: message
                })
            });
            
            const data = await response.json();
            
            // Remove loading
            removeMessage(loadingId);
            
            if (data.success) {
                // Update conversation ID if provided
                if (data.conversation_id) {
                    currentConversationId = data.conversation_id;
                }
                
                // Format and display response
                const formattedResponse = formatMarkdown(data.response);
                addMessage('assistant', formattedResponse);
                
                // Reload conversations to update list
                loadConversations();
            } else {
                addMessage('assistant', `❌ Error: ${data.error}`);
            }
        } catch (error) {
            removeMessage(loadingId);
            addMessage('assistant', `❌ Error: ${error.message}`);
            throw error;
        }
    } finally {
        sendButton.disabled = false;
        sendButton.innerHTML = '<span>Send</span><svg width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M2 10L18 2L12 18L10 10L2 10Z" fill="currentColor"/></svg>';
        messageInput.focus();
    }
}

// Handle file upload
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Clear welcome message
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    // Show file info
    const fileInfo = `
        <div class="file-info">
            <span class="file-info-icon">📎</span>
            <span class="file-info-name">${file.name}</span>
            <span class="file-info-size">${formatFileSize(file.size)}</span>
        </div>
    `;
    addMessage('user', fileInfo);
    
    // Disable send button
    const sendButton = document.getElementById('sendButton');
    sendButton.disabled = true;
    sendButton.innerHTML = '<span>Analyzing...</span>';
    
    try {
        // Show loading
        const loadingId = addMessage('assistant', '', true);
        
        // Upload file
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        // Remove loading
        removeMessage(loadingId);
        
        if (data.success) {
            // Update conversation ID if provided
            if (data.conversation_id) {
                currentConversationId = data.conversation_id;
            }
            
            // Format and display response
            const formattedResponse = formatMarkdown(data.response);
            addMessage('assistant', formattedResponse);
            
            // Reload conversations to update list
            loadConversations();
        } else {
            addMessage('assistant', `❌ Error: ${data.error}`);
        }
    } catch (error) {
        const loadingMsg = document.querySelector('.message.assistant .loading');
        if (loadingMsg) {
            loadingMsg.closest('.message').remove();
        }
        addMessage('assistant', `❌ Error: ${error.message}`);
    } finally {
        sendButton.disabled = false;
        sendButton.innerHTML = '<span>Send</span><svg width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M2 10L18 2L12 18L10 10L2 10Z" fill="currentColor"/></svg>';
        // Reset file input
        event.target.value = '';
    }
}

// Add message to chat
function addMessage(role, content, isLoading = false) {
    const chatMessages = document.getElementById('chatMessages');
    const messageId = 'msg-' + Date.now();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.id = messageId;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'U' : 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (isLoading) {
        contentDiv.innerHTML = '<div class="loading"><div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div></div>';
    } else {
        contentDiv.innerHTML = content;
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

// Remove message
function removeMessage(messageId) {
    const message = document.getElementById(messageId);
    if (message) {
        message.remove();
    }
}

// Format markdown-like text
function formatMarkdown(text) {
    if (!text) return '';
    
    // Convert code blocks first (before other processing)
    text = text.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    
    // Convert inline code
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Convert markdown headers
    text = text.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    
    // Convert bold
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Convert lists - handle multiple list items
    const lines = text.split('\n');
    let inList = false;
    let result = [];
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        
        if (line.match(/^[-*]\s+/)) {
            if (!inList) {
                result.push('<ul>');
                inList = true;
            }
            const content = line.replace(/^[-*]\s+/, '');
            result.push(`<li>${content}</li>`);
        } else {
            if (inList) {
                result.push('</ul>');
                inList = false;
            }
            if (line) {
                result.push(line);
            }
        }
    }
    
    if (inList) {
        result.push('</ul>');
    }
    
    text = result.join('\n');
    
    // Convert remaining line breaks to <br> (but not inside code/pre)
    text = text.split('\n').map(line => {
        if (line.trim().startsWith('<pre>') || line.trim().startsWith('<code>') || 
            line.trim().startsWith('</pre>') || line.trim().startsWith('</code>') ||
            line.trim().startsWith('<h2>') || line.trim().startsWith('<h3>') ||
            line.trim().startsWith('<ul>') || line.trim().startsWith('</ul>') ||
            line.trim().startsWith('<li>') || line.trim().startsWith('</li>')) {
            return line;
        }
        return line ? line + '<br>' : '';
    }).join('\n');
    
    // Clean up multiple <br> tags
    text = text.replace(/(<br>\s*){3,}/g, '<br><br>');
    
    return text;
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Load conversations list
async function loadConversations() {
    try {
        const response = await fetch('/api/chat/conversations');
        const data = await response.json();
        
        const conversationsItems = document.getElementById('conversationsItems');
        
        if (data.success && data.conversations && data.conversations.length > 0) {
            conversationsItems.innerHTML = '';
            
            data.conversations.forEach(conv => {
                const item = document.createElement('div');
                item.className = 'conversation-item';
                if (conv.id === currentConversationId) {
                    item.classList.add('active');
                }
                
                const time = new Date(conv.last_updated).toLocaleDateString();
                const icon = conv.log_filename ? '📎' : '💬';
                
                item.innerHTML = `
                    <span class="conversation-item-icon">${icon}</span>
                    <div class="conversation-item-content">
                        <div class="conversation-item-title">${escapeHtml(conv.title)}</div>
                        <div class="conversation-item-time">${time}</div>
                    </div>
                    <span class="conversation-item-delete" onclick="event.stopPropagation(); deleteConversation('${conv.id}')">×</span>
                `;
                
                item.onclick = () => loadConversation(conv.id);
                conversationsItems.appendChild(item);
            });
        } else {
            conversationsItems.innerHTML = '<div class="conversation-empty">No conversations yet</div>';
        }
    } catch (error) {
        console.error('Failed to load conversations:', error);
        const conversationsItems = document.getElementById('conversationsItems');
        if (conversationsItems) {
            conversationsItems.innerHTML = '<div class="conversation-empty">Failed to load</div>';
        }
    }
}

// Load a specific conversation
async function loadConversation(conversationId) {
    try {
        currentConversationId = conversationId;
        
        // Update active state
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.classList.remove('active');
        });
        
        // Show loading
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = '<div class="welcome-message"><p>Loading conversation...</p></div>';
        
        const response = await fetch(`/api/chat/conversation/${conversationId}`);
        const data = await response.json();
        
        chatMessages.innerHTML = '';
        
        if (data.success && data.messages && data.messages.length > 0) {
            // Display messages (oldest first)
            const reversedMessages = [...data.messages].reverse();
            for (const item of reversedMessages) {
                addMessage('user', item.user_message);
                addMessage('assistant', formatMarkdown(item.assistant_response));
            }
            
            chatMessages.scrollTop = chatMessages.scrollHeight;
        } else {
            showWelcomeMessage();
        }
        
        // Reload conversations to update active state
        loadConversations();
    } catch (error) {
        console.error('Failed to load conversation:', error);
        showWelcomeMessage();
    }
}

// Start new conversation
async function startNewConversation() {
    try {
        const response = await fetch('/api/chat/new', { method: 'POST' });
        const data = await response.json();
        
        if (data.success) {
            currentConversationId = data.conversation_id;
            showWelcomeMessage();
        }
    } catch (error) {
        console.error('Failed to start new conversation:', error);
    }
}

// Delete conversation
async function deleteConversation(conversationId) {
    if (!confirm('Delete this conversation?')) {
        return;
    }
    
    // TODO: Implement delete endpoint
    console.log('Delete conversation:', conversationId);
    loadConversations();
    
    if (conversationId === currentConversationId) {
        startNewConversation();
    }
}

// Show welcome message
function showWelcomeMessage() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <h3>👋 Welcome to Log Analyzer</h3>
            <p>Upload a log file or paste log content to get started with AI-powered analysis.</p>
            <div class="welcome-features">
                <div class="feature-card">
                    <span class="feature-icon">🎯</span>
                    <h4>Root Cause Analysis</h4>
                    <p>Identifies primary failures</p>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">🤖</span>
                    <h4>ML-Powered Detection</h4>
                    <p>Multiple AI models working together</p>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">⚡</span>
                    <h4>Real-Time Analysis</h4>
                    <p>Instant results</p>
                </div>
            </div>
        </div>
    `;
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// New chat
async function newChat() {
    await startNewConversation();
    showWelcomeMessage();
    document.getElementById('messageInput').value = '';
    document.getElementById('messageInput').focus();
    loadConversations();
}

